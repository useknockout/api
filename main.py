"""
useknockout — state-of-the-art background removal API.

Powered by BiRefNet (MIT license, commercial-safe), served on Modal GPUs.

Deploy:
    modal deploy main.py

Test (multipart file upload):
    curl -X POST "$URL/remove" \
      -H "Authorization: Bearer $API_TOKEN" \
      -F "file=@cat.jpg" \
      -o cat-nobg.png

Test (remote URL):
    curl -X POST "$URL/remove-url" \
      -H "Authorization: Bearer $API_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"url":"https://example.com/cat.jpg"}' \
      -o cat-nobg.png
"""
import base64
import io
import os
from typing import List, Optional

import modal

APP_NAME = "api"
MODEL_REPO = "ZhengPeng7/BiRefNet"
MODEL_INPUT_SIZE = (1024, 1024)
MAX_IMAGE_BYTES = 25 * 1024 * 1024  # 25 MB


def _download_model() -> None:
    """Bake weights into the image at build time so cold starts are fast."""
    from transformers import AutoModelForImageSegmentation

    AutoModelForImageSegmentation.from_pretrained(
        MODEL_REPO, trust_remote_code=True
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.44.2",
        "pillow==10.4.0",
        "timm==1.0.9",
        "kornia==0.7.3",
        "einops==0.8.0",
        "huggingface_hub==0.24.6",
        "fastapi[standard]==0.115.0",
        "python-multipart==0.0.9",
        "requests==2.32.3",
        "pydantic==2.9.2",
        "numpy==1.26.4",
        "pymatting==1.1.12",
    )
    .run_function(_download_model)
)

# Module-level imports available inside the container only.
# This lets FastAPI resolve UploadFile/Header/etc. via get_type_hints().
with image.imports():
    import numpy as np
    import requests
    import torch
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
    from PIL import Image, UnidentifiedImageError
    from pydantic import BaseModel, HttpUrl
    from pymatting import estimate_foreground_ml
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu="L4",
    scaledown_window=300,  # keep warm 5 min between requests
    timeout=600,
    max_containers=10,
    secrets=[modal.Secret.from_name("knockout-secrets")],
)
class Knockout:
    @modal.enter()
    def load(self) -> None:
        torch.set_float32_matmul_precision("high")

        self.model = AutoModelForImageSegmentation.from_pretrained(
            MODEL_REPO, trust_remote_code=True
        )
        self.model.to("cuda").eval().half()

        self.transform = transforms.Compose([
            transforms.Resize(MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.to_pil = transforms.ToPILImage()

    def _check_auth(self, authorization: Optional[str]) -> None:
        raw = os.environ.get("API_TOKEN", "").strip()
        if not raw:
            return  # no tokens configured → open API (dev only)

        valid = {t.strip() for t in raw.split(",") if t.strip()}
        if not valid:
            return

        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        presented = authorization.split(" ", 1)[1].strip()
        if presented not in valid:
            raise HTTPException(status_code=403, detail="Invalid token")

    def _check_format(self, fmt: str, allowed=frozenset({"png", "webp"})) -> str:
        fmt = fmt.lower()
        if fmt == "jpeg":
            fmt = "jpg"
        if fmt not in allowed:
            raise HTTPException(400, f"format must be one of {sorted(allowed)}")
        return fmt

    def _parse_color(self, hex_color: str):
        h = hex_color.strip().lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) != 6 or not all(c in "0123456789abcdefABCDEF" for c in h):
            raise HTTPException(400, f"Invalid hex color: {hex_color!r}")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    def _open_image(self, data: bytes):
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit")
        try:
            image_obj = Image.open(io.BytesIO(data))
            image_obj.load()
            return image_obj
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "Invalid or unsupported image")

    def _get_mask(self, image_obj):
        """Run BiRefNet on an RGB image, return (rgb_image, mask_pil)."""
        rgb = image_obj.convert("RGB")
        tensor = self.transform(rgb).unsqueeze(0).to("cuda").half()
        with torch.no_grad():
            preds = self.model(tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze().float()
        mask = self.to_pil(pred).resize(rgb.size)
        return rgb, mask

    def _clean_foreground(self, rgb: "Image.Image", mask: "Image.Image") -> "Image.Image":
        """
        Estimate pure foreground RGB at mask edges using closed-form matting.
        Eliminates color spill / halo from the original background.

        Skipped when FOREGROUND_REFINE=false. Downscaled internally to keep
        compute bounded (pymatting is O(N) but heavy at 4K+).
        """
        if os.environ.get("FOREGROUND_REFINE", "true").strip().lower() in {"false", "0", "no", "off"}:
            return rgb

        w, h = rgb.size
        max_dim = 1024  # matting scales ~linearly; cap work to keep latency low
        scale = min(1.0, max_dim / max(w, h))

        if scale < 1.0:
            sw, sh = int(round(w * scale)), int(round(h * scale))
            rgb_small = rgb.resize((sw, sh), Image.LANCZOS)
            mask_small = mask.resize((sw, sh), Image.LANCZOS)
        else:
            rgb_small = rgb
            mask_small = mask

        fg_arr = np.asarray(rgb_small, dtype=np.float32) / 255.0
        alpha_arr = np.asarray(mask_small.convert("L"), dtype=np.float32) / 255.0

        try:
            clean = estimate_foreground_ml(fg_arr, alpha_arr)
            clean_u8 = np.clip(clean * 255.0, 0.0, 255.0).astype(np.uint8)
            clean_img = Image.fromarray(clean_u8, mode="RGB")
            if scale < 1.0:
                clean_img = clean_img.resize((w, h), Image.LANCZOS)
            return clean_img
        except Exception:
            # Any pymatting failure → degrade gracefully to raw RGB
            return rgb

    def _remove(self, image_obj):
        rgb, mask = self._get_mask(image_obj)
        clean_rgb = self._clean_foreground(rgb, mask)
        result = clean_rgb.convert("RGBA")
        result.putalpha(mask)
        return result

    def _composite_on_bg(self, image_obj, bg_image_or_color):
        """Composite foreground onto a solid color or image background."""
        rgb, mask = self._get_mask(image_obj)
        clean_rgb = self._clean_foreground(rgb, mask)
        if isinstance(bg_image_or_color, tuple):
            bg = Image.new("RGB", rgb.size, bg_image_or_color)
        else:
            bg = bg_image_or_color.convert("RGB").resize(rgb.size, Image.LANCZOS)
        bg.paste(clean_rgb, (0, 0), mask)
        return bg

    _FORMAT_TO_PIL = {"png": "PNG", "webp": "WEBP", "jpg": "JPEG"}
    _FORMAT_TO_MEDIA = {"png": "image/png", "webp": "image/webp", "jpg": "image/jpeg"}

    def _encode(self, image_out, fmt: str) -> bytes:
        buf = io.BytesIO()
        pil_fmt = self._FORMAT_TO_PIL[fmt]
        save_kwargs = {"optimize": True}
        if pil_fmt == "JPEG":
            save_kwargs["quality"] = 92
            if image_out.mode != "RGB":
                image_out = image_out.convert("RGB")
        image_out.save(buf, format=pil_fmt, **save_kwargs)
        return buf.getvalue()

    def _response(self, image_out, fmt: str):
        content = self._encode(image_out, fmt)
        return Response(content=content, media_type=self._FORMAT_TO_MEDIA[fmt])

    @modal.asgi_app(label="api")
    def fastapi_app(self):
        web = FastAPI(
            title="useknockout",
            description="State-of-the-art background removal API.",
            version="0.1.0",
        )

        web.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["POST", "GET"],
            allow_headers=["*"],
        )

        class UrlBody(BaseModel):
            url: HttpUrl
            format: str = "png"

        class BatchUrlBody(BaseModel):
            urls: List[HttpUrl]
            format: str = "png"

        @web.get("/")
        def root():
            return {
                "name": "useknockout",
                "version": "0.2.0",
                "endpoints": [
                    "POST /remove",
                    "POST /remove-url",
                    "POST /replace-bg",
                    "POST /remove-batch",
                    "POST /remove-batch-url",
                    "GET /health",
                ],
                "docs": "/docs",
            }

        @web.get("/health")
        def health():
            return {"status": "ok", "model": MODEL_REPO}

        @web.get("/remove")
        def remove_info():
            return {
                "error": "Use POST with multipart/form-data",
                "method": "POST",
                "body": "multipart form with field 'file'",
                "headers": {"Authorization": "Bearer <token>"},
                "example_curl": "curl -X POST https://useknockout--api.modal.run/remove -H 'Authorization: Bearer <token>' -F 'file=@image.jpg' -o out.png",
                "docs": "https://useknockout--api.modal.run/docs",
                "sdk": "npm i @useknockout/node",
            }

        @web.get("/remove-url")
        def remove_url_info():
            return {
                "error": "Use POST with JSON body",
                "method": "POST",
                "body": {"url": "https://example.com/image.jpg", "format": "png"},
                "headers": {"Authorization": "Bearer <token>", "Content-Type": "application/json"},
                "docs": "https://useknockout--api.modal.run/docs",
                "sdk": "npm i @useknockout/node",
            }

        @web.post("/remove")
        def remove_endpoint(
            file: UploadFile = File(...),
            format: str = "png",
            authorization: Optional[str] = Header(default=None),
        ):
            self._check_auth(authorization)
            fmt = self._check_format(format)
            data = file.file.read()
            image_obj = self._open_image(data)
            result = self._remove(image_obj)
            return self._response(result, fmt)

        @web.post("/remove-url")
        def remove_url_endpoint(
            body: UrlBody,
            authorization: Optional[str] = Header(default=None),
        ):
            self._check_auth(authorization)
            fmt = self._check_format(body.format)

            try:
                resp = requests.get(str(body.url), timeout=15)
                resp.raise_for_status()
            except requests.RequestException as e:
                raise HTTPException(400, f"Could not fetch image: {e}")

            image_obj = self._open_image(resp.content)
            result = self._remove(image_obj)
            return self._response(result, fmt)

        @web.post("/replace-bg")
        def replace_bg_endpoint(
            file: UploadFile = File(...),
            bg_color: str = Form("#FFFFFF"),
            bg_url: Optional[str] = Form(None),
            format: str = Form("png"),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Remove the background and composite the subject onto a new background.

            Provide either:
              - bg_color: hex color (default #FFFFFF). Examples: "#000000", "#ff5733".
              - bg_url: URL of a background image (takes precedence over bg_color).

            Output is opaque (no alpha). Use `format=jpg` for smallest file size.
            """
            self._check_auth(authorization)
            fmt = self._check_format(format, allowed=frozenset({"png", "webp", "jpg"}))

            data = file.file.read()
            fg = self._open_image(data)

            if bg_url:
                try:
                    bg_resp = requests.get(bg_url, timeout=15)
                    bg_resp.raise_for_status()
                    bg = self._open_image(bg_resp.content)
                    composited = self._composite_on_bg(fg, bg)
                except requests.RequestException as e:
                    raise HTTPException(400, f"Could not fetch bg_url: {e}")
            else:
                color = self._parse_color(bg_color)
                composited = self._composite_on_bg(fg, color)

            return self._response(composited, fmt)

        @web.post("/remove-batch")
        def remove_batch_endpoint(
            files: List[UploadFile] = File(...),
            format: str = "png",
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Remove backgrounds from up to 10 images in one call.

            Returns JSON: {"count": N, "results": [{filename, success, format, data_base64 | error}]}.
            """
            self._check_auth(authorization)
            fmt = self._check_format(format)

            if len(files) > 10:
                raise HTTPException(400, "Max 10 images per batch")
            if not files:
                raise HTTPException(400, "At least 1 file required")

            results = []
            for upload in files:
                item = {"filename": upload.filename}
                try:
                    data = upload.file.read()
                    image_obj = self._open_image(data)
                    out = self._remove(image_obj)
                    content = self._encode(out, fmt)
                    item.update({
                        "success": True,
                        "format": fmt,
                        "size_bytes": len(content),
                        "data_base64": base64.b64encode(content).decode("ascii"),
                    })
                except HTTPException as he:
                    item.update({"success": False, "error": he.detail})
                except Exception as e:
                    item.update({"success": False, "error": str(e)})
                results.append(item)

            return {"count": len(results), "format": fmt, "results": results}

        @web.post("/remove-batch-url")
        def remove_batch_url_endpoint(
            body: BatchUrlBody,
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Remove backgrounds from up to 10 remote images in one call.

            Body: {"urls": ["https://...", ...], "format": "png" | "webp"}
            """
            self._check_auth(authorization)
            fmt = self._check_format(body.format)

            if len(body.urls) > 10:
                raise HTTPException(400, "Max 10 urls per batch")
            if not body.urls:
                raise HTTPException(400, "At least 1 url required")

            results = []
            for url in body.urls:
                url_str = str(url)
                item = {"url": url_str}
                try:
                    resp = requests.get(url_str, timeout=15)
                    resp.raise_for_status()
                    image_obj = self._open_image(resp.content)
                    out = self._remove(image_obj)
                    content = self._encode(out, fmt)
                    item.update({
                        "success": True,
                        "format": fmt,
                        "size_bytes": len(content),
                        "data_base64": base64.b64encode(content).decode("ascii"),
                    })
                except HTTPException as he:
                    item.update({"success": False, "error": he.detail})
                except requests.RequestException as re:
                    item.update({"success": False, "error": f"fetch failed: {re}"})
                except Exception as e:
                    item.update({"success": False, "error": str(e)})
                results.append(item)

            return {"count": len(results), "format": fmt, "results": results}

        return web
