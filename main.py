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
import io
import os
from typing import Optional

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
    )
    .run_function(_download_model)
)

# Module-level imports available inside the container only.
# This lets FastAPI resolve UploadFile/Header/etc. via get_type_hints().
with image.imports():
    import requests
    import torch
    from fastapi import FastAPI, File, Header, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
    from PIL import Image, UnidentifiedImageError
    from pydantic import BaseModel, HttpUrl
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu="L4",
    scaledown_window=60,
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

    def _check_format(self, fmt: str) -> str:
        fmt = fmt.lower()
        if fmt not in {"png", "webp"}:
            raise HTTPException(400, "format must be 'png' or 'webp'")
        return fmt

    def _open_image(self, data: bytes):
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit")
        try:
            image_obj = Image.open(io.BytesIO(data))
            image_obj.load()
            return image_obj
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "Invalid or unsupported image")

    def _remove(self, image_obj):
        original_size = image_obj.size
        rgb = image_obj.convert("RGB")
        tensor = self.transform(rgb).unsqueeze(0).to("cuda").half()
        with torch.no_grad():
            preds = self.model(tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze().float()
        mask = self.to_pil(pred).resize(original_size)
        result = rgb.convert("RGBA")
        result.putalpha(mask)
        return result

    def _encode(self, image_out, fmt: str) -> bytes:
        buf = io.BytesIO()
        image_out.save(buf, format="PNG" if fmt == "png" else "WEBP", optimize=True)
        return buf.getvalue()

    def _response(self, image_out, fmt: str):
        content = self._encode(image_out, fmt)
        media_type = "image/png" if fmt == "png" else "image/webp"
        return Response(content=content, media_type=media_type)

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

        @web.get("/")
        def root():
            return {
                "name": "useknockout",
                "version": "0.1.0",
                "endpoints": ["POST /remove", "POST /remove-url", "GET /health"],
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

        return web
