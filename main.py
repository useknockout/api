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

# Swin2SR — SwinV2 Transformer super-res (successor to SwinIR). Apache-2.0.
# Better than Real-ESRGAN on real photos: preserves skin/hair texture instead
# of the painted/plastic look Real-ESRGAN produces on faces.
SWIN2SR_X4_REPO = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
SWIN2SR_X2_REPO = "caidas/swin2SR-classical-sr-x2-64"


UPSCALE_WEIGHTS_DIR = "/root/weights"
REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)
GFPGAN_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
)
FACEXLIB_DETECTION_URL = (
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"
)
FACEXLIB_PARSING_URL = (
    "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"
)


def _download_model() -> None:
    """Bake all model weights into the image at build time so cold starts are fast."""
    import os
    import urllib.request

    from transformers import (
        AutoImageProcessor,
        AutoModelForImageSegmentation,
        Swin2SRForImageSuperResolution,
    )

    AutoModelForImageSegmentation.from_pretrained(MODEL_REPO, trust_remote_code=True)

    # Bake Swin2SR weights into image so cold starts skip the HF download.
    for repo in (SWIN2SR_X4_REPO, SWIN2SR_X2_REPO):
        Swin2SRForImageSuperResolution.from_pretrained(repo)
        AutoImageProcessor.from_pretrained(repo)

    os.makedirs(UPSCALE_WEIGHTS_DIR, exist_ok=True)

    # Real-ESRGAN + GFPGAN main weights — explicit paths used at load time.
    direct_downloads = {
        "RealESRGAN_x4plus.pth": REALESRGAN_URL,
        "GFPGANv1.4.pth": GFPGAN_URL,
    }
    for name, url in direct_downloads.items():
        dest = os.path.join(UPSCALE_WEIGHTS_DIR, name)
        if not os.path.exists(dest):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, dest)

    # facexlib auto-downloads detection + parsing weights into gfpgan/weights/.
    # Pre-bake them so the first /face-restore request doesn't pay the network cost.
    import gfpgan as _gfpgan_mod

    gfpgan_weights_dir = os.path.join(os.path.dirname(_gfpgan_mod.__file__), "weights")
    os.makedirs(gfpgan_weights_dir, exist_ok=True)
    facexlib_downloads = {
        "detection_Resnet50_Final.pth": FACEXLIB_DETECTION_URL,
        "parsing_parsenet.pth": FACEXLIB_PARSING_URL,
    }
    for name, url in facexlib_downloads.items():
        dest = os.path.join(gfpgan_weights_dir, name)
        if not os.path.exists(dest):
            print(f"Downloading {name} -> gfpgan/weights/...")
            urllib.request.urlretrieve(url, dest)


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
        "opencv-python-headless==4.10.0.84",
    )
    .pip_install(
        "basicsr==1.4.2",
        "facexlib==0.3.0",
        "realesrgan==0.3.0",
        "gfpgan==1.3.8",
    )
    # basicsr install bumps numpy to 2.x — pin back to 1.26.4 to keep pymatting + PIL stable.
    .pip_install("numpy==1.26.4")
    # basicsr 1.4.2 + facexlib import `torchvision.transforms.functional_tensor`,
    # removed in torchvision 0.17+. Patch every file in site-packages that
    # references it. Uses grep to find files (no Python import — would crash).
    # Then nuke __pycache__ so stale .pyc bytecode doesn't shadow the new .py.
    .run_commands(
        "grep -rl 'torchvision.transforms.functional_tensor' "
        "/usr/local/lib/python3.11/site-packages/ "
        "| xargs --no-run-if-empty "
        "sed -i 's/torchvision.transforms.functional_tensor/torchvision.transforms.functional/g'",
        "find /usr/local/lib/python3.11/site-packages/ -type d -name __pycache__ "
        "-exec rm -rf {} + 2>/dev/null; true"
    )
    .run_function(_download_model)
)

# Module-level imports available inside the container only.
# This lets FastAPI resolve UploadFile/Header/etc. via get_type_hints().
with image.imports():
    import numpy as np
    import requests
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
    from gfpgan import GFPGANer
    from PIL import Image, ImageDraw, ImageFilter, UnidentifiedImageError
    from pydantic import BaseModel, HttpUrl
    from pymatting import estimate_foreground_cf, estimate_foreground_ml
    from realesrgan import RealESRGANer
    from torchvision import transforms
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageSegmentation,
        Swin2SRForImageSuperResolution,
    )

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

        # Real-ESRGAN x4 upscaler. Tile inference keeps VRAM bounded for big inputs.
        rrdb = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
        self.upscaler = RealESRGANer(
            scale=4,
            model_path=f"{UPSCALE_WEIGHTS_DIR}/RealESRGAN_x4plus.pth",
            model=rrdb,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=True,
            gpu_id=0,
        )

        # GFPGAN portrait restorer — two variants:
        #   face_restorer       → original bg preserved (no Real-ESRGAN bg pass)
        #                         avoids skin-tone bleed into bg around face edges
        #   face_restorer_full  → bg also upscaled via Real-ESRGAN (legacy v0.5.0 behavior)
        self.face_restorer = GFPGANer(
            model_path=f"{UPSCALE_WEIGHTS_DIR}/GFPGANv1.4.pth",
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        self.face_restorer_full = GFPGANer(
            model_path=f"{UPSCALE_WEIGHTS_DIR}/GFPGANv1.4.pth",
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=self.upscaler,
        )

        # Swin2SR — default upscaler. Better photo quality than Real-ESRGAN
        # (which is trained heavily on synthetic/anime and produces a painted
        # look on real photos). x4 = real-world BSRGAN-PSNR weights, x2 = classical.
        self.swin2sr_x4 = Swin2SRForImageSuperResolution.from_pretrained(
            SWIN2SR_X4_REPO
        ).to("cuda").eval().half()
        self.swin2sr_x2 = Swin2SRForImageSuperResolution.from_pretrained(
            SWIN2SR_X2_REPO
        ).to("cuda").eval().half()
        self.swin2sr_proc_x4 = AutoImageProcessor.from_pretrained(SWIN2SR_X4_REPO)
        self.swin2sr_proc_x2 = AutoImageProcessor.from_pretrained(SWIN2SR_X2_REPO)

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
        self._bump_counter()
        return rgb, mask

    def _bump_counter(self) -> None:
        """Increment public processed-image counter. Never raises."""
        try:
            from datetime import datetime
            stats = modal.Dict.from_name("knockout-stats", create_if_missing=True)
            today = datetime.utcnow().strftime("%Y-%m-%d")
            stats["total"] = int(stats.get("total", 0)) + 1
            stats[f"day:{today}"] = int(stats.get(f"day:{today}", 0)) + 1
        except Exception:
            pass  # counter is best-effort; never block processing

    def _swin2sr_upscale(self, image_obj, scale: int):
        """
        Swin2SR super-resolution with tiled inference + linear-blend overlap.

        Swin2SR has no built-in tiling, so we slice the input into overlapping
        tiles, run each through the model, and blend overlap regions to hide
        seams. Tile size = 256 (32x window_size=8), overlap = 32 px.

        scale: 2 or 4. Picks the matching pretrained Swin2SR variant.
        Returns: PIL.Image RGB at (W*scale, H*scale).
        """
        if scale not in (2, 4):
            raise HTTPException(400, "scale must be 2 or 4")

        model = self.swin2sr_x4 if scale == 4 else self.swin2sr_x2

        rgb = image_obj.convert("RGB")
        src = np.asarray(rgb, dtype=np.float32) / 255.0  # H, W, 3
        h, w, _ = src.shape

        tile = 256
        overlap = 32
        step = tile - overlap
        out_h, out_w = h * scale, w * scale
        accum = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight = np.zeros((out_h, out_w, 1), dtype=np.float32)

        # Pre-compute 1-D triangular blend window (peak in center → seamless overlap).
        def _blend_window(length: int, ov: int) -> np.ndarray:
            win = np.ones(length, dtype=np.float32)
            ramp = np.linspace(0.0, 1.0, ov, endpoint=False, dtype=np.float32)
            win[:ov] = ramp
            win[-ov:] = ramp[::-1]
            return win

        # Iterate tiles. Last tile snaps to edge so we cover the right/bottom border.
        ys = list(range(0, max(1, h - overlap), step))
        if ys[-1] + tile < h:
            ys.append(h - tile if h > tile else 0)
        xs = list(range(0, max(1, w - overlap), step))
        if xs[-1] + tile < w:
            xs.append(w - tile if w > tile else 0)

        for y in ys:
            for x in xs:
                ty = max(0, min(y, max(0, h - tile)))
                tx = max(0, min(x, max(0, w - tile)))
                tile_h = min(tile, h - ty)
                tile_w = min(tile, w - tx)
                tile_arr = src[ty:ty + tile_h, tx:tx + tile_w, :]

                tile_pil = Image.fromarray(
                    np.clip(tile_arr * 255.0, 0, 255).astype(np.uint8)
                )
                processor = self.swin2sr_proc_x4 if scale == 4 else self.swin2sr_proc_x2
                inputs = processor(tile_pil, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to("cuda").half()

                with torch.no_grad():
                    output = model(pixel_values=pixel_values).reconstruction

                # output: 1, 3, h*scale_padded, w*scale_padded — crop to expected size.
                arr = output.squeeze(0).clamp_(0, 1).float().cpu().numpy()
                arr = np.transpose(arr, (1, 2, 0))[: tile_h * scale, : tile_w * scale, :]

                ah, aw = arr.shape[:2]
                ov = overlap * scale
                wy = _blend_window(ah, min(ov, ah // 2)) if ah > ov else np.ones(ah, dtype=np.float32)
                wx = _blend_window(aw, min(ov, aw // 2)) if aw > ov else np.ones(aw, dtype=np.float32)
                window = (wy[:, None] * wx[None, :])[:, :, None]

                oy, ox = ty * scale, tx * scale
                accum[oy:oy + ah, ox:ox + aw, :] += arr * window
                weight[oy:oy + ah, ox:ox + aw, :] += window

        result = accum / np.clip(weight, 1e-6, None)
        result_u8 = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(result_u8, mode="RGB")

    def _pil_to_bgr(self, image_obj):
        """PIL Image (any mode) → contiguous BGR uint8 ndarray expected by Real-ESRGAN/GFPGAN."""
        rgb = np.array(image_obj.convert("RGB"))
        return np.ascontiguousarray(rgb[:, :, ::-1])

    def _bgr_to_pil(self, bgr_arr):
        """BGR uint8 ndarray → RGB PIL Image."""
        return Image.fromarray(bgr_arr[:, :, ::-1])

    def _clean_foreground(self, rgb: "Image.Image", mask: "Image.Image", *, fast: bool = False) -> "Image.Image":
        """
        Estimate pure foreground RGB at mask edges using closed-form matting.
        Eliminates color spill / halo from the original background.

        Skipped when FOREGROUND_REFINE=false. Downscaled internally to keep
        compute bounded (closed-form is O(N²) but solver is sparse).
        """
        if os.environ.get("FOREGROUND_REFINE", "true").strip().lower() in {"false", "0", "no", "off"}:
            return rgb

        w, h = rgb.size
        max_dim = 1024 if fast else 2048
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
            estimator = estimate_foreground_ml if fast else estimate_foreground_cf
            clean = estimator(fg_arr, alpha_arr)
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

    def _bounding_box(self, mask, threshold: int = 10):
        """Find tight (left, top, right, bottom) bounding box of mask pixels above threshold."""
        arr = np.asarray(mask.convert("L"))
        rows = np.any(arr > threshold, axis=1)
        cols = np.any(arr > threshold, axis=0)
        if not rows.any() or not cols.any():
            return None
        top = int(np.argmax(rows))
        bottom = int(len(rows) - np.argmax(rows[::-1]))
        left = int(np.argmax(cols))
        right = int(len(cols) - np.argmax(cols[::-1]))
        return (left, top, right, bottom)

    def _dilate_mask(self, mask, radius: int):
        """Expand mask by `radius` pixels (integer). Used for stroke/outline effects."""
        if radius <= 0:
            return mask
        # Odd-sized window for PIL MaxFilter
        size = radius * 2 + 1
        return mask.filter(ImageFilter.MaxFilter(size))

    def _checkerboard(self, size, square: int = 16, a=(230, 230, 230), b=(255, 255, 255)):
        """Generate a checkerboard RGB image matching `size` = (w, h). Used for /compare preview."""
        w, h = size
        img = Image.new("RGB", (w, h), a)
        draw = ImageDraw.Draw(img)
        for y in range(0, h, square):
            for x in range(0, w, square):
                if ((x // square) + (y // square)) % 2 == 0:
                    draw.rectangle([x, y, x + square - 1, y + square - 1], fill=b)
        return img

    def _composite_shadow(self, cutout_rgba, mask, bg, offset=(8, 12), blur=14, opacity=0.45,
                         shadow_color=(0, 0, 0)):
        """Add drop shadow under cutout then paste on bg. cutout_rgba is the alpha cutout."""
        w, h = bg.size
        shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        # Shadow = mask, offset, blurred, tinted
        shadow_mask = mask.convert("L").filter(ImageFilter.GaussianBlur(radius=blur))
        shadow_alpha_val = int(round(opacity * 255))
        shadow_rgb = Image.new("RGB", (w, h), shadow_color)
        shadow_full = Image.merge("RGBA", (*shadow_rgb.split(), shadow_mask.point(lambda p: min(p, shadow_alpha_val))))
        shadow_layer.alpha_composite(shadow_full, dest=offset)
        out = bg.convert("RGBA")
        out.alpha_composite(shadow_layer)
        out.alpha_composite(cutout_rgba)
        return out.convert("RGB") if bg.mode == "RGB" else out

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
            description="State-of-the-art background removal + upscaling API.",
            version="0.6.0",
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

        class EstimateBody(BaseModel):
            endpoint: str
            width: int
            height: int

        @web.get("/")
        def root():
            return {
                "name": "useknockout",
                "version": "0.6.0",
                "endpoints": [
                    "POST /remove",
                    "POST /remove-url",
                    "POST /replace-bg",
                    "POST /remove-batch",
                    "POST /remove-batch-url",
                    "POST /mask",
                    "POST /smart-crop",
                    "POST /shadow",
                    "POST /sticker",
                    "POST /outline",
                    "POST /studio-shot",
                    "POST /compare",
                    "POST /headshot",
                    "POST /preview",
                    "POST /upscale",
                    "POST /face-restore",
                    "POST /estimate",
                    "GET /stats",
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

        @web.post("/mask")
        def mask_endpoint(
            file: UploadFile = File(...),
            format: str = Form("png"),
            authorization: Optional[str] = Header(default=None),
        ):
            """Return just the alpha mask as a grayscale PNG/WebP (0 = bg, 255 = subject)."""
            self._check_auth(authorization)
            fmt = self._check_format(format)
            data = file.file.read()
            image_obj = self._open_image(data)
            _, mask = self._get_mask(image_obj)
            return self._response(mask.convert("L"), fmt)

        @web.post("/smart-crop")
        def smart_crop_endpoint(
            file: UploadFile = File(...),
            padding: int = Form(24),
            transparent: bool = Form(True),
            format: str = Form("png"),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Auto-crop to the subject's tight bounding box + padding (pixels).

            `transparent=true` (default): return cropped cutout with transparent background.
            `transparent=false`: return cropped region from the original image (bg preserved).
            """
            self._check_auth(authorization)
            allowed = frozenset({"png", "webp", "jpg"}) if not transparent else frozenset({"png", "webp"})
            fmt = self._check_format(format, allowed=allowed)
            data = file.file.read()
            image_obj = self._open_image(data)
            rgb, mask = self._get_mask(image_obj)

            bbox = self._bounding_box(mask)
            if bbox is None:
                raise HTTPException(400, "No subject detected in image")

            left, top, right, bottom = bbox
            pad = max(0, int(padding))
            w, h = rgb.size
            left = max(0, left - pad)
            top = max(0, top - pad)
            right = min(w, right + pad)
            bottom = min(h, bottom + pad)

            if transparent:
                clean_rgb = self._clean_foreground(rgb, mask)
                cutout = clean_rgb.convert("RGBA")
                cutout.putalpha(mask)
                cropped = cutout.crop((left, top, right, bottom))
            else:
                cropped = rgb.crop((left, top, right, bottom))

            return self._response(cropped, fmt)

        @web.post("/shadow")
        def shadow_endpoint(
            file: UploadFile = File(...),
            bg_color: str = Form("#FFFFFF"),
            bg_url: Optional[str] = Form(None),
            shadow_color: str = Form("#000000"),
            shadow_offset_x: int = Form(8),
            shadow_offset_y: int = Form(12),
            shadow_blur: int = Form(14),
            shadow_opacity: float = Form(0.45),
            format: str = Form("png"),
            authorization: Optional[str] = Header(default=None),
        ):
            """Compose subject onto new bg with a configurable drop shadow."""
            self._check_auth(authorization)
            fmt = self._check_format(format, allowed=frozenset({"png", "webp", "jpg"}))
            data = file.file.read()
            image_obj = self._open_image(data)
            rgb, mask = self._get_mask(image_obj)

            if bg_url:
                try:
                    r = requests.get(bg_url, timeout=15)
                    r.raise_for_status()
                    bg = self._open_image(r.content).convert("RGB").resize(rgb.size, Image.LANCZOS)
                except requests.RequestException as e:
                    raise HTTPException(400, f"Could not fetch bg_url: {e}")
            else:
                bg = Image.new("RGB", rgb.size, self._parse_color(bg_color))

            clean_rgb = self._clean_foreground(rgb, mask)
            cutout = clean_rgb.convert("RGBA")
            cutout.putalpha(mask)
            composed = self._composite_shadow(
                cutout,
                mask,
                bg,
                offset=(int(shadow_offset_x), int(shadow_offset_y)),
                blur=max(0, int(shadow_blur)),
                opacity=max(0.0, min(1.0, float(shadow_opacity))),
                shadow_color=self._parse_color(shadow_color),
            )
            return self._response(composed, fmt)

        @web.post("/sticker")
        def sticker_endpoint(
            file: UploadFile = File(...),
            stroke_color: str = Form("#FFFFFF"),
            stroke_width: int = Form(20),
            format: str = Form("png"),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Sticker style — subject with a thick outline on a transparent background.
            Perfect for WhatsApp/iMessage/Telegram stickers.
            """
            self._check_auth(authorization)
            fmt = self._check_format(format)
            data = file.file.read()
            image_obj = self._open_image(data)
            rgb, mask = self._get_mask(image_obj)

            width = max(1, min(int(stroke_width), 80))
            dilated = self._dilate_mask(mask, width)

            stroke_rgb = Image.new("RGB", rgb.size, self._parse_color(stroke_color))
            stroke_layer = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
            stroke_layer.paste(stroke_rgb, (0, 0), dilated)

            clean_rgb = self._clean_foreground(rgb, mask)
            subject = clean_rgb.convert("RGBA")
            subject.putalpha(mask)

            out = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
            out.alpha_composite(stroke_layer)
            out.alpha_composite(subject)
            return self._response(out, fmt)

        @web.post("/outline")
        def outline_endpoint(
            file: UploadFile = File(...),
            outline_color: str = Form("#000000"),
            outline_width: int = Form(4),
            format: str = Form("png"),
            authorization: Optional[str] = Header(default=None),
        ):
            """Subject on transparent bg with a thin configurable outline."""
            self._check_auth(authorization)
            fmt = self._check_format(format)
            data = file.file.read()
            image_obj = self._open_image(data)
            rgb, mask = self._get_mask(image_obj)

            width = max(1, min(int(outline_width), 60))
            dilated = self._dilate_mask(mask, width)
            outline_only = Image.new("L", rgb.size, 0)
            # Outline = dilated mask minus original mask
            dilated_arr = np.asarray(dilated.convert("L"), dtype=np.int16)
            mask_arr = np.asarray(mask.convert("L"), dtype=np.int16)
            ring_arr = np.clip(dilated_arr - mask_arr, 0, 255).astype(np.uint8)
            outline_only = Image.fromarray(ring_arr, mode="L")

            ring_rgb = Image.new("RGB", rgb.size, self._parse_color(outline_color))
            ring_layer = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
            ring_layer.paste(ring_rgb, (0, 0), outline_only)

            clean_rgb = self._clean_foreground(rgb, mask)
            subject = clean_rgb.convert("RGBA")
            subject.putalpha(mask)

            out = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
            out.alpha_composite(ring_layer)
            out.alpha_composite(subject)
            return self._response(out, fmt)

        @web.post("/studio-shot")
        def studio_shot_endpoint(
            file: UploadFile = File(...),
            bg_color: str = Form("#FFFFFF"),
            aspect: str = Form("1:1"),
            padding: int = Form(48),
            shadow: bool = Form(True),
            format: str = Form("jpg"),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            E-commerce preset — cutout → tight crop → centered on bg with shadow → standard aspect.

            `aspect`: "1:1", "4:5", "16:9", "3:2", or "W:H" (ints). Default 1:1.
            """
            self._check_auth(authorization)
            fmt = self._check_format(format, allowed=frozenset({"png", "webp", "jpg"}))
            data = file.file.read()
            image_obj = self._open_image(data)
            rgb, mask = self._get_mask(image_obj)

            try:
                aw_str, ah_str = aspect.split(":")
                aw, ah = int(aw_str), int(ah_str)
                if aw <= 0 or ah <= 0:
                    raise ValueError()
            except Exception:
                raise HTTPException(400, "aspect must be in 'W:H' format, e.g. '1:1' or '4:5'")

            bbox = self._bounding_box(mask)
            if bbox is None:
                raise HTTPException(400, "No subject detected in image")
            left, top, right, bottom = bbox

            clean_rgb = self._clean_foreground(rgb, mask)
            cutout = clean_rgb.convert("RGBA")
            cutout.putalpha(mask)

            subject_w = right - left
            subject_h = bottom - top
            pad = max(0, int(padding))

            # Target canvas: subject + 2*padding, padded out to aspect ratio
            base_w = subject_w + pad * 2
            base_h = subject_h + pad * 2
            target_w = max(base_w, int(round(base_h * aw / ah)))
            target_h = max(base_h, int(round(target_w * ah / aw)))
            # Re-check W after H adjustment (keeps ratio exact)
            if round(target_w * ah / aw) != target_h:
                target_w = int(round(target_h * aw / ah))

            bg_rgb = Image.new("RGB", (target_w, target_h), self._parse_color(bg_color))

            subject_cut = cutout.crop((left, top, right, bottom))
            subject_mask = mask.crop((left, top, right, bottom))

            paste_x = (target_w - subject_w) // 2
            paste_y = (target_h - subject_h) // 2

            if shadow:
                full_mask_for_shadow = Image.new("L", (target_w, target_h), 0)
                full_mask_for_shadow.paste(subject_mask, (paste_x, paste_y))
                full_cutout = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
                full_cutout.paste(subject_cut, (paste_x, paste_y))
                composed = self._composite_shadow(
                    full_cutout,
                    full_mask_for_shadow,
                    bg_rgb,
                    offset=(8, 12),
                    blur=14,
                    opacity=0.35,
                    shadow_color=(0, 0, 0),
                )
            else:
                composed = bg_rgb.convert("RGBA")
                composed.paste(subject_cut, (paste_x, paste_y), subject_cut)
                composed = composed.convert("RGB")

            return self._response(composed, fmt)

        @web.post("/compare")
        def compare_endpoint(
            file: UploadFile = File(...),
            format: str = Form("png"),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Before/after preview — original on the left, cutout over a checkerboard on the right.
            Perfect for marketing screenshots and social media.
            """
            self._check_auth(authorization)
            fmt = self._check_format(format)
            data = file.file.read()
            image_obj = self._open_image(data)
            rgb, mask = self._get_mask(image_obj)

            clean_rgb = self._clean_foreground(rgb, mask)
            cutout = clean_rgb.convert("RGBA")
            cutout.putalpha(mask)

            w, h = rgb.size
            canvas = Image.new("RGB", (w * 2, h), (255, 255, 255))
            canvas.paste(rgb, (0, 0))
            checker = self._checkerboard((w, h))
            canvas.paste(checker, (w, 0))
            canvas_rgba = canvas.convert("RGBA")
            canvas_rgba.alpha_composite(cutout, dest=(w, 0))
            canvas = canvas_rgba.convert("RGB")

            return self._response(canvas, fmt)

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

        @web.get("/stats")
        def stats_endpoint():
            """
            Public usage counter. Used for landing-page social proof.

            Returns total images processed all-time, today, and a 7-day rolling
            breakdown. Eventually consistent across containers (best-effort).
            """
            from datetime import datetime, timedelta
            try:
                stats = modal.Dict.from_name("knockout-stats", create_if_missing=True)
                today = datetime.utcnow().strftime("%Y-%m-%d")
                last_7 = []
                for i in range(7):
                    d = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
                    last_7.append({"date": d, "count": int(stats.get(f"day:{d}", 0))})
                return {
                    "total_processed": int(stats.get("total", 0)),
                    "today": int(stats.get(f"day:{today}", 0)),
                    "last_7_days": last_7,
                }
            except Exception as e:
                return {
                    "error": "stats unavailable",
                    "detail": str(e),
                    "total_processed": 0,
                    "today": 0,
                    "last_7_days": [],
                }

        @web.post("/upscale")
        def upscale_endpoint(
            file: UploadFile = File(...),
            scale: int = Form(4),
            model: str = Form("swin2sr"),
            face_enhance: bool = Form(False),
            format: str = Form("png"),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Super-resolution. Two backends:

            - `model=swin2sr` (default, v0.6.0+): SwinV2 transformer, sharper detail
              and natural texture on real photos. Successor to SwinIR.
            - `model=realesrgan`: Real-ESRGAN x4plus. Better on anime / illustrations,
              tends to produce a painted look on photos.

            `scale` 2 or 4. `face_enhance=true` routes through GFPGAN (Real-ESRGAN
            backend only — kept for backwards compatibility).
            """
            self._check_auth(authorization)
            fmt = self._check_format(format, allowed=frozenset({"png", "webp", "jpg"}))
            if scale not in (2, 4):
                raise HTTPException(400, "scale must be 2 or 4")

            model_choice = model.strip().lower()
            if model_choice not in {"swin2sr", "realesrgan"}:
                raise HTTPException(400, "model must be 'swin2sr' or 'realesrgan'")

            data = file.file.read()
            image_obj = self._open_image(data)

            try:
                if face_enhance:
                    bgr = self._pil_to_bgr(image_obj)
                    _, _, output_bgr = self.face_restorer.enhance(
                        bgr,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True,
                    )
                    if output_bgr is None:
                        raise HTTPException(500, "upscale produced no output")
                    output_pil = self._bgr_to_pil(output_bgr)
                elif model_choice == "swin2sr":
                    output_pil = self._swin2sr_upscale(image_obj, scale)
                else:
                    bgr = self._pil_to_bgr(image_obj)
                    output_bgr, _ = self.upscaler.enhance(bgr, outscale=scale)
                    if output_bgr is None:
                        raise HTTPException(500, "upscale produced no output")
                    output_pil = self._bgr_to_pil(output_bgr)
            except HTTPException:
                raise
            except RuntimeError as e:
                raise HTTPException(500, f"upscale failed: {e}")

            self._bump_counter()
            return self._response(output_pil, fmt)

        @web.post("/face-restore")
        def face_restore_endpoint(
            file: UploadFile = File(...),
            only_center_face: bool = Form(False),
            bg_enhance: bool = Form(False),
            format: str = Form("png"),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            GFPGAN v1.4 portrait restoration. Fixes blurry / damaged / low-res faces.

            By default the background is preserved as-is — avoids skin-tone bleed
            into bg around face edges (common on warm-toned bgs).

            Set `bg_enhance=true` to also upscale the background 2x via Real-ESRGAN
            (recommended only when bg has cool/neutral tones).

            Set `only_center_face=true` to restore only the most prominent face (faster).
            """
            self._check_auth(authorization)
            fmt = self._check_format(format, allowed=frozenset({"png", "webp", "jpg"}))

            data = file.file.read()
            image_obj = self._open_image(data)
            bgr = self._pil_to_bgr(image_obj)

            restorer = self.face_restorer_full if bg_enhance else self.face_restorer
            try:
                _, _, output_bgr = restorer.enhance(
                    bgr,
                    has_aligned=False,
                    only_center_face=only_center_face,
                    paste_back=True,
                )
            except RuntimeError as e:
                raise HTTPException(500, f"face-restore failed: {e}")

            if output_bgr is None:
                raise HTTPException(500, "face-restore produced no output")

            self._bump_counter()
            return self._response(self._bgr_to_pil(output_bgr), fmt)

        @web.post("/preview")
        def preview_endpoint(
            file: UploadFile = File(...),
            max_dim: int = Form(512),
            format: str = Form("png"),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Fast low-res preview cutout for UX progress indicators.

            Downscales the input to `max_dim` (64-1024) on the long edge and
            skips the pymatting refinement pass. Returns a transparent PNG/WebP.
            ~80ms warm vs ~200ms for /remove.
            """
            self._check_auth(authorization)
            fmt = self._check_format(format)
            data = file.file.read()
            image_obj = self._open_image(data)

            rgb_full = image_obj.convert("RGB")
            w, h = rgb_full.size
            md = max(64, min(int(max_dim), 1024))
            scale = min(1.0, md / max(w, h))
            if scale < 1.0:
                rgb_small = rgb_full.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            else:
                rgb_small = rgb_full

            tensor = self.transform(rgb_small).unsqueeze(0).to("cuda").half()
            with torch.no_grad():
                preds = self.model(tensor)[-1].sigmoid().cpu()
            pred = preds[0].squeeze().float()
            mask = self.to_pil(pred).resize(rgb_small.size)
            self._bump_counter()

            result = rgb_small.convert("RGBA")
            result.putalpha(mask)
            return self._response(result, fmt)

        @web.post("/estimate")
        def estimate_endpoint(body: EstimateBody):
            """
            Predict latency + cost for a given endpoint and image size.

            No GPU work — pure lookup against measured baselines. Intended
            for client-side progress UI and pre-flight billing checks.
            """
            LATENCY_MS_BASE = {
                "remove": 200, "remove-url": 250, "replace-bg": 220,
                "mask": 150, "smart-crop": 180, "shadow": 230,
                "sticker": 220, "outline": 220, "studio-shot": 280,
                "compare": 240, "preview": 80, "headshot": 280,
                "remove-batch": 200, "remove-batch-url": 250,
                "upscale": 1500, "face-restore": 2200,
            }
            ep = body.endpoint.strip().lstrip("/")
            if ep not in LATENCY_MS_BASE:
                raise HTTPException(400, f"unknown endpoint: {ep!r}")

            w = max(1, int(body.width))
            h = max(1, int(body.height))
            px = w * h
            base_ms = LATENCY_MS_BASE[ep]
            # +50% per million pixels above 1MP, capped at +400%
            extra_factor = min(4.0, max(0.0, (px - 1_000_000) / 1_000_000) * 0.5)
            est_ms = int(round(base_ms * (1.0 + extra_factor)))

            return {
                "endpoint": ep,
                "image_pixels": px,
                "est_latency_ms_warm": est_ms,
                "est_latency_ms_cold": est_ms + 8000,  # ~8s cold start on L4
                "est_cost_usd": 0.005,
                "free_during_beta": True,
                "note": "warm = container already running; cold = first request after scaledown",
            }

        @web.post("/headshot")
        def headshot_endpoint(
            file: UploadFile = File(...),
            bg_color: str = Form("#FFFFFF"),
            bg_blur: bool = Form(False),
            blur_radius: int = Form(20),
            aspect: str = Form("4:5"),
            padding: int = Form(64),
            head_top_ratio: float = Form(0.18),
            format: str = Form("jpg"),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            LinkedIn-ready headshot preset.

            Removes background, crops to subject + padding, centers on a portrait
            canvas (default 4:5), and either fills with a solid color or a blurred
            copy of the original (set `bg_blur=true`). `head_top_ratio` controls
            how much empty space sits above the subject (default 18% of canvas).
            """
            self._check_auth(authorization)
            fmt = self._check_format(format, allowed=frozenset({"png", "webp", "jpg"}))
            data = file.file.read()
            image_obj = self._open_image(data)
            rgb, mask = self._get_mask(image_obj)

            try:
                aw_str, ah_str = aspect.split(":")
                aw, ah = int(aw_str), int(ah_str)
                if aw <= 0 or ah <= 0:
                    raise ValueError()
            except Exception:
                raise HTTPException(400, "aspect must be in 'W:H' format, e.g. '4:5'")

            bbox = self._bounding_box(mask)
            if bbox is None:
                raise HTTPException(400, "No subject detected in image")
            left, top, right, bottom = bbox

            clean_rgb = self._clean_foreground(rgb, mask)
            cutout = clean_rgb.convert("RGBA")
            cutout.putalpha(mask)

            subject_w = right - left
            subject_h = bottom - top
            pad = max(0, int(padding))

            base_w = subject_w + pad * 2
            base_h = subject_h + pad * 2
            target_w = max(base_w, int(round(base_h * aw / ah)))
            target_h = max(base_h, int(round(target_w * ah / aw)))
            if round(target_w * ah / aw) != target_h:
                target_w = int(round(target_h * aw / ah))

            if bg_blur:
                blur_r = max(1, min(int(blur_radius), 80))
                bg_full = rgb.copy().filter(ImageFilter.GaussianBlur(radius=blur_r))
                bg_canvas = bg_full.resize((target_w, target_h), Image.LANCZOS)
            else:
                bg_canvas = Image.new("RGB", (target_w, target_h), self._parse_color(bg_color))

            subject_cut = cutout.crop((left, top, right, bottom))
            subject_mask = mask.crop((left, top, right, bottom))

            paste_x = (target_w - subject_w) // 2
            top_ratio = max(0.0, min(0.5, float(head_top_ratio)))
            paste_y = int(round(target_h * top_ratio))
            # clamp so subject fits
            paste_y = min(paste_y, target_h - subject_h - pad)
            paste_y = max(pad, paste_y)

            full_mask = Image.new("L", (target_w, target_h), 0)
            full_mask.paste(subject_mask, (paste_x, paste_y))
            full_cutout = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
            full_cutout.paste(subject_cut, (paste_x, paste_y))

            composed = self._composite_shadow(
                full_cutout,
                full_mask,
                bg_canvas,
                offset=(6, 10),
                blur=18,
                opacity=0.30,
                shadow_color=(0, 0, 0),
            )
            return self._response(composed, fmt)

        return web
