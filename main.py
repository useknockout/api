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
import hashlib
import io
import json
import math
import os
import secrets
import shutil
import subprocess
import tempfile
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import modal


def _now_iso() -> str:
    """ISO-8601 UTC timestamp for Postgres timestamptz columns."""
    return datetime.now(timezone.utc).isoformat()

APP_NAME = "api"
MODEL_REPO = "ZhengPeng7/BiRefNet"
MODEL_INPUT_SIZE = (1024, 1024)
MAX_IMAGE_BYTES = 25 * 1024 * 1024  # 25 MB

# --- Tier-based endpoint gating -------------------------------------------
# Billable endpoints a signed-up FREE-tier user may call. Anything billable
# NOT in this set requires a paid tier (payg/volume/enterprise); free callers
# get a 402 upsell. /estimate is ungated (no auth) so it is intentionally
# absent. Paid-only set = edits (replace-bg, smart-crop, outline, sticker) +
# AI enhancement (upscale, face-restore, colorize, inpaint) + e-commerce
# presets (studio-shot, headshot) + creative (shadow, silhouette) + batch
# (remove-batch, remove-batch-url).
FREE_TIER_ENDPOINTS = frozenset({
    "/remove",
    "/remove-url",
    "/mask",
    "/compare",
    "/preview",
})

# The anonymous shared demo key is a throttled taste of the product: it may
# only hit these endpoints, output is downscaled, and there is a global daily
# cap. Everything else 402s with a signup nudge.
DEMO_ENDPOINTS = frozenset({
    "/remove",
    "/replace-bg",
    "/mask",
    "/sticker",
    "/compare",
})
# Demo output cap. Was 512 after a token-abuse incident, but the public
# playground runs on the demo key — so every visitor's first impression was a
# 512px thumbnail upscaled in the browser, next to competitors showing full-res.
# 1536 still bounds abuse (and the global daily cap is the real guard) while
# looking like the product actually is. Override with DEMO_MAX_DIM env.
DEMO_MAX_DIM = int(os.environ.get("DEMO_MAX_DIM", "") or 1536)

# Signed-up free tier: images/month, no card. Raised 10 -> 30 on 2026-08-13 to
# match withoutbg's "50 free" signup grant — except theirs is a ONE-TIME grant
# that expires in 30 days and ours recurs every month and never expires, so at
# 50 we are strictly more generous than the competitor we were being compared
# against (their 50 is one-time and expires in 30 days; ours recurs).
FREE_MONTHLY_QUOTA = int(os.environ.get("FREE_MONTHLY_QUOTA", "") or 30)
DEMO_DAILY_CAP_DEFAULT = 500       # global anonymous calls/day; DEMO_DAILY_CAP overrides
DEMO_IP_DAILY_CAP_DEFAULT = 10     # per-IP anonymous calls/day; DEMO_IP_DAILY_CAP overrides
DEMO_IP_SALT = os.environ.get("DEMO_IP_SALT", "knockout-demo-ip-v1")  # raw IPs never stored

# CascadePSP's fast=False path refines in ~900px tiles at NATIVE resolution and
# fuses them. A tile landing entirely inside a large flat region (the inside of
# an open box, a plain backdrop panel) carries no boundary evidence, so some
# tiles flip to background and the fused alpha comes back with grid-shaped
# holes. Kravento hit this on a 3072px box on 2026-08-25: 8 interior holes,
# 13.6% of the subject destroyed. The SAME image at 1280px is spotless — the
# bug is purely a function of pixel dimensions. Cap what the refiner sees, then
# upscale the alpha back. 1600 still localises edges far better than the 1024
# default-engine mask, which is where the halo win came from in the first place.
# 900 == the refiner's own tile size. Anything larger tiles and can seam; 1600
# was tried first and still left 7.9% of the box interior semi-transparent.
PRODUCT_REFINE_MAX_DIM = int(os.environ.get("PRODUCT_REFINE_MAX_DIM", "") or 900)
# Refinement may legitimately shrink a boundary; it must never punch a hole
# through the middle of the subject. Measured away from the edge, so ordinary
# boundary tightening cannot trip it.
PRODUCT_REFINE_MAX_INTERIOR_LOSS = 0.01
# engine=auto only: product-v1 must not be WORSE than the default mask it is
# replacing. Auto already computes the default first, so it can compare and
# keep the default when escalation backfires. Shape routing cannot see damage;
# this can. 1% of the subject interior going transparent is the limit.
AUTO_MAX_PRODUCT_REGRESSION = 0.01

# ---- /replace-bg-ai (AI-generated backgrounds) ----------------------------
# EXPERIMENT, allowlist-gated. The generative model NEVER sees the product: we
# cut the subject with BiRefNet, generate only the backdrop from a text prompt,
# and composite. So product pixels are provably untouched and output stays at
# source resolution — neither is true of generative image EDITING.
#
# Public model name -> Azure deployment name. Azure deployment names are chosen
# at deploy time and need not match the catalog id, so every entry is
# overridable by env (AI_BG_DEPLOYMENT_<UPPER_SNAKE>) — a wrong guess is a
# secret edit, not a redeploy.
# (deployment_name, route). Azure serves three different image APIs and picking
# the wrong one 404s:
#   aoai    - Azure OpenAI native models. Deployment in the PATH.
#             {res}.openai.azure.com/openai/deployments/{dep}/images/generations
#   foundry - "sold directly by Azure" partner models (BFL FLUX). Model in BODY.
#             {res}.services.ai.azure.com/openai/v1/images/generations
#   mai     - Microsoft's own MAI image family, its own namespace. Model in BODY.
#             {res}.services.ai.azure.com/mai/v1/images/generations
AI_BG_AZURE_MODELS = {
    "flux2-pro":           ("FLUX.2-pro", "bfl"),
    "flux2-flex":          ("FLUX.2-flex", "bfl"),
    "flux1-kontext-pro":   ("FLUX.1-Kontext-pro", "foundry"),
    "gpt-image-2":         ("gpt-image-2", "aoai"),
    "mai-image-2e":        ("MAI-Image-2e", "mai"),
    "mai-image-2.5":       ("MAI-Image-2.5", "mai"),
    "mai-image-2.5-pro":   ("MAI-Image-2.5-Pro", "mai"),
    "mai-image-2.5-flash": ("MAI-Image-2.5-Flash", "mai"),
}
AI_BG_GOOGLE_MODELS = {
    "nano-banana": "gemini-2.5-flash-image",
}
# ASU AIML gateway — OWNER-ONLY, for free model evaluation. The token is Troy's
# university work credential, so customer traffic must never touch it: these
# models 403 for everyone except the internal is_legacy token, allowlist or not.
# (The gateway strips image INPUTS, which is why this feature generates only the
# backdrop from text — image-to-image evaluation there is impossible.)
AI_BG_ASU_MODELS = {
    "asu-gpt-image-2":  ("openai", "gpt_image2"),
    "asu-nano-banana":  ("gcp-deepmind", "nano_banana_pro"),
    "asu-gemini-flash": ("gcp-deepmind", "geminiflash2_5_image"),
}
AI_BG_ASU_URL = "https://api-main.aiml.asu.edu/query"
AI_BG_DEFAULT_MODEL = os.environ.get("AI_BG_DEFAULT_MODEL", "flux2-pro").strip() or "flux2-pro"
AI_BG_API_VERSION = os.environ.get("AI_BG_API_VERSION", "2025-04-01-preview")
# Empty allowlist = feature OFF for everyone (the default). Comma-separated
# Supabase user_ids enable it. The owner's internal token always passes.
# Read at REQUEST time, not import time: Modal injects secret env vars into the
# container, and module-level globals can evaluate before that lands. Reading
# lazily also means adding a user is a secret edit with no redeploy.
def _ai_bg_allowlist() -> frozenset:
    return frozenset(
        u.strip() for u in os.environ.get("AI_BG_ALLOWLIST", "").split(",") if u.strip()
    )
AI_BG_DAILY_CAP_DEFAULT = 50       # global calls/day; each one spends real provider money
AI_BG_PROMPT_MAX = 500

# ---- /video/remove (async jobs) ----
VIDEO_MAX_SECONDS = 15             # hard cap per clip: 15s ProRes ~305MB stays under the 500MB storage limit; 30s would exceed it
VIDEO_FPS_CAP = 30                 # frames processed per second of video, max
VIDEO_MAX_BYTES = 200 * 1024 * 1024
VIDEO_MAX_DIM = 1920               # frames downscaled to this longest side before inference
VIDEO_METER_EVENT = os.environ.get("STRIPE_VIDEO_METER_EVENT", "video.seconds").strip() or "video.seconds"
# Video bills on its OWN Stripe meter (not the image meter): 1 unit = 1 output
# second, priced at $0.10/second. 15s clip = 15 units = $1.50. Set the price to
# $0.10 ($10/unit? no — unit_amount 10 cents) on the video.seconds meter.
VIDEO_BUCKET = "video-jobs"
VIDEO_FORMATS = frozenset({"prores4444", "webm", "mp4"})
VIDEO_INPUT_EXTS = frozenset({"mp4", "mov", "avi", "webm", "mkv"})

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

    # DDColor (Apache-2.0) — colorization. Pre-fetch the modelscope snapshot
    # so cold starts skip the ~870 MB download. Network errors here are
    # non-fatal: pipeline() will lazy-fetch at request time as a fallback.
    try:
        from modelscope import snapshot_download

        print("Pre-fetching DDColor weights (~870 MB)...")
        snapshot_download("damo/cv_ddcolor_image-colorization")
    except Exception as e:
        print(f"DDColor pre-fetch skipped: {e!r}")

    # LaMa weights (~200 MB) — instantiating SimpleLama triggers the one-time
    # weight download into the user cache dir. Cold starts then skip the fetch.
    try:
        from simple_lama_inpainting import SimpleLama

        print("Pre-fetching LaMa weights (~200 MB)...")
        SimpleLama()
    except Exception as e:
        print(f"LaMa pre-fetch skipped: {e!r}")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")  # ffmpeg: /video/remove demux/remux (ProRes 4444, VP9 alpha)
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
    # DDColor (Apache-2.0) for /colorize via ModelScope. ModelScope brings
    # its own pipeline registry — keeps DDColor's basicsr fork isolated from
    # the basicsr we already use for Real-ESRGAN/GFPGAN.
    # ModelScope's pipelines.base unconditionally imports a dependency chain
    # that requires datasets, oss2, addict, simplejson, sortedcontainers —
    # none of which are auto-installed by `pip install modelscope`. Front-load
    # all of them to avoid iterative rebuilds chasing missing modules.
    .pip_install(
        "modelscope==1.18.1",
        "datasets==2.21.0",
        "oss2==2.18.5",
        "addict==2.4.0",
        "simplejson==3.19.2",
        "sortedcontainers==2.4.0",
    )
    # LaMa (Apache-2.0) for /inpaint via simple-lama-inpainting wrapper.
    # Resolution-robust Large Mask Inpainting — deterministic, no prompts.
    # Weight download (~200 MB) baked into the image below via SimpleLama() warmup.
    .pip_install("simple-lama-inpainting==0.1.2")
    # psd-tools (MIT) for layered PSD export (format=psd). Needs >=1.11 for
    # create_pixel_layer (real transparent layers; frompil flattens to an
    # opaque Background). Re-pin numpy AND pillow in the same layer so psd-tools
    # can't silently bump them (it pulls pillow 12.x + numpy 2.x otherwise) and
    # break pymatting/PIL.
    .pip_install("psd-tools>=1.11,<2", "numpy==1.26.4", "pillow==10.4.0")
    # PyJWT + cryptography for the web-app portal credential (Path 1.5 in
    # _check_auth): verifies Supabase session JWTs (ES256) against the
    # project's JWKS endpoint. No shared JWT secret is stored anywhere.
    .pip_install("pyjwt==2.9.0", "cryptography==43.0.1")
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
    from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
    from gfpgan import GFPGANer
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, UnidentifiedImageError
    from pydantic import BaseModel, HttpUrl
    from pymatting import estimate_foreground_cf, estimate_foreground_ml
    from realesrgan import RealESRGANer
    from torchvision import transforms
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageSegmentation,
        Swin2SRForImageSuperResolution,
    )

    # ModelScope pipeline for DDColor (/colorize endpoint). Imported lazily
    # at container-init time — heavy import, so don't pull at module scope.
    from modelscope.outputs import OutputKeys
    from modelscope.pipelines import pipeline as ms_pipeline

    # LaMa (Apache-2.0) — large-mask inpainting for /inpaint.
    from simple_lama_inpainting import SimpleLama

app = modal.App(APP_NAME, image=image)

# ---- product-v1 engine (S3OD + CascadePSP), isolated container ------------
#
# Alternate cutout engine for flat product photography (lightbox flat-lays,
# e-commerce sheets). S3OD (okupyn/s3od, MIT) produces the coarse mask;
# CascadePSP (segmentation-refinement, MIT) refines it at native resolution.
# Chosen over BiRefNet for this domain in the 2026-08-17 bakeoff — see
# docs/superpowers/specs/2026-08-17-white-halo-problem.md and
# eval/cases/kravento-film/.
#
# Lives in its OWN image + container: its torch 2.6 stack must not touch the
# main image's torch 2.4 / numpy 1.26 pins (basicsr/pymatting/modelscope all
# depend on those), and isolation keeps cold starts, VRAM and rollbacks of the
# experiment separate from every existing endpoint.
product_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers>=4.48",
        "timm",
        "einops",
        "safetensors",
        "huggingface_hub",
        "pillow",
        "numpy",
        "opencv-python-headless",
    )
    .pip_install("segmentation-refinement")
    .pip_install("git+https://github.com/KupynOrest/s3od.git")
)


@app.cls(gpu="L4", scaledown_window=300, timeout=600, image=product_image)
class ProductEngine:
    @modal.enter()
    def load(self):
        from s3od import BackgroundRemoval
        import segmentation_refinement as refine

        self.det = BackgroundRemoval(model_id="okupyn/s3od")
        self.refiner = refine.Refiner(device="cuda")

    @modal.method()
    def cutout(self, image_png: bytes) -> bytes:
        """RGB image bytes in, native-resolution grayscale alpha PNG out."""
        import io
        import cv2
        import numpy as np
        from PIL import Image as PILImage

        im = PILImage.open(io.BytesIO(image_png)).convert("RGB")
        res = self.det.remove_background(im)
        arr = np.squeeze(np.asarray(res.predicted_mask))
        if arr.max() <= 1.5:
            arr = (arr * 255).clip(0, 255).astype("uint8")
        else:
            arr = arr.astype("uint8")
        alpha = PILImage.fromarray(arr)
        if alpha.size != im.size:
            alpha = alpha.resize(im.size, PILImage.LANCZOS)

        # ---- guard 1: never let the refiner tile ------------------------
        # See PRODUCT_REFINE_MAX_DIM. Refine small, then scale the alpha back.
        longest = max(im.size)
        if longest > PRODUCT_REFINE_MAX_DIM:
            s = PRODUCT_REFINE_MAX_DIM / float(longest)
            small = (max(1, int(round(im.width * s))), max(1, int(round(im.height * s))))
            im_r = im.resize(small, PILImage.LANCZOS)
            alpha_r = alpha.resize(small, PILImage.LANCZOS)
            print(f"product engine: refining at {small} (native {im.size})")
        else:
            im_r, alpha_r = im, alpha

        bgr = cv2.cvtColor(np.asarray(im_r), cv2.COLOR_RGB2BGR)
        refined = self.refiner.refine(bgr, np.asarray(alpha_r), fast=False, L=900)
        refined_img = PILImage.fromarray(refined)
        if refined_img.size != im.size:
            refined_img = refined_img.resize(im.size, PILImage.LANCZOS)

        # ---- guard 2: refinement must not eat the subject ---------------
        # Independent of guard 1 on purpose: if some future image tiles badly
        # anyway, we ship the unrefined S3OD mask rather than a holed cutout.
        try:
            from scipy import ndimage

            # Compare CONTINUOUS alpha, not a >128 threshold. The tile seams
            # come back at ~50% alpha, which a binary test scores as "kept"
            # while the pixel is visibly half gone. That mistake is why a 7.9%
            # semi-transparent wash was first reported as fixed.
            base = np.asarray(alpha).astype(np.float32) / 255.0
            ref = np.asarray(refined_img).astype(np.float32) / 255.0
            solid = ndimage.binary_fill_holes(base > 0.5)
            # ignore a boundary band; only interior losses count
            band = max(4, int(0.004 * max(im.size)))
            interior = ndimage.binary_erosion(solid, iterations=band)
            if interior.any():
                drop = (base - ref)[interior]
                lost = float((drop > 0.2).sum()) / float(interior.sum())
                if lost > PRODUCT_REFINE_MAX_INTERIOR_LOSS:
                    print(f"product engine: refinement made {lost:.1%} of the "
                          f"subject interior transparent - DISCARDED, unrefined mask")
                    refined_img = alpha
        except Exception as e:  # a broken guard must never fail the request
            print(f"product engine: interior guard skipped ({e!r})")

        buf = io.BytesIO()
        refined_img.save(buf, format="PNG")
        return buf.getvalue()


@app.cls(
    gpu="L4",
    scaledown_window=300,  # keep warm 5 min between requests
    timeout=1800,  # video jobs run inside the class (30s @ 30fps = 900 frames)
    max_containers=10,
    secrets=[
        modal.Secret.from_name("knockout-secrets"),
        # AI-background provider creds. Separate secret so the experiment can be
        # deleted in one command without touching production credentials.
        modal.Secret.from_name("knockout-ai-bg", required_keys=[]),
    ],
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

        # DDColor — diffusion-free colorization (Apache-2.0). ConvNeXt-Large
        # backbone predicts ab channels in LAB color space. Single feed-forward
        # (no diffusion sampling), ~500ms warm on L4. Inputs can be color or
        # B&W; the model treats input as grayscale internally.
        self.colorizer = ms_pipeline(
            "image-colorization",
            model="damo/cv_ddcolor_image-colorization",
        )

        # LaMa — large-mask inpainting (Apache-2.0). Resolution-robust, deterministic,
        # no prompts. Used by /inpaint. Loads cached weights downloaded at build time.
        self.inpainter = SimpleLama()

    # =========================================================================
    # Auth + usage logging
    # =========================================================================
    # Two paths:
    #   1. Legacy / public-beta — token in API_TOKEN env (comma-separated).
    #      Returns context with user_id=None, tier="free". No DB lookup.
    #   2. Per-user kno_live_<32> / kno_test_<32> — SHA-256 hashed and looked
    #      up in Supabase tokens table. Returns full context (user_id, token_id,
    #      tier) used by usage logging + meter reporting.
    #
    # _check_auth now returns the context dict so endpoints can call _log_usage
    # afterwards. On failure it raises HTTPException as before.

    def _supabase_request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        prefer: Optional[str] = None,
    ) -> Tuple[int, bytes]:
        """Talk to Supabase REST. Service role bypasses RLS — only run server-side."""
        url = os.environ["SUPABASE_URL"].rstrip("/") + path
        if params:
            url += "?" + urllib.parse.urlencode(params, safe=",.()*=:")
        headers = {
            "apikey": os.environ["SUPABASE_SERVICE_ROLE_KEY"],
            "Authorization": f"Bearer {os.environ['SUPABASE_SERVICE_ROLE_KEY']}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, method=method, headers=headers, data=data)
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception:
            return 0, b""

    # ---- Supabase Storage (video jobs) --------------------------------------

    def _storage_request(self, method: str, path: str, data: Optional[bytes] = None,
                         content_type: str = "application/octet-stream",
                         timeout: int = 120) -> Tuple[int, bytes]:
        """Raw call against Supabase Storage. Service role only, server-side."""
        url = os.environ["SUPABASE_URL"].rstrip("/") + "/storage/v1" + path
        headers = {
            "apikey": os.environ["SUPABASE_SERVICE_ROLE_KEY"],
            "Authorization": f"Bearer {os.environ['SUPABASE_SERVICE_ROLE_KEY']}",
            "Content-Type": content_type,
        }
        req = urllib.request.Request(url, method=method, headers=headers, data=data)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def _storage_upload(self, path: str, data: bytes, content_type: str) -> None:
        status, body = self._storage_request(
            "POST", f"/object/{VIDEO_BUCKET}/{path}", data=data, content_type=content_type)
        if status not in (200, 201):
            raise RuntimeError(f"storage upload failed ({status}): {body[:200]!r}")

    def _storage_download(self, path: str) -> bytes:
        status, body = self._storage_request("GET", f"/object/{VIDEO_BUCKET}/{path}")
        if status != 200:
            raise RuntimeError(f"storage download failed ({status}): {body[:200]!r}")
        return body

    def _storage_signed_url(self, path: str, expires_s: int = 3600) -> str:
        status, body = self._storage_request(
            "POST", f"/object/sign/{VIDEO_BUCKET}/{path}",
            data=json.dumps({"expiresIn": expires_s}).encode(),
            content_type="application/json")
        if status != 200:
            raise RuntimeError(f"sign failed ({status}): {body[:200]!r}")
        signed = json.loads(body).get("signedURL", "")
        return os.environ["SUPABASE_URL"].rstrip("/") + "/storage/v1" + signed

    def _job_update(self, job_id: str, **fields) -> None:
        """Patch a video_jobs row. Best-effort — worker keeps going on failure."""
        fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            self._supabase_request(
                "PATCH", "/rest/v1/video_jobs",
                params={"id": f"eq.{job_id}"}, body=fields, prefer="return=minimal")
        except Exception:
            pass

    def _job_get(self, job_id: str) -> Optional[dict]:
        status, body = self._supabase_request(
            "GET", "/rest/v1/video_jobs", params={"id": f"eq.{job_id}", "select": "*"})
        if status != 200:
            return None
        rows = json.loads(body)
        return rows[0] if rows else None

    def _check_auth(self, authorization: Optional[str]) -> dict:
        """Returns a TokenContext dict. Raises HTTPException on auth failure."""
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        presented = authorization.split(" ", 1)[1].strip()
        if not presented:
            raise HTTPException(status_code=401, detail="Empty bearer token")

        # Hard-retired tokens — recognized only so we can return a helpful 402
        # upsell instead of a generic auth error. No access. Env-driven so a
        # leaked key can be killed without a redeploy.
        retired = set()
        retired_env = os.environ.get("API_TOKEN_RETIRED", "").strip()
        if retired_env:
            retired |= {t.strip() for t in retired_env.split(",") if t.strip()}
        if presented in retired:
            raise HTTPException(
                status_code=402,
                detail=(
                    "This key has been retired. Create a free account at "
                    "useknockout.com/signin — 30 images/month free, no card, "
                    "then pay-as-you-go at $0.05/image (4x cheaper than remove.bg)."
                ),
            )

        # Anonymous shared demo key(s) — a throttled taste of the product. The
        # old public-beta key now lands here instead of being killed: it stays
        # the frictionless "try it in 3 seconds" hook, but downstream gating
        # restricts it to /remove only, downscales output, and enforces a
        # global daily cap (see _check_endpoint_access / _enforce_demo_limit).
        # is_legacy keeps it out of the per-user usage table + monthly quota.
        demo_keys = {"kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f"}
        demo_env = os.environ.get("API_TOKEN_DEMO", "").strip()
        if demo_env:
            demo_keys |= {t.strip() for t in demo_env.split(",") if t.strip()}
        if presented in demo_keys:
            return {
                "user_id": None,
                "token_id": None,
                "tier": "free",
                "is_legacy": True,
                "is_demo": True,
            }

        # Path 1: legacy / public-beta token via API_TOKEN env.
        legacy_raw = os.environ.get("API_TOKEN", "").strip()
        legacy_set = {t.strip() for t in legacy_raw.split(",") if t.strip()}
        if presented in legacy_set:
            return {"user_id": None, "token_id": None, "tier": "free", "is_legacy": True}

        # Path 1.5: portal session credential from the web app.
        # Format: knoportal.<token_row_uuid>.<supabase access JWT (ES256)>
        # The browser never holds a plaintext kno_* key; it proves identity with
        # the user's Supabase session JWT, and the token row id names WHICH of
        # their keys to act as. Ownership is enforced in the lookup query.
        if presented.startswith("knoportal."):
            return self._check_portal_auth(presented[len("knoportal."):])

        # Path 2: per-user kno_* token. SHA-256 hashed lookup.
        if not presented.startswith("kno_"):
            raise HTTPException(status_code=401, detail="Invalid token format")

        hashed = hashlib.sha256(presented.encode("utf-8")).hexdigest()

        status, body = self._supabase_request(
            "GET",
            "/rest/v1/tokens",
            params={
                "select": "id,user_id,scopes,revoked_at",
                "hashed_token": f"eq.{hashed}",
                "revoked_at": "is.null",
                "limit": "1",
            },
        )
        if status != 200:
            raise HTTPException(status_code=503, detail="Auth service unavailable")
        try:
            rows = json.loads(body) if body else []
        except json.JSONDecodeError:
            rows = []
        if not rows:
            raise HTTPException(status_code=401, detail="Invalid or revoked token")

        return self._ctx_from_token_row(rows[0])

    _jwks_client = None  # class-level PyJWKClient cache (fetches Supabase JWKS once per container)

    def _check_portal_auth(self, rest: str) -> dict:
        """Verify a knoportal.<key_id>.<jwt> credential. See Path 1.5 above.

        JWT verification is asymmetric (ES256 via the project's JWKS endpoint,
        confirmed live 2026-08-23) — no shared secret enters this codebase.
        """
        key_id, _, jwt_token = rest.partition(".")
        if not key_id or not jwt_token or "." not in jwt_token:
            raise HTTPException(status_code=401, detail="Invalid portal credential format")
        try:
            import jwt as pyjwt
            from jwt import PyJWKClient
            if Knockout._jwks_client is None:
                jwks_url = os.environ["SUPABASE_URL"].rstrip("/") + "/auth/v1/.well-known/jwks.json"
                Knockout._jwks_client = PyJWKClient(jwks_url, cache_keys=True)
            signing_key = Knockout._jwks_client.get_signing_key_from_jwt(jwt_token)
            payload = pyjwt.decode(
                jwt_token, signing_key.key,
                algorithms=["ES256"], audience="authenticated",
            )
        except HTTPException:
            raise
        except Exception as e:
            print(f"portal auth: JWT verification failed: {type(e).__name__}")
            raise HTTPException(status_code=401, detail="Invalid or expired portal session")
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Invalid portal session")

        # Ownership enforced in the query: the key row must belong to the JWT's user.
        status, body = self._supabase_request(
            "GET",
            "/rest/v1/tokens",
            params={
                "select": "id,user_id,scopes,revoked_at",
                "id": f"eq.{key_id}",
                "user_id": f"eq.{sub}",
                "revoked_at": "is.null",
                "limit": "1",
            },
        )
        if status != 200:
            raise HTTPException(status_code=503, detail="Auth service unavailable")
        try:
            rows = json.loads(body) if body else []
        except json.JSONDecodeError:
            rows = []
        if not rows:
            raise HTTPException(status_code=401, detail="Portal key not found or revoked")
        return self._ctx_from_token_row(rows[0])

    def _ctx_from_token_row(self, row: dict) -> dict:
        """Token row → TokenContext. Shared by the raw-key and portal paths so
        tier resolution can never diverge between them."""
        token_id = row["id"]
        user_id = row["user_id"]
        scopes = row.get("scopes") or []

        # Look up tier on the user.
        ustatus, ubody = self._supabase_request(
            "GET",
            "/rest/v1/users",
            params={
                "select": "tier,stripe_customer_id",
                "id": f"eq.{user_id}",
                "limit": "1",
            },
        )
        tier = "free"
        stripe_customer_id = None
        if ustatus == 200:
            try:
                urows = json.loads(ubody) if ubody else []
                if urows:
                    tier = urows[0].get("tier") or "free"
                    stripe_customer_id = urows[0].get("stripe_customer_id")
            except json.JSONDecodeError:
                pass

        # Bump last_used_at (best-effort, fire-and-forget).
        try:
            self._supabase_request(
                "PATCH",
                "/rest/v1/tokens",
                params={"id": f"eq.{token_id}"},
                body={"last_used_at": _now_iso()},
            )
        except Exception:
            pass

        return {
            "user_id": user_id,
            "token_id": token_id,
            "tier": tier,
            "scopes": scopes,
            "stripe_customer_id": stripe_customer_id,
            "is_legacy": False,
        }

    def _check_endpoint_access(self, ctx: dict, endpoint: str) -> None:
        """Tier-based endpoint gate.

        - Demo key: /remove only (DEMO_ENDPOINTS).
        - Signed-up free tier: blocked from paid endpoints (anything not in
          FREE_TIER_ENDPOINTS).
        - Internal full-access API_TOKEN (is_legacy, not demo): ungated.
        - Paid tiers: ungated.
        """
        if ctx.get("is_demo"):
            if endpoint not in DEMO_ENDPOINTS:
                allowed = ", ".join(sorted(DEMO_ENDPOINTS))
                raise HTTPException(
                    status_code=402,
                    detail=(
                        f"The shared demo key only supports {allowed} (low-res). "
                        "Create a free account at useknockout.com/signin for your "
                        "own key — 30 full-quality images/month free across the "
                        "core endpoints, no card."
                    ),
                )
            return
        # Internal legacy full-access token (API_TOKEN env) — not tier-gated.
        if ctx.get("is_legacy"):
            return
        if ctx.get("tier") == "free" and endpoint not in FREE_TIER_ENDPOINTS:
            raise HTTPException(
                status_code=402,
                detail=(
                    f"{endpoint} is a paid endpoint. Your free tier covers the "
                    f"{len(FREE_TIER_ENDPOINTS)} core endpoints (background "
                    "removal + helpers). Upgrade for edits, AI enhancement, "
                    "e-commerce presets & batch at useknockout.com/pricing — "
                    "pay-as-you-go $0.05/image, no minimum."
                ),
            )

    def _enforce_demo_limit(self, ctx: dict) -> None:
        """Global daily cap on the anonymous shared demo key. No-op otherwise.

        Soft cap: read-modify-write on a date-keyed counter in the existing
        knockout-stats modal.Dict (no DB change). Minor overshoot under
        concurrency is fine — this is a cost guard, not a billing meter.
        """
        if not ctx.get("is_demo"):
            return
        try:
            cap = int(os.environ.get("DEMO_DAILY_CAP", "") or DEMO_DAILY_CAP_DEFAULT)
        except ValueError:
            cap = DEMO_DAILY_CAP_DEFAULT
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"demo-{day}"
        try:
            d = modal.Dict.from_name("knockout-stats", create_if_missing=True)
            used = int(d.get(key, 0))
            if used >= cap:
                raise HTTPException(
                    status_code=402,
                    detail=(
                        "The shared demo key has hit today's global free limit. "
                        "Create a free account at useknockout.com/signin for your "
                        "own key — 30 images/month free, no card, available now."
                    ),
                )
            d[key] = used + 1

            # Per-IP daily cap, on top of the global one. Requested by the
            # web-app session (2026-08-23): the demo key is public and CORS is
            # open, so browser-side throttles are decoration — only this
            # server-side counter is real. IPs are salted-hashed, never stored.
            ip = (ctx.get("client_ip") or "").strip()
            if ip:
                try:
                    ip_cap = int(os.environ.get("DEMO_IP_DAILY_CAP", "")
                                 or DEMO_IP_DAILY_CAP_DEFAULT)
                except ValueError:
                    ip_cap = DEMO_IP_DAILY_CAP_DEFAULT
                iph = hashlib.sha256((DEMO_IP_SALT + ip).encode("utf-8")).hexdigest()[:16]
                ip_key = f"demo-ip:{day}:{iph}"
                ip_used = int(d.get(ip_key, 0))
                if ip_used >= ip_cap:
                    raise HTTPException(
                        status_code=429,
                        detail=(
                            "Daily free limit reached for this connection. "
                            "Sign in at useknockout.com/signin for your own key — "
                            "30 images/month free, no card."
                        ),
                    )
                d[ip_key] = ip_used + 1
        except HTTPException:
            raise
        except Exception:
            # Never fail a real call because the counter store hiccuped.
            pass

    def _check_scope(self, ctx: dict, endpoint: str) -> None:
        """If a token has scopes, deny calls to endpoints not in the list."""
        scopes = ctx.get("scopes") or []
        if not scopes:
            return  # full-access token
        if endpoint not in scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Token not authorized for {endpoint}",
            )

    def _is_pro(self, ctx: dict) -> bool:
        """True for Knockout Plus ('pro') and enterprise ('volume') tiers."""
        return ctx.get("tier") in {"pro", "volume"}

    def _require_pro(self, ctx: dict, feature: str = "This feature") -> None:
        """Gate premium features behind Knockout Plus. 402 for everyone else.

        Internal full-access (is_legacy) bypasses, so the owner's own key still
        works for testing. Premium = despill, watermarks, saved presets.
        """
        if ctx.get("is_legacy") or self._is_pro(ctx):
            return
        raise HTTPException(
            status_code=402,
            detail=(
                f"{feature} requires Knockout Plus. Upgrade at "
                "useknockout.com/pricing to unlock edge despill, saved presets, "
                "custom watermarks, and PSD-included exports for $10/month."
            ),
        )

    def _enforce_quota(self, ctx: dict) -> None:
        """Free tier: FREE_MONTHLY_QUOTA images/month. Paid tiers: no cap."""
        if ctx.get("is_legacy"):
            return
        if ctx.get("tier") != "free":
            return
        user_id = ctx.get("user_id")
        if not user_id:
            return
        # Rough check: count rows in usage_current_month view.
        status, body = self._supabase_request(
            "GET",
            "/rest/v1/usage_current_month",
            params={
                "select": "call_count",
                "user_id": f"eq.{user_id}",
                "limit": "1",
            },
        )
        if status == 200 and body:
            try:
                rows = json.loads(body)
                if rows:
                    used = int(rows[0].get("call_count") or 0)
                    if used >= FREE_MONTHLY_QUOTA:
                        raise HTTPException(
                            status_code=402,
                            detail=(
                                f"Free tier monthly quota ({FREE_MONTHLY_QUOTA}) exhausted. "
                                "It resets next month, or upgrade at useknockout.com/pricing."
                            ),
                        )
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

    def _log_usage(
        self,
        ctx: dict,
        endpoint: str,
        status: int,
        latency_ms: int,
        units: int = 1,
        meter_event: Optional[str] = None,
        skip_meter: bool = False,
    ) -> None:
        """Insert a usage row + fire a Stripe meter event for paid tiers.

        units: billing weight (1 = one base image).
        meter_event: override the meter to fire (e.g. 'psd.exported' for the
            $0.10 PSD add-on, which replaces the base image meter).
        skip_meter: bill nothing for this call (e.g. a Plus subscriber's PSD
            export, which is included in their flat plan).
        """
        if ctx.get("is_legacy"):
            # Public-beta calls — don't pollute the per-user usage table.
            return
        user_id = ctx.get("user_id")
        token_id = ctx.get("token_id")
        if not user_id:
            return

        # 1. Usage row in Supabase
        try:
            self._supabase_request(
                "POST",
                "/rest/v1/usage",
                body={
                    "user_id": user_id,
                    "token_id": token_id,
                    "endpoint": endpoint,
                    "status": status,
                    "latency_ms": latency_ms,
                    # Billing weight: images/psd = 1 per call, collage = N photos,
                    # video = output seconds. Lets the dashboard compute exact spend.
                    "units": max(1, int(units)),
                },
                prefer="return=minimal",
            )
        except Exception:
            pass

        # 2. Stripe meter event for paid tiers (pro pays base per-image too —
        #    the $10/mo unlocks features, image volume is still metered).
        if 200 <= status < 300 and not skip_meter and ctx.get("tier") in {"payg", "volume", "pro"}:
            self._report_meter(ctx, units=units, event_name=meter_event)

    def _begin(self, authorization: Optional[str], endpoint: str,
               client_ip: Optional[str] = None) -> Tuple[dict, float]:
        """One call → auth + scope + quota + start timer. Use at top of each handler.

        client_ip: first-hop X-Forwarded-For, passed only by endpoints that
        enforce the per-IP demo cap (currently /remove). Never stored raw.
        """
        ctx = self._check_auth(authorization)
        if client_ip:
            ctx["client_ip"] = client_ip
        self._check_endpoint_access(ctx, endpoint)
        self._check_scope(ctx, endpoint)
        self._enforce_demo_limit(ctx)
        self._enforce_quota(ctx)
        return ctx, time.perf_counter()

    def _end(self, ctx: dict, endpoint: str, start: float, status: int = 200, units: int = 1,
             meter_event: Optional[str] = None, skip_meter: bool = False) -> None:
        """Use after handler finishes. Records usage row + fires Stripe meter.

        units: billing weight (default 1 base image).
        meter_event / skip_meter: see _log_usage (PSD add-on + Plus exemption).
        """
        latency_ms = int((time.perf_counter() - start) * 1000)
        self._log_usage(ctx, endpoint, status, latency_ms, units=units,
                        meter_event=meter_event, skip_meter=skip_meter)

    def _report_meter(self, ctx: dict, units: int = 1, event_name: Optional[str] = None) -> None:
        """Fire one Stripe meter event per successful call. units = billed weight.

        event_name overrides the default base meter — e.g. PSD export fires its
        own 'psd.exported' meter ($0.10) instead of the base images.processed.
        """
        sk = os.environ.get("STRIPE_SECRET_KEY", "").strip()
        if not event_name:
            event_name = os.environ.get("STRIPE_METER_EVENT_NAME", "images.processed").strip()
        customer = ctx.get("stripe_customer_id")
        if not sk or not customer:
            return
        try:
            data = urllib.parse.urlencode({
                "event_name": event_name,
                "timestamp": int(time.time()),
                "payload[value]": str(max(1, int(units))),
                "payload[stripe_customer_id]": customer,
                # Unique per call. token+millisecond alone collides under
                # concurrent calls (Stripe dedupes identical identifiers →
                # silent under-billing). Random suffix guarantees each billable
                # call is counted exactly once.
                "identifier": f"uk_{ctx.get('token_id') or 'na'}_{int(time.time() * 1000)}_{secrets.token_hex(4)}",
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.stripe.com/v1/billing/meter_events",
                method="POST",
                headers={
                    "Authorization": f"Bearer {sk}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data=data,
            )
            urllib.request.urlopen(req, timeout=3).read()
        except Exception:
            # Meter loss tolerated — don't fail the customer call on Stripe hiccups.
            pass

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
            # Honor EXIF orientation — phone photos store rotation as a tag
            # instead of rotating pixels. Without this, portrait shots come
            # out sideways. exif_transpose bakes the rotation in + strips
            # the tag so all downstream processing sees upright pixels.
            image_obj = ImageOps.exif_transpose(image_obj)
            return image_obj
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "Invalid or unsupported image")

    def _downscale_max(self, image_obj, max_dim: int):
        """Downscale so the longest side is <= max_dim. Used to cap demo output."""
        w, h = image_obj.size
        if max(w, h) <= max_dim:
            return image_obj
        scale = max_dim / float(max(w, h))
        return image_obj.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            Image.LANCZOS,
        )

    def _get_mask(self, image_obj, refine: bool = True, count: bool = True):
        """Run BiRefNet on an RGB image, return (rgb_image, mask_pil).

        Pads the image to a square BEFORE the model's fixed 1024x1024 resize so
        the aspect ratio is preserved. Resizing a non-square image straight to
        1024x1024 squishes it, which wrecks thin, low-contrast boundaries (water
        reflections, fine hair, etc). We pad to a square, infer, then crop the
        mask back to the original frame.

        refine=False skips the guided-filter edge refinement — for callers that
        only need a region (e.g. /inpaint), where the full-res float32 filter
        buffers are pure wasted memory + latency.

        count=False skips the public stats counter — for internal second passes
        (detect=high_recall) that are still one user-facing image.
        """
        rgb = image_obj.convert("RGB")
        w, h = rgb.size
        if w != h:
            side = max(w, h)
            square = Image.new("RGB", (side, side), (0, 0, 0))
            square.paste(rgb, (0, 0))
        else:
            side = w
            square = rgb
        tensor = self.transform(square).unsqueeze(0).to("cuda").half()
        with torch.no_grad():
            preds = self.model(tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze().float()
        mask = self.to_pil(pred).resize((side, side))
        if w != h:
            mask = mask.crop((0, 0, w, h))  # drop the padded region
        if refine:
            mask = self._refine_alpha(rgb, mask)
        if count:
            self._bump_counter()
        return rgb, mask

    def _refine_alpha(self, rgb, mask, radius: int = 8, eps: float = 1e-3):
        """Edge-aware alpha refinement via a guided filter (guide = the image).

        BiRefNet masks are soft and can wander off the true boundary on thin,
        low-contrast edges (water reflections, fine hair). A guided filter snaps
        the alpha to real image edges. Cheap — a handful of box filters. Falls
        back to the input mask on any failure.
        """
        try:
            import cv2
            I = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
            p = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
            if I.shape != p.shape:
                p = np.asarray(mask.convert("L").resize(rgb.size), dtype=np.float32) / 255.0
            r = (radius, radius)
            mean_I = cv2.boxFilter(I, -1, r)
            mean_p = cv2.boxFilter(p, -1, r)
            cov_Ip = cv2.boxFilter(I * p, -1, r) - mean_I * mean_p
            var_I = cv2.boxFilter(I * I, -1, r) - mean_I * mean_I
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            q = cv2.boxFilter(a, -1, r) * I + cv2.boxFilter(b, -1, r)
            q = np.clip(q * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(q, mode="L")
        except Exception:
            return mask

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
        processor = self.swin2sr_proc_x4 if scale == 4 else self.swin2sr_proc_x2

        rgb = image_obj.convert("RGB")
        src = np.asarray(rgb, dtype=np.uint8)  # H, W, 3
        h, w, _ = src.shape

        # Core-crop tiling. Run each tile WITH a halo of surrounding context, but
        # keep only its non-overlapping CORE in the output. Cores tile exactly
        # (no seams) and are never averaged against another tile (no smear). The
        # old approach blended overlapping SR predictions with a window — on
        # small images tiles overlapped almost entirely, so averaging two
        # different model outputs across most of the frame smeared away the
        # high-frequency detail and the result looked like a plain resize.
        core = 192   # non-overlapping region written to the output (input px)
        halo = 32    # context fed to the model around each core, then discarded
        out = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)

        for cy in range(0, h, core):
            for cx in range(0, w, core):
                ch = min(core, h - cy)
                cw = min(core, w - cx)
                # Padded input patch = core + halo, clamped to image bounds.
                py0, py1 = max(0, cy - halo), min(h, cy + ch + halo)
                px0, px1 = max(0, cx - halo), min(w, cx + cw + halo)
                patch = src[py0:py1, px0:px1, :]

                inputs = processor(Image.fromarray(patch), return_tensors="pt")
                pixel_values = inputs["pixel_values"].to("cuda").half()
                with torch.no_grad():
                    rec = model(pixel_values=pixel_values).reconstruction

                arr = rec.squeeze(0).clamp_(0, 1).float().cpu().numpy()
                arr = np.transpose(arr, (1, 2, 0))
                ph, pw = py1 - py0, px1 - px0
                arr = arr[: ph * scale, : pw * scale, :]  # drop the model's window padding

                # Crop this tile's core out of the padded SR patch, place exactly.
                oy_in, ox_in = (cy - py0) * scale, (cx - px0) * scale
                core_sr = arr[oy_in: oy_in + ch * scale, ox_in: ox_in + cw * scale, :]
                oy, ox = cy * scale, cx * scale
                out[oy: oy + ch * scale, ox: ox + cw * scale, :] = \
                    np.clip(core_sr * 255.0, 0, 255).astype(np.uint8)

        return Image.fromarray(out, mode="RGB")

    def _pil_to_bgr(self, image_obj):
        """PIL Image (any mode) → contiguous BGR uint8 ndarray expected by Real-ESRGAN/GFPGAN."""
        rgb = np.array(image_obj.convert("RGB"))
        return np.ascontiguousarray(rgb[:, :, ::-1])

    def _bgr_to_pil(self, bgr_arr):
        """BGR uint8 ndarray → RGB PIL Image."""
        return Image.fromarray(bgr_arr[:, :, ::-1])

    def _clean_foreground(self, rgb: "Image.Image", mask: "Image.Image", *, fast: bool = False,
                          strength: Optional[float] = None) -> "Image.Image":
        """
        Estimate pure foreground RGB at mask edges using closed-form matting.
        Eliminates color spill / halo from the original background (despill).

        strength: 0.0-1.0 blend between raw RGB (0) and fully-despilled (1).
            None = full despill (1.0), the default. Lets callers dial edge
            decontamination down for speed or to preserve original edge color.

        Skipped when FOREGROUND_REFINE=false. Downscaled internally to keep
        compute bounded (closed-form is O(N²) but solver is sparse).
        """
        if os.environ.get("FOREGROUND_REFINE", "true").strip().lower() in {"false", "0", "no", "off"}:
            return rgb

        s = 1.0 if strength is None else max(0.0, min(float(strength), 1.0))
        if s == 0.0:
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
            # Partial despill — blend matted result back toward raw RGB.
            if s < 1.0:
                clean_img = Image.blend(rgb.convert("RGB"), clean_img, s)
            return clean_img
        except Exception:
            # Any pymatting failure → degrade gracefully to raw RGB
            return rgb

    @staticmethod
    def _despill_strength(despill):
        """Normalize a 0-100 despill request to a 0.0-1.0 matting strength.
        None passes through (helper applies its full-despill default)."""
        if despill is None:
            return None
        return max(0.0, min(float(despill) / 100.0, 1.0))

    @staticmethod
    def _harden_alpha(mask, k: float = 4.0, erode_px: int = 1):
        """Commit ambiguous alpha to opaque-or-transparent (product-shot edges).

        BiRefNet + guided-filter refinement leave a soft mid-alpha band around
        subjects (measured ~4-8% of pixels). On hard-surface subjects that
        renders as a grey halo. Steepening the alpha curve around 0.5 removes
        ~86% of the halo while thin SOLID structures (club shafts, jewelry
        chains) survive, because they sit at alpha~1, not mid-alpha
        (eval/cases: chain mass -0.46%, hair is the exception — keep soft for
        portraits). A 1px erosion then trims the residual fringe the way
        remove.bg/Canva do: trade one pixel of true edge for a clean cut.
        """
        from PIL import ImageFilter
        a = np.asarray(mask, dtype=np.float32) / 255.0
        a = np.clip((a - 0.5) * k + 0.5, 0.0, 1.0)
        out = Image.fromarray((a * 255).astype(np.uint8), mode="L")
        if erode_px > 0:
            out = out.filter(ImageFilter.MinFilter(2 * erode_px + 1))
        return out

    def _check_edge(self, edge: str) -> str:
        e = (edge or "soft").strip().lower()
        if e not in ("soft", "hard"):
            raise HTTPException(400, "edge must be 'soft' or 'hard'")
        return e

    def _check_detect(self, detect: str) -> str:
        d = (detect or "standard").strip().lower()
        if d not in ("standard", "high_recall"):
            raise HTTPException(400, "detect must be 'standard' or 'high_recall'")
        return d

    def _check_engine(self, engine: str) -> str:
        e = (engine or "standard").strip().lower()
        if e not in ("standard", "product-v1", "auto"):
            raise HTTPException(400, "engine must be 'standard', 'product-v1', or 'auto'")
        return e

    # engine=auto routing thresholds. Tuned 2026-08-21 on the default-engine
    # masks in eval/cases/ (see the table in the docstring of _auto_should_escalate).
    # Tightened 40 -> 22 on 2026-08-25. Measured on every default-engine mask
    # we hold: products cluster at 15.7-19.3 (Kravento film 15.7, shoe 16.1,
    # flowerbox 17.0-18.3, Kravento box 19.3) and the first case we do NOT want
    # to escalate is a golden retriever at 24.7 — product-v1 keeps its fur but
    # hardens the edge (feathered pixels 16.6k -> 5.0k), which reads as a cutout
    # on an animal. 22 sits in the empty gap between the two clusters. Erring
    # low is the safe direction: a missed escalation is a halo, an unwanted one
    # can damage the subject.
    AUTO_MAX_COMPLEXITY = 22.0   # perimeter^2 / area
    AUTO_MAX_THIN_LOSS = 0.08    # fraction of subject lost to erosion

    def _auto_should_escalate(self, rgb, mask) -> Tuple[bool, dict]:
        """Decide whether the default result warrants a product-v1 rerun.

        Measured on the DEFAULT engine's own mask, so the decision is about the
        result we already have rather than a guess about the photo.

        Two shape signals, both scale-normalised. Values measured on our eval
        set (default-engine masks), escalate-worthy on the left:

            film-white 15.8 / 0.047   flower-pale 110.2 / 0.119
            film-tan   16.0 / 0.047   foliage     114.3 / 0.117
            shoe       16.1 / 0.038   hair        122.4 / 0.091
            dog        24.7 / 0.053   chain       514.0 / 0.231

        Flat opaque products cluster under 25; anything with fur, foliage or
        thin structure sits above 110. Thresholds sit in that 4x gap.

        NOTE: an edge-band "does this look like background" halo score was tried
        here first and REJECTED — it scored flower 0.638 and chain 0.643 against
        film 0.28-0.31, i.e. backwards. Shape separates cleanly; colour does not.

        Bias is deliberate: when in doubt, do not escalate. A missed escalation
        is the behaviour customers already had; a wrong escalation can delete
        pale foliage (verified 2026-08-19, eval/cases/flower-pale-leaves/).
        """
        try:
            from scipy import ndimage
        except ImportError:
            return False, {"reason": "scipy unavailable"}
        m = np.asarray(mask.convert("L")) > 128
        area = int(m.sum())
        if area < 5000:
            return False, {"reason": "subject too small", "area": area}
        perim = int((m & ~ndimage.binary_erosion(m, iterations=1)).sum())
        complexity = (perim ** 2) / area
        r = max(2, int(0.012 * (area ** 0.5)))
        thin_loss = 1.0 - (ndimage.binary_erosion(m, iterations=r).sum() / area)
        esc = complexity < self.AUTO_MAX_COMPLEXITY and thin_loss < self.AUTO_MAX_THIN_LOSS
        return esc, {"complexity": round(float(complexity), 1),
                     "thin_loss": round(float(thin_loss), 3)}

    def _acquire_mask_auto(self, image_obj, detect: str, decontaminate: bool):
        """engine=auto: default mask, escalate to product-v1 only if it fits.

        Returns (rgb, mask, engine_used). Never raises on product-v1 failure —
        falls back to the default result we already computed, so auto cannot be
        worse than standard.
        """
        rgb, mask = self._acquire_mask(image_obj, detect=detect, decontaminate=decontaminate)
        esc, sig = self._auto_should_escalate(rgb, mask)
        print(f"engine=auto: escalate={esc} {sig}")
        if not esc:
            return rgb, mask, "standard"
        try:
            rgb2, mask2 = self._product_mask(image_obj)
        except Exception as e:
            print(f"engine=auto: product-v1 failed, keeping default result: {e!r}")
            return rgb, mask, "standard"

        # Shape said "product", but shape cannot see whether product-v1 damaged
        # the subject. We already hold the default mask, so compare them: any
        # region the default kept solid and product-v1 made transparent is a
        # regression, not a refinement. Kravento's gold box is exactly this
        # case (complexity 19.3 - reads as a product, comes back holed).
        # Continuous alpha, not a >128 test: the damage arrives at ~50%.
        try:
            from scipy import ndimage as _ndi

            a_def = np.asarray(mask.convert("L")).astype(np.float32) / 255.0
            a_prd = np.asarray(mask2.convert("L").resize(mask.size)).astype(np.float32) / 255.0
            solid = _ndi.binary_fill_holes(a_def > 0.5)
            band = max(4, int(0.004 * max(mask.size)))
            interior = _ndi.binary_erosion(solid, iterations=band)
            if interior.any():
                worse = float(((a_def - a_prd)[interior] > 0.2).sum()) / float(interior.sum())
                if worse > AUTO_MAX_PRODUCT_REGRESSION:
                    print(f"engine=auto: product-v1 made {worse:.1%} of the subject "
                          f"transparent vs default - REJECTED, using standard")
                    return rgb, mask, "standard"
        except Exception as e:
            print(f"engine=auto: regression check skipped ({e!r})")

        return rgb2, mask2, "product-v1"

    def _product_mask(self, image_obj):
        """Cutout via the isolated ProductEngine container (S3OD + CascadePSP).

        Cross-container call: adds ~1-2s warm, a cold start on first use. The
        refinement pass alone runs 15-20s on a 12MP image, which is why this
        engine is opt-in and gated to paid tiers rather than a default.
        """
        import io as _io
        rgb = image_obj.convert("RGB")
        buf = _io.BytesIO()
        rgb.save(buf, format="PNG")
        try:
            mask_png = ProductEngine().cutout.remote(buf.getvalue())
        except Exception as e:
            print(f"product engine failed: {e!r}")
            raise HTTPException(502, "product engine unavailable, retry or use engine=standard")
        mask = Image.open(_io.BytesIO(mask_png)).convert("L")
        if mask.size != rgb.size:
            mask = mask.resize(rgb.size, Image.LANCZOS)
        return rgb, mask

    def _require_paid_compute(self, ctx: dict, feature: str = "This mode") -> None:
        """Gate 2x-inference modes to paying tiers. 402 for demo/free.

        detect=high_recall runs two model passes for one billed image — without
        this gate the shared demo key or a free account could buy double GPU
        at single-request cost (the exact amplification the demo cap exists to
        prevent). Internal full-access (is_legacy) bypasses for testing.
        """
        if ctx.get("is_legacy"):
            return
        if not ctx.get("is_demo") and ctx.get("tier") not in (None, "free"):
            return  # any paid tier (payg/pro/volume)
        raise HTTPException(
            status_code=402,
            detail=(
                f"{feature} runs double inference and needs a paid plan. "
                "Pay-as-you-go at useknockout.com/pricing, no minimum."
            ),
        )

    @staticmethod
    def _boost_for_detect(rgb):
        """Chroma/contrast-boosted copy of the image for a second detection pass.

        Low-contrast product shots (e.g. a beige sheet on white paper, measured
        ~4% RGB separation on a real client catalog) sit at the edge of what the
        model can see: the mask bites chunks out of the product exactly where it
        meets a near-identical background. Boosting chroma pushes those near-white
        tones apart so pass 2 recovers what pass 1 missed. The boost is NOT safe
        alone — it recovers one boundary while destroying another — which is why
        it only ever feeds the union in detect=high_recall, never replaces pass 1.
        """
        b = ImageOps.autocontrast(rgb, cutoff=1)
        b = ImageEnhance.Color(b).enhance(2.2)
        return ImageEnhance.Contrast(b).enhance(1.35)

    @staticmethod
    def _srgb_to_lab_weighted(rgb_arr):
        """sRGB (float 0-255, HxWx3) -> Lab with L scaled by 0.4.

        Chroma (a/b) carries the product-vs-background decision; de-emphasizing
        lightness separates e.g. white base paper from a pale pink sheet, which
        differ almost entirely in chroma. Validated: full-weight L left a pale
        sheet's 53k-px white sliver untouched; 0.4 removed all but 7px.
        """
        c = rgb_arr / 255.0
        c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
        M = np.array([[0.4124, 0.3576, 0.1805],
                      [0.2126, 0.7152, 0.0722],
                      [0.0193, 0.1192, 0.9505]], dtype=np.float32)
        xyz = c @ M.T
        xyz /= np.array([0.9505, 1.0, 1.089], dtype=np.float32)
        f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
        L = (116 * f[..., 1] - 16) * 0.4
        a = 500 * (f[..., 0] - f[..., 1])
        b = 200 * (f[..., 1] - f[..., 2])
        return np.stack([L, a, b], axis=-1).astype(np.float32)

    @staticmethod
    def _kmeans_np(pts, k=3, iters=25, seed=0):
        """Tiny numpy k-means (<=20k pts x 3 dims) — no sklearn dependency."""
        rng = np.random.default_rng(seed)
        C = pts[rng.choice(len(pts), size=min(k, len(pts)), replace=False)].copy()
        for _ in range(iters):
            d = ((pts[:, None, :] - C[None]) ** 2).sum(-1)
            lbl = d.argmin(1)
            for j in range(len(C)):
                m = lbl == j
                if m.any():
                    C[j] = pts[m].mean(0)
        return C

    def _decontaminate_mask(self, rgb, mask, band_r=15, bg_margin=1.5, fg_margin=2.0,
                            samples=20000, work=1024):
        """Trimap-band color decontamination (the step generic salience lacks).

        The model finds objects, not colors — so a white strip of base paper
        beside a pink sheet, or a black rig fragment beside a white sheet, gets
        included even though its color screams background. This walks only the
        uncertain band around the mask boundary, builds per-image k-means color
        models of confident-foreground and confident-background (Lab, chroma-
        weighted), and reassigns band pixels ONLY on strong evidence margins.
        Asymmetric: removing needs 1.5x, adding needs 2.0x (adding is riskier),
        and alpha is only ever raised where the model already gave >32 support.
        Black products are safe by construction: a black core makes black a
        FOREGROUND color (validated: synthetic black-on-white, interior 100.0%
        intact). Analysis at 1024 work-scale; band_r=15 there covers ~45px at
        3072 input — wider than the widest measured contamination strip.
        Fails open: any error returns the input mask unchanged.
        """
        try:
            from PIL import ImageFilter
            alpha_u8 = np.asarray(mask.convert("L"))
            H, W = alpha_u8.shape
            s = min(1.0, work / float(max(H, W)))  # never upscale small inputs for analysis
            wh, ww = max(1, int(round(H * s))), max(1, int(round(W * s)))
            rgb_w = np.asarray(rgb.resize((ww, wh), Image.BILINEAR), dtype=np.float32)
            a_w = np.asarray(mask.convert("L").resize((ww, wh), Image.BILINEAR)).astype(np.float32)

            size = 2 * band_r + 1
            op_img = Image.fromarray(((a_w > 128) * 255).astype(np.uint8))
            core_fg = np.asarray(op_img.filter(ImageFilter.MinFilter(size))) > 128
            far_bg = ~(np.asarray(op_img.filter(ImageFilter.MaxFilter(size))) > 128)
            band = ~core_fg & ~far_bg
            if not band.any() or not core_fg.any() or not far_bg.any():
                return mask

            lab = self._srgb_to_lab_weighted(rgb_w)
            rng = np.random.default_rng(0)

            def sample(m):
                ys, xs = np.where(m)
                idx = rng.choice(len(ys), size=min(samples, len(ys)), replace=False)
                return lab[ys[idx], xs[idx]]

            Cf = self._kmeans_np(sample(core_fg))
            Cb = self._kmeans_np(sample(far_bg))

            bys, bxs = np.where(band)
            p = lab[bys, bxs]
            dfg = np.sqrt(((p[:, None, :] - Cf[None]) ** 2).sum(-1)).min(1)
            dbg = np.sqrt(((p[:, None, :] - Cb[None]) ** 2).sum(-1)).min(1)
            to_bg = dbg * bg_margin < dfg
            to_fg = (dfg * fg_margin < dbg) & (a_w[bys, bxs] > 32)

            out_w = a_w.copy()
            out_w[bys[to_bg], bxs[to_bg]] = 0.0
            out_w[bys[to_fg], bxs[to_fg]] = 255.0
            out_img = Image.fromarray(out_w.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))

            # apply changes only inside the (upsampled) band — outside it the
            # original full-resolution alpha passes through untouched
            up_clean = np.asarray(out_img.resize((W, H), Image.BILINEAR), dtype=np.float32)
            band_full = np.asarray(
                Image.fromarray((band * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
            ) > 64
            out = alpha_u8.astype(np.float32).copy()
            out[band_full] = up_clean[band_full]
            return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="L")
        except Exception:
            return mask  # cleanup is best-effort; never break the request

    def _acquire_mask(self, image_obj, detect: str = "standard", decontaminate: bool = False):
        """Shared mask acquisition for /remove, /studio-shot, /smart-crop.

        detect=high_recall: second inference on a chroma-boosted copy,
        per-pixel max of the RAW alphas, then ONE guided-filter refine against
        the original image. The passes fail in DIFFERENT places on low-contrast
        boundaries, so the union keeps whatever either was confident about
        (validated on client flat-lay shots: both eaten corners recovered).
        Fusing raw-then-refining ties the final boundary to the pixels actually
        returned — refining each pass separately would snap pass 2's alpha to
        the boosted copy's shifted edges. Union is inclusion-biased (can only
        add alpha) — hence opt-in and named high_recall, not "better": it
        trades precision for recall by construction.
        """
        if detect == "high_recall":
            from PIL import ImageChops
            rgb, raw1 = self._get_mask(image_obj, refine=False)
            _, raw2 = self._get_mask(self._boost_for_detect(rgb), refine=False, count=False)
            mask = self._refine_alpha(rgb, ImageChops.lighter(raw1, raw2))
        else:
            rgb, mask = self._get_mask(image_obj)
        if decontaminate:
            mask = self._decontaminate_mask(rgb, mask)
        return rgb, mask

    def _remove(self, image_obj, despill=None, edge: str = "soft", detect: str = "standard",
                decontaminate: bool = False, engine: str = "standard", info=None):
        """info: optional dict; receives {"engine": <engine that actually ran>}."""
        if engine == "product-v1":
            # detect/decontaminate are BiRefNet-path knobs; ignored here by design.
            rgb, mask = self._product_mask(image_obj)
            used = "product-v1"
        elif engine == "auto":
            rgb, mask, used = self._acquire_mask_auto(image_obj, detect, decontaminate)
        else:
            rgb, mask = self._acquire_mask(image_obj, detect=detect, decontaminate=decontaminate)
            used = "standard"
        if info is not None:
            info["engine"] = used
        if edge == "hard":
            mask = self._harden_alpha(mask)
        clean_rgb = self._clean_foreground(rgb, mask, strength=self._despill_strength(despill))
        result = clean_rgb.convert("RGBA")
        result.putalpha(mask)
        return result

    # ---- AI background generation (/replace-bg-ai) -----------------------

    def _check_bg_model(self, model: str, ctx: dict) -> str:
        m = (model or "auto").strip().lower()
        if m == "auto":
            m = AI_BG_DEFAULT_MODEL
        known = set(AI_BG_AZURE_MODELS) | set(AI_BG_GOOGLE_MODELS) | set(AI_BG_ASU_MODELS)
        if m not in known:
            allowed = ", ".join(sorted(set(AI_BG_AZURE_MODELS) | set(AI_BG_GOOGLE_MODELS)))
            raise HTTPException(400, f"model must be 'auto' or one of: {allowed}")
        if m in AI_BG_ASU_MODELS and not ctx.get("is_legacy"):
            # Owner-only. A university work token must never serve customer
            # traffic, so this is refused even for allowlisted users.
            raise HTTPException(403, "That model is not available on this account")
        return m

    @staticmethod
    def _check_prompt(prompt: str) -> str:
        p = (prompt or "").strip()
        if not p:
            raise HTTPException(400, "prompt is required")
        if len(p) > AI_BG_PROMPT_MAX:
            raise HTTPException(400, f"prompt must be {AI_BG_PROMPT_MAX} characters or fewer")
        return p

    def _require_ai_bg(self, ctx: dict) -> None:
        """Allowlist gate for the AI-background experiment.

        Empty AI_BG_ALLOWLIST = off for everyone, so shipping this code changes
        nothing until a user_id is explicitly added. Turning it off later is a
        secret edit, not a deploy, and no customer can come to depend on it.
        """
        if ctx.get("is_legacy"):
            return
        uid = str(ctx.get("user_id") or "")
        if uid and uid in _ai_bg_allowlist():
            return
        raise HTTPException(
            status_code=402,
            detail="AI backgrounds are in limited preview. Contact support@useknockout.com for access.",
        )

    def _enforce_ai_bg_cap(self) -> None:
        """Global daily cap — every call spends real money at the provider."""
        try:
            cap = int(os.environ.get("AI_BG_DAILY_CAP", "") or AI_BG_DAILY_CAP_DEFAULT)
        except ValueError:
            cap = AI_BG_DAILY_CAP_DEFAULT
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"aibg-{day}"
        try:
            d = modal.Dict.from_name("knockout-stats", create_if_missing=True)
            used = int(d.get(key, 0))
            if used >= cap:
                raise HTTPException(429, "AI background generation has hit today's global limit. Try again tomorrow.")
            d[key] = used + 1
        except HTTPException:
            raise
        except Exception:
            pass  # counter is best-effort; never block on Dict failure

    @staticmethod
    def _bg_aspect(size) -> str:
        """Nearest provider-supported size for the source aspect ratio.

        The backdrop is scaled to cover the real frame afterwards, so it only
        needs the right SHAPE — generating at 1-2MP never caps our output
        resolution the way generative editing would.
        """
        w, h = size
        r = w / float(h) if h else 1.0
        if r >= 1.25:
            return "1536x1024"
        if r <= 0.8:
            return "1024x1536"
        return "1024x1024"

    def _generate_background(self, prompt: str, model: str, size):
        """Text-to-image backdrop, cached by (prompt, model, aspect).

        The cache is the main cost lever: a caller running one prompt across a
        20-image catalog pays for one generation, not twenty.
        """
        aspect = self._bg_aspect(size)
        ckey = "aibg:" + hashlib.sha256(f"{model}|{aspect}|{prompt}".encode("utf-8")).hexdigest()
        cache = None
        try:
            cache = modal.Dict.from_name("knockout-ai-bg-cache", create_if_missing=True)
            hit = cache.get(ckey)
            if hit:
                return Image.open(io.BytesIO(base64.b64decode(hit))).convert("RGB")
        except Exception:
            cache = None  # cache is optional

        if model in AI_BG_ASU_MODELS:
            raw = self._gen_bg_asu(prompt, model)
        elif model in AI_BG_GOOGLE_MODELS:
            raw = self._gen_bg_google(prompt, AI_BG_GOOGLE_MODELS[model], aspect)
        else:
            raw = self._gen_bg_azure(prompt, model, aspect)

        try:
            if cache is not None:
                cache[ckey] = base64.b64encode(raw).decode("ascii")
        except Exception:
            pass
        return Image.open(io.BytesIO(raw)).convert("RGB")

    @staticmethod
    def _azure_services_host(aoai_endpoint: str) -> str:
        """Derive the Foundry services host from the Azure OpenAI endpoint.

        Same resource, different API surface:
          https://ai-agents-os.openai.azure.com
          https://ai-agents-os.services.ai.azure.com
        Overridable via AZURE_FOUNDRY_SERVICES_ENDPOINT when they diverge.
        """
        override = (os.environ.get("AZURE_FOUNDRY_SERVICES_ENDPOINT") or "").rstrip("/")
        if override:
            return override
        return aoai_endpoint.replace(".openai.azure.com", ".services.ai.azure.com")

    def _gen_bg_azure(self, prompt: str, model: str, aspect: str) -> bytes:
        endpoint = (os.environ.get("AZURE_FOUNDRY_ENDPOINT") or "").rstrip("/")
        key = os.environ.get("AZURE_FOUNDRY_API_KEY") or ""
        if not endpoint or not key:
            raise HTTPException(503, "AI backgrounds are not configured on this deployment")
        default_dep, route = AI_BG_AZURE_MODELS[model]
        # Deployment names are chosen at deploy time and need not match the
        # catalog id, so allow an env override per model.
        env_name = "AI_BG_DEPLOYMENT_" + model.upper().replace("-", "_").replace(".", "_")
        deployment = os.environ.get(env_name) or default_dep

        headers = {"Api-Key": key, "Content-Type": "application/json"}
        payload = {"prompt": prompt, "n": 1, "size": aspect}
        if route == "bfl":
            # Black Forest Labs provider passthrough. FLUX.2 is sold directly by
            # Azure but is NOT on the OpenAI-compatible image route — it 404s
            # there. Own host, own path, model slug in the URL, Bearer auth,
            # and width/height instead of `size`.
            slug = deployment.lower().replace(".", "-")  # FLUX.2-pro -> flux-2-pro
            # The docs show *.api.cognitive.microsoft.com, but that hostname does
            # not resolve for this resource — the BFL passthrough lives on the
            # same services.ai host as the other Foundry models.
            host = self._azure_services_host(endpoint)
            url = f"{host}/providers/blackforestlabs/v1/{slug}?api-version=preview"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            w, h = (int(x) for x in aspect.split("x"))
            payload = {"model": deployment, "prompt": prompt, "width": w, "height": h,
                       "output_format": "jpeg"}
        elif route == "aoai":
            url = (f"{endpoint}/openai/deployments/{deployment}"
                   f"/images/generations?api-version={AI_BG_API_VERSION}")
        elif route == "mai":
            url = f"{self._azure_services_host(endpoint)}/mai/v1/images/generations"
            payload["model"] = deployment
        else:  # foundry — partner models sold directly by Azure (BFL FLUX)
            url = (f"{self._azure_services_host(endpoint)}"
                   f"/openai/v1/images/generations?api-version=preview")
            payload["model"] = deployment

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=180)
        except requests.RequestException as e:
            raise HTTPException(502, f"Background generation failed: {e}")
        if r.status_code >= 400:
            raise HTTPException(
                502,
                f"Background generation failed ({r.status_code}) [{route} {deployment}]: {r.text[:180]}",
            )
        body = r.json()
        if route == "bfl":
            # BFL returns its own envelope, not the OpenAI data[] shape.
            found = self._find_image_in_json(body)
            if found is None:
                raise HTTPException(502, "Background generation returned no image")
            return found
        return self._decode_image_response(body)

    def _gen_bg_google(self, prompt: str, model: str, aspect: str) -> bytes:
        key = os.environ.get("GOOGLE_API_KEY") or ""
        if not key:
            raise HTTPException(503, "AI backgrounds are not configured on this deployment")
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={key}")
        try:
            r = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=120,
            )
        except requests.RequestException as e:
            raise HTTPException(502, f"Background generation failed: {e}")
        if r.status_code >= 400:
            raise HTTPException(502, f"Background generation failed ({r.status_code}): {r.text[:200]}")
        body = r.json()
        try:
            for part in body["candidates"][0]["content"]["parts"]:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    return base64.b64decode(inline["data"])
        except (KeyError, IndexError, TypeError):
            pass
        raise HTTPException(502, "Background generation returned no image")

    def _gen_bg_asu(self, prompt: str, model: str) -> bytes:
        """ASU AIML gateway (owner-only evaluation path).

        Response envelope is not formally documented, so the image is located by
        walking the JSON for the first b64 blob or image URL rather than
        assuming a key path.
        """
        token = os.environ.get("ASU_AIML_TOKEN") or ""
        project = os.environ.get("ASU_AIML_PROJECT_ID") or ""
        if not token or not project:
            raise HTTPException(503, "ASU evaluation models are not configured on this deployment")
        provider, name = AI_BG_ASU_MODELS[model]
        try:
            r = requests.post(
                AI_BG_ASU_URL,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={
                    "endpoint": "image",
                    "request_source": "override_params",
                    "query": prompt,
                    "model_provider": provider,
                    "model_name": name,
                    "project_id": project,
                    "enable_history": False,
                    "response_format": {"type": "json"},
                },
                timeout=180,
            )
        except requests.RequestException as e:
            raise HTTPException(502, f"Background generation failed: {e}")
        if r.status_code >= 400:
            raise HTTPException(502, f"Background generation failed ({r.status_code}): {r.text[:200]}")
        found = self._find_image_in_json(r.json())
        if found is None:
            raise HTTPException(502, "Background generation returned no image")
        return found

    def _find_image_in_json(self, node, depth: int = 0):
        """Walk an unknown JSON envelope for image bytes (b64 blob or URL)."""
        if depth > 8:
            return None
        if isinstance(node, str):
            if node.startswith("http") and len(node) < 2000:
                try:
                    ir = requests.get(node, timeout=60)
                    ir.raise_for_status()
                    if ir.headers.get("content-type", "").startswith("image/"):
                        return ir.content
                except requests.RequestException:
                    return None
                return None
            if len(node) > 500:
                s = node.split(",", 1)[-1] if node.startswith("data:") else node
                try:
                    raw = base64.b64decode(s, validate=True)
                except Exception:
                    return None
                return raw if raw[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1", b"RIFF") or raw[:3] == b"\xff\xd8\xff" else None
            return None
        if isinstance(node, dict):
            for v in node.values():
                got = self._find_image_in_json(v, depth + 1)
                if got:
                    return got
        elif isinstance(node, list):
            for v in node:
                got = self._find_image_in_json(v, depth + 1)
                if got:
                    return got
        return None

    @staticmethod
    def _decode_image_response(body: dict) -> bytes:
        """Pull image bytes out of an OpenAI-shaped images response.

        Providers return either inline b64 or a short-lived URL depending on
        model and version, so handle both rather than assuming one.
        """
        try:
            item = body["data"][0]
        except (KeyError, IndexError, TypeError):
            raise HTTPException(502, "Background generation returned no image")
        b64 = item.get("b64_json")
        if b64:
            return base64.b64decode(b64)
        url = item.get("url")
        if url:
            try:
                ir = requests.get(url, timeout=60)
                ir.raise_for_status()
                return ir.content
            except requests.RequestException as e:
                raise HTTPException(502, f"Could not fetch generated background: {e}")
        raise HTTPException(502, "Background generation returned no image")

    @staticmethod
    def _cover_resize(bg, size):
        """Scale-and-center-crop the backdrop to exactly `size` (no distortion)."""
        tw, th = size
        bw, bh = bg.size
        scale = max(tw / float(bw), th / float(bh))
        nw, nh = max(1, int(round(bw * scale))), max(1, int(round(bh * scale)))
        bg = bg.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - tw) // 2, (nh - th) // 2
        return bg.crop((left, top, left + tw, top + th))

    def _composite_on_bg(self, image_obj, bg_image_or_color, despill=None,
                         detect: str = "standard", decontaminate: bool = False):
        """Composite foreground onto a solid color or image background."""
        rgb, mask = self._acquire_mask(image_obj, detect=detect, decontaminate=decontaminate)
        clean_rgb = self._clean_foreground(rgb, mask, strength=self._despill_strength(despill))
        if isinstance(bg_image_or_color, tuple):
            bg = bg_image_or_color  # solid color — scalar fast path in _composite_linear
        else:
            bg = bg_image_or_color.convert("RGB").resize(rgb.size, Image.LANCZOS)
        return self._composite_linear(clean_rgb, bg, mask)

    _SRGB_TO_LIN_LUT = None  # lazy 256-entry uint8 sRGB -> linear float32 table

    @classmethod
    def _srgb_to_lin_lut(cls):
        if cls._SRGB_TO_LIN_LUT is None:
            x = np.arange(256, dtype=np.float32) / 255.0
            cls._SRGB_TO_LIN_LUT = np.where(
                x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4
            ).astype(np.float32)
        return cls._SRGB_TO_LIN_LUT

    def _composite_linear(self, fg_rgb, bg, alpha):
        """Alpha-composite fg over bg in LINEAR light, not gamma-encoded sRGB.

        Blending in sRGB darkens semi-transparent edges, leaving a faint halo
        ring on hair/glass/reflections. Convert to linear light, blend, convert
        back to sRGB — the halo goes away.

        `bg` is an RGB PIL image or a (r, g, b) tuple; a solid color converts
        as 3 scalars and broadcasts, skipping a full-frame background array.
        The uint8 -> linear step is a 256-entry LUT — no per-pixel pow() on
        the way in; only the final linear -> sRGB pass needs real math.
        """
        lut = self._srgb_to_lin_lut()
        fg = lut[np.asarray(fg_rgb.convert("RGB"))]
        if isinstance(bg, tuple):
            bg_lin = lut[np.asarray(bg, dtype=np.uint8)]  # shape (3,) — broadcasts
        else:
            bg_lin = lut[np.asarray(bg.convert("RGB"))]
        al = (np.asarray(alpha.convert("L"), dtype=np.float32) / 255.0)[:, :, None]
        out = fg * al + bg_lin * (1.0 - al)
        out = np.clip(out, 0.0, 1.0)
        out = np.where(out <= 0.0031308, out * 12.92, 1.055 * np.power(out, 1 / 2.4) - 0.055)
        return Image.fromarray((out * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")

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

    # ---- Collage layout ------------------------------------------------------

    _COLLAGE_POSITIONS = frozenset({"TL", "T", "TR", "L", "C", "R", "BL", "B", "BR"})

    @staticmethod
    def _collage_rects(canvas_w: int, canvas_h: int, n_others: int,
                       position: str, main_frac: float = 0.65):
        """Deterministic collage template: a main rect + equal satellite cells.

        The main image anchors at `position` and takes `main_frac` of the
        canvas; satellites fill the leftover space in equal cells — an L-shape
        of two strips for corner anchors, one full-length strip for edge
        anchors, top+bottom strips for center. Fixed templates over generic
        bin-packing: predictable, testable, and it's what the reference
        e-commerce collages actually look like.

        Returns (main_rect, [satellite_rects]) as (x0, y0, x1, y1) pixel boxes.
        """
        W, H = canvas_w, canvas_h
        mw, mh = int(round(W * main_frac)), int(round(H * main_frac))

        def hcells(x0, x1, y0, y1, k):  # k cells left -> right
            cw = (x1 - x0) / k
            return [(int(x0 + i * cw), y0, int(x0 + (i + 1) * cw), y1) for i in range(k)]

        def vcells(x0, x1, y0, y1, k):  # k cells top -> bottom
            ch = (y1 - y0) / k
            return [(x0, int(y0 + i * ch), x1, int(y0 + (i + 1) * ch)) for i in range(k)]

        p = position
        if p in ("T", "B"):
            main = (0, 0, W, mh) if p == "T" else (0, H - mh, W, H)
            sy0, sy1 = (mh, H) if p == "T" else (0, H - mh)
            return main, hcells(0, W, sy0, sy1, n_others)
        if p in ("L", "R"):
            main = (0, 0, mw, H) if p == "L" else (W - mw, 0, W, H)
            sx0, sx1 = (mw, W) if p == "L" else (0, W - mw)
            return main, vcells(sx0, sx1, 0, H, n_others)
        if p == "C":
            main = ((W - mw) // 2, (H - mh) // 2, (W + mw) // 2, (H + mh) // 2)
            top_k = (n_others + 1) // 2
            bot_k = n_others - top_k
            cells = hcells(0, W, 0, (H - mh) // 2, top_k)
            if bot_k:
                cells += hcells(0, W, (H + mh) // 2, H, bot_k)
            return main, cells
        # Corners — satellites fill the L-shape: a full-width strip on the
        # opposite vertical side + a column beside the main block.
        x0 = 0 if "L" in p else W - mw
        y0 = 0 if "T" in p else H - mh
        main = (x0, y0, x0 + mw, y0 + mh)
        sy0, sy1 = (mh, H) if "T" in p else (0, H - mh)      # horizontal strip
        sx0, sx1 = (mw, W) if "L" in p else (0, W - mw)      # vertical strip
        h_area = W * (sy1 - sy0)
        v_area = (sx1 - sx0) * mh
        k_h = min(n_others, max(1, round(n_others * h_area / (h_area + v_area))))
        k_v = n_others - k_h
        cells = hcells(0, W, sy0, sy1, k_h)
        if k_v:
            cells += vcells(sx0, sx1, y0, y0 + mh, k_v)
        return main, cells

    @staticmethod
    def _fit_in_cell(cutout, cell, pad: int):
        """Scale a cutout to fit inside `cell` minus padding, centered.

        Returns (resized_image, (paste_x, paste_y)). Aspect preserved; small
        cutouts are upscaled so cells read uniformly.
        """
        x0, y0, x1, y1 = cell
        avail_w = max(8, x1 - x0 - 2 * pad)
        avail_h = max(8, y1 - y0 - 2 * pad)
        scale = min(avail_w / cutout.width, avail_h / cutout.height)
        nw = max(1, int(round(cutout.width * scale)))
        nh = max(1, int(round(cutout.height * scale)))
        resized = cutout.resize((nw, nh), Image.LANCZOS)
        return resized, (x0 + (x1 - x0 - nw) // 2, y0 + (y1 - y0 - nh) // 2)

    def _inpaint(self, image_obj, mask, dilation: int):
        """
        LaMa-based inpainting with full-resolution preservation.

        Pipeline: dilate mask → downscale to 1024 max-edge → run LaMa →
        upscale result → composite over the original at full resolution so
        unmasked pixels stay byte-identical to input.

        `image_obj` and `mask` are PIL.Image. Returns PIL.Image RGB at
        original full resolution.
        """
        rgb = image_obj.convert("RGB")
        mask_l = mask.convert("L") if mask.mode != "L" else mask

        dilation = max(0, min(int(dilation), 32))
        mask_dilated = self._dilate_mask(mask_l, dilation)

        w, h = rgb.size
        max_dim = 1024
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            sw, sh = int(round(w * scale)), int(round(h * scale))
            rgb_small = rgb.resize((sw, sh), Image.LANCZOS)
            mask_small = mask_dilated.resize((sw, sh), Image.NEAREST)
        else:
            rgb_small = rgb
            mask_small = mask_dilated

        inpainted_small = self.inpainter(rgb_small, mask_small)

        if inpainted_small.size != (w, h):
            inpainted = inpainted_small.resize((w, h), Image.LANCZOS)
        else:
            inpainted = inpainted_small

        # Composite — only use inpainted where dilated mask is non-zero.
        return Image.composite(inpainted, rgb, mask_dilated)

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
    _FORMAT_TO_MEDIA = {
        "png": "image/png",
        "webp": "image/webp",
        "jpg": "image/jpeg",
        "psd": "image/vnd.adobe.photoshop",
    }

    def _encode_psd(self, image_out) -> bytes:
        """Encode a PIL image as a layered Photoshop .psd with real transparency.

        Builds an empty document then adds the cutout as a PixelLayer, so the
        alpha is stored as layer transparency that Photoshop honors. (frompil
        flattens to an opaque Background layer — alpha ignored — so we don't use
        it.) Requires psd-tools >= 1.11 for create_pixel_layer.

        psd-tools imported lazily — it pulls a non-trivial dependency tree we
        only want loaded when PSD is actually requested.
        """
        from psd_tools import PSDImage
        rgba = image_out.convert("RGBA")
        w, h = rgba.size
        psd = PSDImage.new(mode="RGB", size=(w, h), depth=8)
        psd.create_pixel_layer(rgba, name="Cutout", top=0, left=0, opacity=255)
        buf = io.BytesIO()
        psd.save(buf)
        return buf.getvalue()

    def _encode(self, image_out, fmt: str, quality: Optional[int] = None) -> bytes:
        """Encode a PIL image to bytes.

        quality: 1-100 for lossy formats (jpg/webp). Ignored for png (lossless)
        and psd (raw). Defaults: jpg=92, webp=80. Higher = larger, better fidelity.
        """
        if fmt == "psd":
            return self._encode_psd(image_out)
        buf = io.BytesIO()
        pil_fmt = self._FORMAT_TO_PIL[fmt]
        if pil_fmt == "JPEG":
            if image_out.mode != "RGB":
                image_out = image_out.convert("RGB")
            q = 92 if quality is None else max(1, min(int(quality), 100))
            image_out.save(buf, format="JPEG", quality=q, optimize=True)
        elif pil_fmt == "WEBP":
            q = 80 if quality is None else max(1, min(int(quality), 100))
            # method=4 balances encode speed vs size; carries alpha for cutouts.
            image_out.save(buf, format="WEBP", quality=q, method=4)
        else:  # PNG — lossless, quality knob does not apply.
            image_out.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def _response(self, image_out, fmt: str, quality: Optional[int] = None, headers=None):
        content = self._encode(image_out, fmt, quality=quality)
        return Response(content=content, media_type=self._FORMAT_TO_MEDIA[fmt],
                        headers=headers or None)

    def _apply_resize(self, image_obj, max_dim=None, width=None, height=None):
        """Resize output. Precedence: explicit width/height over max_dim.

        - width AND height: exact box (may change aspect).
        - width OR height alone: that axis fixed, other scaled to keep aspect.
        - max_dim: longest side <= max_dim, aspect preserved (down or up).
        All dims clamped to [1, 8000]. No-op if nothing passed.
        """
        w, h = image_obj.size
        if width or height:
            if width and height:
                tw, th = int(width), int(height)
            elif width:
                tw = int(width)
                th = max(1, int(round(h * tw / w)))
            else:
                th = int(height)
                tw = max(1, int(round(w * th / h)))
            tw = max(1, min(tw, 8000))
            th = max(1, min(th, 8000))
            if (tw, th) == (w, h):
                return image_obj
            return image_obj.resize((tw, th), Image.LANCZOS)
        if max_dim:
            md = max(1, min(int(max_dim), 8000))
            scale = md / float(max(w, h))
            if scale == 1.0:
                return image_obj
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            return image_obj.resize((nw, nh), Image.LANCZOS)
        return image_obj

    def _apply_watermark(self, image_obj, text, opacity: float = 0.5):
        """Overlay a text watermark in the bottom-right corner.

        Auto-scales font to the image, white text + 1px dark shadow for
        legibility on any background. opacity 0.0-1.0. No-op if text falsy.
        """
        if not text:
            return image_obj
        text = str(text)[:64]
        orig_mode = image_obj.mode
        base = image_obj.convert("RGBA")
        w, h = base.size
        font_size = max(14, int(min(w, h) * 0.04))
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:  # very old Pillow — no size arg
            font = ImageFont.load_default()
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        margin = max(8, int(min(w, h) * 0.02))
        x = max(0, w - tw - margin)
        y = max(0, h - th - margin)
        a = max(0, min(int(round(float(opacity) * 255)), 255))
        draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, a))
        draw.text((x, y), text, font=font, fill=(255, 255, 255, a))
        out = Image.alpha_composite(base, layer)
        return out if orig_mode == "RGBA" else out.convert(orig_mode)

    def _finalize(self, image_obj, fmt: str, *, quality=None, max_dim=None,
                  width=None, height=None, watermark=None, watermark_opacity=0.5):
        """Shared output path: resize -> watermark -> encode. Returns bytes."""
        image_obj = self._apply_resize(image_obj, max_dim=max_dim, width=width, height=height)
        image_obj = self._apply_watermark(image_obj, watermark, watermark_opacity)
        return self._encode(image_obj, fmt, quality=quality)

    def _finalize_response(self, image_obj, fmt: str, headers=None, **kw):
        content = self._finalize(image_obj, fmt, **kw)
        return Response(content=content, media_type=self._FORMAT_TO_MEDIA[fmt],
                        headers=headers or None)

    @staticmethod
    def _engine_headers(engine_used: str) -> dict:
        """Tell the caller which cutout engine actually ran.

        Only interesting when engine=auto, where the routing decision is ours
        and the caller cannot predict it. Emitted on every engine-aware
        endpoint so clients can log or display it without branching.
        """
        return {"x-knockout-engine": engine_used or "standard"}

    # Output knobs a saved preset may set as defaults. Restricted to params
    # whose endpoint default is None, so an explicit request value can always
    # be told apart from "omitted" and cleanly overrides the preset.
    _PRESET_KEYS = frozenset({"quality", "max_dim", "width", "height", "despill", "watermark"})

    @staticmethod
    def _coalesce(explicit, cfg: dict, key: str):
        """Explicit request value wins; else fall back to the preset config."""
        return explicit if explicit is not None else cfg.get(key)

    def _apply_preset(self, ctx: dict, preset: str, params: dict) -> dict:
        """Overlay a saved preset's defaults onto explicit request params.

        `params` maps preset key -> the request's explicit value (None =
        omitted). Only the keys the calling endpoint supports are passed in,
        and explicit values always win. Single shared path so endpoints can't
        drift on which keys they coalesce.
        """
        cfg = self._get_preset_config(ctx, preset)
        return {k: self._coalesce(v, cfg, k) for k, v in params.items()}

    def _get_preset_config(self, ctx: dict, name: str) -> dict:
        """Fetch a user's named preset config (filtered to _PRESET_KEYS). 404 if absent."""
        user_id = ctx.get("user_id")
        if not user_id:
            raise HTTPException(400, "Presets require a per-user API key")
        status, body = self._supabase_request(
            "GET",
            "/rest/v1/presets",
            params={
                "select": "config",
                "user_id": f"eq.{user_id}",
                "name": f"eq.{name}",
                "limit": "1",
            },
        )
        if status != 200:
            raise HTTPException(503, "Preset service unavailable")
        try:
            rows = json.loads(body) if body else []
        except json.JSONDecodeError:
            rows = []
        if not rows:
            raise HTTPException(404, f"Preset {name!r} not found")
        cfg = rows[0].get("config") or {}
        return {k: v for k, v in cfg.items() if k in self._PRESET_KEYS}

    # ---- Video background removal (async worker) -----------------------------

    @modal.method()
    def process_video_job(self, job_id: str) -> None:
        """Async video job: demux -> per-frame BiRefNet -> temporal smoothing ->
        composite -> remux (+audio) -> upload result -> bill.

        Runs on the same warm class/GPU as the web app. Spawned by
        POST /video/remove; progress + result land in the video_jobs row.
        """
        job = self._job_get(job_id)
        if not job:
            return
        params = job.get("params") or {}
        fmt = params.get("format", "prores4444")
        bg_color = params.get("bg_color")          # None => transparent output
        bg_blur = max(0, min(int(params.get("bg_blur", 0)), 100))
        has_bg_image = bool(params.get("has_bg_image"))
        speed = max(0.25, min(float(params.get("speed", 1.0)), 4.0))
        # Cap at 95: smoothing=100 made every frame reuse the FIRST frame's
        # mask forever (a = 0*current + 1.0*prev) — a frozen matte, not
        # "maximum smoothing". 95 is the highest value that still converges.
        smoothing = max(0, min(int(params.get("smoothing", 30)), 95)) / 100.0
        in_ext = params.get("in_ext", "mp4")

        tmp = tempfile.mkdtemp(prefix=f"vj-{job_id[:8]}-")
        try:
            self._job_update(job_id, status="processing", progress=1)

            # 1. Fetch input
            src = os.path.join(tmp, f"in.{in_ext}")
            with open(src, "wb") as f:
                f.write(self._storage_download(f"{job_id}/in.{in_ext}"))

            # 2. Probe fps (duration was validated at submit)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=avg_frame_rate", "-of", "json", src],
                capture_output=True, text=True, timeout=60)
            try:
                num, den = json.loads(probe.stdout)["streams"][0]["avg_frame_rate"].split("/")
                src_fps = (float(num) / float(den)) if float(den) else VIDEO_FPS_CAP
            except Exception:
                src_fps = VIDEO_FPS_CAP
            fps = min(src_fps or VIDEO_FPS_CAP, VIDEO_FPS_CAP)

            # 3. Demux to frames (downscaled to bound inference cost) + audio
            frames_dir = os.path.join(tmp, "frames")
            os.makedirs(frames_dir)
            scale = f"scale='min({VIDEO_MAX_DIM},iw)':-2"
            subprocess.run(
                ["ffmpeg", "-y", "-i", src, "-vf", f"fps={fps:.3f},{scale}",
                 os.path.join(frames_dir, "%05d.png")],
                capture_output=True, timeout=300, check=True)
            audio = os.path.join(tmp, "audio.m4a")
            has_audio = subprocess.run(
                ["ffmpeg", "-y", "-i", src, "-vn", "-acodec", "aac", audio],
                capture_output=True, timeout=120).returncode == 0

            frame_files = sorted(os.listdir(frames_dir))
            n_frames = len(frame_files)
            if not n_frames:
                raise RuntimeError("no frames decoded")
            self._job_update(job_id, frames=n_frames, progress=5)

            # 4. Per-frame mask + composite. Temporal EMA on the alpha kills
            #    frame-to-frame matte flicker (refine=False: guided filter is
            #    too slow per-frame; EMA covers the jitter instead).
            #    Background precedence: bg_image > bg_blur (blurred source) >
            #    bg_color > transparent.
            out_dir = os.path.join(tmp, "out")
            os.makedirs(out_dir)
            prev_alpha = None
            color = self._parse_color(bg_color) if bg_color else None
            bg_img = None
            if has_bg_image:
                bg_img = Image.open(io.BytesIO(
                    self._storage_download(f"{job_id}/bg.png"))).convert("RGB")
            bg_img_sized = None  # bg_image resized to frame size (cached once)
            # Map 0-100 blur strength to a Gaussian radius that reads as
            # "portrait mode" at 30+ on 1080p frames.
            blur_radius = 2 + (bg_blur / 100.0) * 38 if bg_blur > 0 else 0
            opaque = bool(bg_img or bg_blur > 0 or color is not None)
            for i, name in enumerate(frame_files):
                frame = Image.open(os.path.join(frames_dir, name)).convert("RGB")
                _, mask = self._get_mask(frame, refine=False)
                a = np.asarray(mask, dtype=np.float32)
                if prev_alpha is not None and smoothing > 0:
                    a = (1.0 - smoothing) * a + smoothing * prev_alpha
                prev_alpha = a
                mask = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), mode="L")
                if bg_img is not None:
                    if bg_img_sized is None or bg_img_sized.size != frame.size:
                        bg_img_sized = bg_img.resize(frame.size, Image.LANCZOS)
                    out = self._composite_linear(frame, bg_img_sized, mask)
                elif bg_blur > 0:
                    out = self._composite_linear(
                        frame, frame.filter(ImageFilter.GaussianBlur(blur_radius)), mask)
                elif color is not None:
                    out = self._composite_linear(frame, color, mask)
                else:
                    out = frame.convert("RGBA")
                    out.putalpha(mask)
                out.save(os.path.join(out_dir, name))
                if i % 30 == 0:
                    self._job_update(job_id, progress=5 + int(85 * i / n_frames))

            # 5. Remux. speed != 1 plays the frame sequence at fps*speed and
            #    tempo-shifts the audio to match (atempo chained: each stage
            #    only accepts 0.5-2.0).
            if opaque or fmt == "mp4":
                out_name, vcodec = "out.mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
                content_type = "video/mp4"
            elif fmt == "webm":
                out_name, vcodec = "out.webm", ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "24"]
                content_type = "video/webm"
            else:  # prores4444 — real 10-bit alpha, drops into Resolve/Premiere/AE
                out_name, vcodec = "out.mov", ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"]
                content_type = "video/quicktime"
            cmd = ["ffmpeg", "-y", "-framerate", f"{fps * speed:.3f}",
                   "-i", os.path.join(out_dir, "%05d.png")]
            if has_audio:
                cmd += ["-i", audio]
                if abs(speed - 1.0) < 1e-6:
                    cmd += ["-c:a", "copy"]
                else:
                    stages = []
                    s = speed
                    while s > 2.0:
                        stages.append("atempo=2.0"); s /= 2.0
                    while s < 0.5:
                        stages.append("atempo=0.5"); s /= 0.5
                    stages.append(f"atempo={s:.4f}")
                    cmd += ["-filter:a", ",".join(stages), "-c:a", "aac"]
                cmd += ["-shortest"]
            cmd += vcodec + [os.path.join(tmp, out_name)]
            subprocess.run(cmd, capture_output=True, timeout=600, check=True)

            # 6. Upload result + finish
            with open(os.path.join(tmp, out_name), "rb") as f:
                self._storage_upload(f"{job_id}/{out_name}", f.read(), content_type)
            self._job_update(job_id, status="done", progress=100,
                             result_path=f"{job_id}/{out_name}")

            # 7. Bill on the dedicated video meter: 1 unit = 1 OUTPUT second,
            #    priced at $0.10/second (separate line from the image meter).
            #    speed shortens the output, so it also shrinks the bill.
            seconds = float(job.get("seconds") or (n_frames / fps))
            units = max(1, math.ceil(seconds / speed))
            status_u, body_u = self._supabase_request(
                "GET", "/rest/v1/users",
                params={"id": f"eq.{job['user_id']}", "select": "tier,stripe_customer_id"})
            urow = (json.loads(body_u) or [{}])[0] if status_u == 200 else {}
            ctx = {"user_id": job["user_id"], "token_id": job.get("token_id"),
                   "tier": urow.get("tier", "payg"),
                   "stripe_customer_id": urow.get("stripe_customer_id"),
                   "is_legacy": False}
            self._log_usage(ctx, "/video/remove", 200, 0, units=units,
                            meter_event=VIDEO_METER_EVENT)
        except Exception as e:
            self._job_update(job_id, status="error", error=str(e)[:500])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @modal.asgi_app(label="api")
    def fastapi_app(self):
        web = FastAPI(
            title="useknockout",
            description="State-of-the-art background removal + upscaling + colorization API.",
            version="0.11.0",
        )

        web.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["POST", "GET", "DELETE"],
            allow_headers=["*"],
        )

        class UrlBody(BaseModel):
            url: HttpUrl
            format: str = "png"
            edge: str = "soft"
            detect: str = "standard"
            decontaminate: bool = False
            engine: str = "standard"
            quality: Optional[int] = None
            max_dim: Optional[int] = None
            width: Optional[int] = None
            height: Optional[int] = None
            despill: Optional[float] = None
            watermark: Optional[str] = None
            watermark_opacity: float = 0.5
            preset: Optional[str] = None

        class BatchUrlBody(BaseModel):
            urls: List[HttpUrl]
            format: str = "png"

        class EstimateBody(BaseModel):
            endpoint: str
            width: int
            height: int

        class PresetBody(BaseModel):
            name: str
            config: dict = {}

        @web.get("/")
        def root():
            return {
                "name": "useknockout",
                "version": "0.13.0",
                "endpoints": [
                    "POST /remove",
                    "POST /remove-url",
                    "POST /psd",
                    "POST /replace-bg",
                    "POST /replace-bg-ai",
                    "POST /remove-batch",
                    "POST /remove-batch-url",
                    "POST /mask",
                    "POST /inpaint",
                    "POST /smart-crop",
                    "POST /shadow",
                    "POST /sticker",
                    "POST /outline",
                    "POST /silhouette",
                    "POST /studio-shot",
                    "POST /collage",
                    "POST /video/remove",
                    "GET /jobs/{job_id}",
                    "POST /compare",
                    "POST /headshot",
                    "POST /preview",
                    "POST /upscale",
                    "POST /face-restore",
                    "POST /colorize",
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

        # GET info handlers for every POST-only endpoint. Converts the 405s
        # browsers / curl visitors produce into 200s with a working curl
        # example. Same pattern as /remove and /remove-url above; registered
        # in a loop to keep the surface area in one place. To add a new
        # endpoint, append to POST_ONLY_INFO and the GET handler appears.
        POST_ONLY_INFO = {
            "/replace-bg": ("multipart 'file' + 'bg_color' hex OR 'bg_url'", "client.replaceBackground({ file, bgColor: '#FFFFFF' })"),
            "/replace-bg-ai": ("multipart 'file' + 'prompt' + optional 'model' (limited preview)", "curl -F file=@p.jpg -F prompt='marble counter' .../replace-bg-ai"),
            "/remove-batch": ("multipart 'files' (1..10 images)", "client.removeBatch({ files: ['./a.jpg', './b.jpg'] })"),
            "/remove-batch-url": ("JSON { urls: string[1..10], format }", "client.removeBatchUrl({ urls: ['https://...'] })"),
            "/mask": ("multipart 'file' — returns grayscale alpha matte", "client.mask({ file })"),
            "/psd": ("multipart 'file' — layered .psd export, premium add-on (2x)", "client.psd({ file, despill: 80 })"),
            "/smart-crop": ("multipart 'file' + 'padding' + 'transparent'", "client.smartCrop({ file, padding: 24 })"),
            "/shadow": ("multipart 'file' + bg/shadow color & offset params", "client.shadow({ file, shadowOffsetY: 12 })"),
            "/sticker": ("multipart 'file' + 'stroke_color' + 'stroke_width'", "client.sticker({ file, strokeWidth: 24 })"),
            "/outline": ("multipart 'file' + 'outline_color' + 'outline_width'", "client.outline({ file, outlineColor: '#000000' })"),
            "/silhouette": ("multipart 'file' + 'subject_color' + 'bg_color'", "client.silhouette({ file, subjectColor: '#1E2960', bgColor: '#F0857C' })"),
            "/inpaint": ("multipart 'file' + optional 'mask' OR 'x,y,w,h' bbox; 'dilation' 0..32", "client.inpaint({ file, mask, dilation: 8 })"),
            "/studio-shot": ("multipart 'file' + 'bg_color' + 'aspect' + 'padding' + 'shadow' + 'transparent' + 'enhance'", "client.studioShot({ file, aspect: '1:1', transparent: true })"),
            "/collage": ("multipart 'files' (2-9) + 'main_index' + 'main_position' (TL..BR|C) — paid tiers, billed N units", "client.collage({ files, mainPosition: 'BR' })"),
            "/video/remove": ("multipart 'file' (mp4/mov/avi/webm/mkv, 15s max) + 'format' (prores4444|webm|mp4) + 'bg_color'|'bg_image'|'bg_blur' + 'speed' (0.25-4) + 'smoothing' — async, $0.10/output-sec, paid tiers", "POST /video/remove -> {job_id}, then GET /jobs/{job_id}"),
            "/compare": ("multipart 'file' — returns side-by-side preview", "client.compare({ file })"),
            "/headshot": ("multipart 'file' + 'bg_color' or 'bg_blur' + 'aspect'", "client.headshot({ file, bgBlur: true })"),
            "/preview": ("multipart 'file' + 'max_dim' (64..1024)", "client.preview({ file, maxDim: 512 })"),
            "/upscale": ("multipart 'file' + 'scale' (2|4) + 'model' (realesrgan[default]|swin2sr)", "client.upscale({ file, scale: 4 })"),
            "/face-restore": ("multipart 'file' + 'only_center_face' + 'bg_enhance'", "client.faceRestore({ file })"),
            "/colorize": ("multipart 'file' — DDColor grayscale→color", "client.colorize({ file })"),
        }

        def _make_post_info_handler(p: str, body: str, sdk: str):
            def info_handler():
                return {
                    "error": "Use POST with multipart/form-data",
                    "method": "POST",
                    "path": p,
                    "body": body,
                    "headers": {"Authorization": "Bearer <token>"},
                    "example_curl": (
                        f"curl -X POST https://useknockout--api.modal.run{p} "
                        f"-H 'Authorization: Bearer <token>' "
                        f"-F 'file=@image.jpg' -o out.png"
                    ),
                    "sdk_example": sdk,
                    "docs": "https://useknockout--api.modal.run/docs",
                }
            info_handler.__name__ = "info_" + p.lstrip("/").replace("-", "_")
            return info_handler

        for _p, (_body, _sdk) in POST_ONLY_INFO.items():
            web.get(_p)(_make_post_info_handler(_p, _body, _sdk))

        # /estimate takes JSON not multipart — its GET-info shape differs.
        @web.get("/estimate")
        def estimate_info():
            return {
                "error": "Use POST with JSON body",
                "method": "POST",
                "body": {"endpoint": "remove", "width": 1024, "height": 1024},
                "headers": {"Authorization": "Bearer <token>", "Content-Type": "application/json"},
                "docs": "https://useknockout--api.modal.run/docs",
                "sdk": "client.estimate({ endpoint: 'remove', width: 1024, height: 1024 })",
            }

        @web.get("/presets")
        def list_presets(authorization: Optional[str] = Header(default=None)):
            """List the calling user's saved presets."""
            ctx = self._check_auth(authorization)
            user_id = ctx.get("user_id")
            if not user_id:
                raise HTTPException(400, "Presets require a per-user API key")
            status, body = self._supabase_request(
                "GET",
                "/rest/v1/presets",
                params={
                    "select": "name,config,created_at,updated_at",
                    "user_id": f"eq.{user_id}",
                    "order": "name",
                },
            )
            if status != 200:
                raise HTTPException(503, "Preset service unavailable")
            try:
                rows = json.loads(body) if body else []
            except json.JSONDecodeError:
                rows = []
            return {"presets": rows}

        @web.post("/presets")
        def upsert_preset(body: PresetBody, authorization: Optional[str] = Header(default=None)):
            """Create or update a named preset (upsert on user_id + name).

            config accepts: quality, max_dim, width, height, despill, watermark.
            Unknown keys are dropped. A preset sets defaults; explicit request
            params always override it.
            """
            ctx = self._check_auth(authorization)
            self._require_pro(ctx, "Saved presets")
            user_id = ctx.get("user_id")
            if not user_id:
                raise HTTPException(400, "Presets require a per-user API key")
            name = (body.name or "").strip()
            if not name or len(name) > 64:
                raise HTTPException(400, "Preset name required (1-64 chars)")
            cfg = {k: v for k, v in (body.config or {}).items() if k in self._PRESET_KEYS}
            status, rbody = self._supabase_request(
                "POST",
                "/rest/v1/presets",
                params={"on_conflict": "user_id,name"},
                body={"user_id": user_id, "name": name, "config": cfg, "updated_at": _now_iso()},
                prefer="resolution=merge-duplicates,return=representation",
            )
            if status not in (200, 201):
                raise HTTPException(503, "Could not save preset")
            try:
                rows = json.loads(rbody) if rbody else []
            except json.JSONDecodeError:
                rows = []
            return {"preset": rows[0] if rows else {"name": name, "config": cfg}}

        @web.delete("/presets/{name}")
        def delete_preset(name: str, authorization: Optional[str] = Header(default=None)):
            """Delete one of the caller's presets by name."""
            ctx = self._check_auth(authorization)
            user_id = ctx.get("user_id")
            if not user_id:
                raise HTTPException(400, "Presets require a per-user API key")
            status, _b = self._supabase_request(
                "DELETE",
                "/rest/v1/presets",
                params={"user_id": f"eq.{user_id}", "name": f"eq.{name}"},
            )
            if status not in (200, 204):
                raise HTTPException(503, "Could not delete preset")
            return {"deleted": name}

        @web.post("/remove")
        def remove_endpoint(
            file: UploadFile = File(...),
            format: str = Form("png"),
            quality: Optional[int] = Form(None),
            max_dim: Optional[int] = Form(None),
            width: Optional[int] = Form(None),
            height: Optional[int] = Form(None),
            despill: Optional[float] = Form(None),
            edge: str = Form("soft"),
            detect: str = Form("standard"),
            decontaminate: bool = Form(False),
            engine: str = Form("standard"),
            watermark: Optional[str] = Form(None),
            watermark_opacity: float = Form(0.5),
            preset: Optional[str] = Form(None),
            authorization: Optional[str] = Header(default=None),
            request: Request = None,
        ):
            # Modal strips client-sent X-Forwarded-For (verified 2026-08-23) and
            # exposes the true client address on request.client — unspoofable.
            client_ip = request.client.host if (request and request.client) else None
            ctx, _t = self._begin(authorization, "/remove", client_ip=client_ip)
            if despill is not None or watermark or preset:
                self._require_pro(ctx, "Premium output (despill, watermark, presets)")
            if preset:
                p = self._apply_preset(ctx, preset, {
                    "quality": quality, "max_dim": max_dim, "width": width,
                    "height": height, "despill": despill, "watermark": watermark,
                })
                quality, max_dim, width, height, despill, watermark = (
                    p["quality"], p["max_dim"], p["width"], p["height"],
                    p["despill"], p["watermark"],
                )
            fmt = self._check_format(format)
            edge = self._check_edge(edge)
            detect = self._check_detect(detect)
            engine = self._check_engine(engine)
            if detect == "high_recall":
                self._require_paid_compute(ctx, "detect=high_recall")
            if engine in ("product-v1", "auto"):
                self._require_paid_compute(ctx, f"engine={engine}")
            data = file.file.read()
            image_obj = self._open_image(data)
            _info = {}
            result = self._remove(image_obj, despill=despill, edge=edge, detect=detect,
                                  decontaminate=decontaminate, engine=engine, info=_info)
            if ctx.get("is_demo"):
                result = self._downscale_max(result, DEMO_MAX_DIM)
                # Resize params could upscale right past the demo cap — drop them.
                max_dim = width = height = None
            resp = self._finalize_response(
                result, fmt, quality=quality, max_dim=max_dim, width=width,
                height=height, watermark=watermark, watermark_opacity=watermark_opacity,
                headers=self._engine_headers(_info.get("engine")),
            )
            self._end(ctx, "/remove", _t)
            return resp

        @web.post("/remove-url")
        def remove_url_endpoint(
            body: UrlBody,
            authorization: Optional[str] = Header(default=None),
        ):
            ctx, _t = self._begin(authorization, "/remove-url")
            # Param parity with multipart /remove (same gates, same pipeline).
            quality, max_dim, width, height = body.quality, body.max_dim, body.width, body.height
            despill, watermark = body.despill, body.watermark
            if despill is not None or watermark or body.preset:
                self._require_pro(ctx, "Premium output (despill, watermark, presets)")
            if body.preset:
                p = self._apply_preset(ctx, body.preset, {
                    "quality": quality, "max_dim": max_dim, "width": width,
                    "height": height, "despill": despill, "watermark": watermark,
                })
                quality, max_dim, width, height, despill, watermark = (
                    p["quality"], p["max_dim"], p["width"], p["height"],
                    p["despill"], p["watermark"],
                )
            fmt = self._check_format(body.format)
            # Validate options BEFORE the outbound fetch — an invalid edge or
            # detect must 400 without us making a network request or image work.
            edge = self._check_edge(body.edge)
            detect = self._check_detect(body.detect)
            engine = self._check_engine(body.engine)
            if detect == "high_recall":
                self._require_paid_compute(ctx, "detect=high_recall")
            if engine in ("product-v1", "auto"):
                self._require_paid_compute(ctx, f"engine={engine}")

            try:
                resp = requests.get(str(body.url), timeout=15)
                resp.raise_for_status()
            except requests.RequestException as e:
                raise HTTPException(400, f"Could not fetch image: {e}")

            image_obj = self._open_image(resp.content)
            _info = {}
            result = self._remove(image_obj, despill=despill, edge=edge, detect=detect,
                                  decontaminate=body.decontaminate, engine=engine, info=_info)
            out_resp = self._finalize_response(
                result, fmt, quality=quality, max_dim=max_dim, width=width,
                height=height, watermark=watermark, watermark_opacity=body.watermark_opacity,
                headers=self._engine_headers(_info.get("engine")),
            )
            self._end(ctx, "/remove-url", _t)
            return out_resp

        @web.post("/psd")
        def psd_endpoint(
            file: UploadFile = File(...),
            despill: Optional[float] = Form(None),
            max_dim: Optional[int] = Form(None),
            width: Optional[int] = Form(None),
            height: Optional[int] = Form(None),
            watermark: Optional[str] = Form(None),
            watermark_opacity: float = Form(0.5),
            preset: Optional[str] = Form(None),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Layered Photoshop (.psd) export — transparent cutout on its own layer,
            ready to edit in Photoshop/Affinity.

            Premium add-on: billed at 2x base ($0.10/image). Output is always PSD.
            Paid tiers only. Supports despill, resize, watermark, and presets.
            """
            ctx, _t = self._begin(authorization, "/psd")
            status_code = 200
            try:
                if despill is not None or watermark or preset:
                    self._require_pro(ctx, "Premium output (despill, watermark, presets)")
                if preset:
                    p = self._apply_preset(ctx, preset, {
                        "max_dim": max_dim, "width": width, "height": height,
                        "despill": despill, "watermark": watermark,
                    })
                    max_dim, width, height, despill, watermark = (
                        p["max_dim"], p["width"], p["height"],
                        p["despill"], p["watermark"],
                    )
                data = file.file.read()
                image_obj = self._open_image(data)
                result = self._remove(image_obj, despill=despill)
                return self._finalize_response(
                    result, "psd", max_dim=max_dim, width=width, height=height,
                    watermark=watermark, watermark_opacity=watermark_opacity,
                )
            except HTTPException as e:
                status_code = e.status_code
                raise
            except Exception:
                status_code = 500
                raise
            finally:
                # Flat $0.10 via its own psd.exported meter (not the base image
                # meter). Included free for Knockout Plus (tier 'pro'). Only
                # fires on 2xx — failures aren't billed.
                is_plus = ctx.get("tier") == "pro"
                self._end(ctx, "/psd", _t, status_code,
                          meter_event="psd.exported", skip_meter=is_plus)

        @web.post("/replace-bg")
        def replace_bg_endpoint(
            file: UploadFile = File(...),
            bg_color: str = Form("#FFFFFF"),
            bg_url: Optional[str] = Form(None),
            format: str = Form("png"),
            quality: Optional[int] = Form(None),
            max_dim: Optional[int] = Form(None),
            width: Optional[int] = Form(None),
            height: Optional[int] = Form(None),
            despill: Optional[float] = Form(None),
            watermark: Optional[str] = Form(None),
            watermark_opacity: float = Form(0.5),
            preset: Optional[str] = Form(None),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Remove the background and composite the subject onto a new background.

            Provide either:
              - bg_color: hex color (default #FFFFFF). Examples: "#000000", "#ff5733".
              - bg_url: URL of a background image (takes precedence over bg_color).

            Output is opaque (no alpha). Use `format=jpg` for smallest file size.
            """
            ctx, _t = self._begin(authorization, "/replace-bg")
            if despill is not None or watermark or preset:
                self._require_pro(ctx, "Premium output (despill, watermark, presets)")
            if preset:
                p = self._apply_preset(ctx, preset, {
                    "quality": quality, "max_dim": max_dim, "width": width,
                    "height": height, "despill": despill, "watermark": watermark,
                })
                quality, max_dim, width, height, despill, watermark = (
                    p["quality"], p["max_dim"], p["width"], p["height"],
                    p["despill"], p["watermark"],
                )
            fmt = self._check_format(format, allowed=frozenset({"png", "webp", "jpg"}))

            data = file.file.read()
            fg = self._open_image(data)

            if bg_url:
                try:
                    bg_resp = requests.get(bg_url, timeout=15)
                    bg_resp.raise_for_status()
                    bg = self._open_image(bg_resp.content)
                    composited = self._composite_on_bg(fg, bg, despill=despill)
                except requests.RequestException as e:
                    raise HTTPException(400, f"Could not fetch bg_url: {e}")
            else:
                color = self._parse_color(bg_color)
                composited = self._composite_on_bg(fg, color, despill=despill)

            if ctx.get("is_demo"):
                composited = self._downscale_max(composited, DEMO_MAX_DIM)
                # Resize params could upscale right past the demo cap — drop them.
                max_dim = width = height = None
            resp = self._finalize_response(
                composited, fmt, quality=quality, max_dim=max_dim, width=width,
                height=height, watermark=watermark, watermark_opacity=watermark_opacity,
            )
            self._end(ctx, "/replace-bg", _t)
            return resp

        @web.post("/replace-bg-ai")
        def replace_bg_ai_endpoint(
            file: UploadFile = File(...),
            prompt: str = Form(...),
            model: str = Form("auto"),
            format: str = Form("jpg"),
            detect: str = Form("standard"),
            decontaminate: bool = Form(False),
            quality: Optional[int] = Form(None),
            max_dim: Optional[int] = Form(None),
            width: Optional[int] = Form(None),
            height: Optional[int] = Form(None),
            despill: Optional[float] = Form(None),
            watermark: Optional[str] = Form(None),
            watermark_opacity: float = Form(0.5),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Replace the background with an AI-generated scene from a text prompt.

            LIMITED PREVIEW — allowlist only.

            The generative model never sees your product. We cut the subject out
            with BiRefNet, generate only the backdrop from `prompt`, then
            composite. So product pixels are unchanged and output stays at source
            resolution — unlike generative image editing, which repaints the
            whole frame and caps resolution.

            `prompt`: scene description, e.g. "white marble countertop, soft
                window light from the left". Max 500 characters.
            `model`: `auto` (default) or an explicit backend —
                flux2-pro, flux2-flex, flux1-kontext-pro, gpt-image-2,
                mai-image-2e, mai-image-2.5, mai-image-2.5-pro,
                mai-image-2.5-flash, nano-banana.
            `detect` / `decontaminate`: same as /remove; they control the cutout,
                not the backdrop.

            Backdrops are cached by (prompt, model, aspect), so running one
            prompt across a catalog generates once.
            """
            ctx, _t = self._begin(authorization, "/replace-bg-ai")
            self._require_ai_bg(ctx)
            prompt = self._check_prompt(prompt)
            model = self._check_bg_model(model, ctx)
            detect = self._check_detect(detect)
            if detect == "high_recall":
                self._require_paid_compute(ctx, "detect=high_recall")
            if despill is not None or watermark:
                self._require_pro(ctx, "Premium output (despill, watermark)")
            fmt = self._check_format(format, allowed=frozenset({"png", "webp", "jpg"}))

            data = file.file.read()
            fg = self._open_image(data)
            self._enforce_ai_bg_cap()

            bg = self._generate_background(prompt, model, fg.size)
            bg = self._cover_resize(bg, fg.size)
            composited = self._composite_on_bg(
                fg, bg, despill=despill, detect=detect, decontaminate=decontaminate,
            )

            resp = self._finalize_response(
                composited, fmt, quality=quality, max_dim=max_dim, width=width,
                height=height, watermark=watermark, watermark_opacity=watermark_opacity,
            )
            # LIMITED PREVIEW: usage row still written (so we can see volume and
            # who used it), but NO Stripe meter fires — provider cost is
            # $0.02-0.19 per background, well above any current per-image price,
            # so charging the normal rate would lose money on every call. Troy
            # absorbs the experiment. Remove skip_meter and add a dedicated
            # meter/price before this leaves preview.
            self._end(ctx, "/replace-bg-ai", _t, skip_meter=True)
            return resp

        @web.post("/remove-batch")
        def remove_batch_endpoint(
            files: List[UploadFile] = File(...),
            format: str = Form("png"),  # was a bare str = query param; every other option is Form
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Remove backgrounds from up to 10 images in one call.

            Returns JSON: {"count": N, "results": [{filename, success, format, data_base64 | error}]}.
            """
            ctx, _t = self._begin(authorization, "/remove-batch")
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

            resp = {"count": len(results), "format": fmt, "results": results}
            self._end(ctx, "/remove-batch", _t)
            return resp

        @web.post("/mask")
        def mask_endpoint(
            file: UploadFile = File(...),
            format: str = Form("png"),
            engine: str = Form("standard"),
            authorization: Optional[str] = Header(default=None),
        ):
            """Return just the alpha mask as a grayscale PNG/WebP (0 = bg, 255 = subject)."""
            ctx, _t = self._begin(authorization, "/mask")
            fmt = self._check_format(format)
            engine = self._check_engine(engine)
            if engine in ("product-v1", "auto"):
                self._require_paid_compute(ctx, f"engine={engine}")
            data = file.file.read()
            image_obj = self._open_image(data)
            if engine == "product-v1":
                _, mask = self._product_mask(image_obj)
                used = "product-v1"
            elif engine == "auto":
                _, mask, used = self._acquire_mask_auto(image_obj, "standard", False)
            else:
                _, mask = self._get_mask(image_obj)
                used = "standard"
            out = mask.convert("L")
            if ctx.get("is_demo"):
                out = self._downscale_max(out, DEMO_MAX_DIM)
            resp = self._response(out, fmt, headers=self._engine_headers(used))
            self._end(ctx, "/mask", _t)
            return resp

        @web.post("/smart-crop")
        def smart_crop_endpoint(
            file: UploadFile = File(...),
            padding: int = Form(24),
            transparent: bool = Form(True),
            format: str = Form("png"),
            detect: str = Form("standard"),
            decontaminate: bool = Form(False),
            engine: str = Form("standard"),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Auto-crop to the subject's tight bounding box + padding (pixels).

            `transparent=true` (default): return cropped cutout with transparent background.
            `transparent=false`: return cropped region from the original image (bg preserved).
            `detect` / `decontaminate`: same as /remove — decontaminate also
            tightens the crop box, since background strips no longer inflate it.
            """
            ctx, _t = self._begin(authorization, "/smart-crop")
            detect = self._check_detect(detect)
            engine = self._check_engine(engine)
            if detect == "high_recall":
                self._require_paid_compute(ctx, "detect=high_recall")
            if engine in ("product-v1", "auto"):
                self._require_paid_compute(ctx, f"engine={engine}")
            allowed = frozenset({"png", "webp", "jpg"}) if not transparent else frozenset({"png", "webp"})
            fmt = self._check_format(format, allowed=allowed)
            data = file.file.read()
            image_obj = self._open_image(data)
            if engine == "product-v1":
                rgb, mask = self._product_mask(image_obj)
                used = "product-v1"
            elif engine == "auto":
                rgb, mask, used = self._acquire_mask_auto(image_obj, detect, decontaminate)
            else:
                rgb, mask = self._acquire_mask(image_obj, detect=detect, decontaminate=decontaminate)
                used = "standard"

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

            resp = self._response(cropped, fmt, headers=self._engine_headers(used))
            self._end(ctx, "/smart-crop", _t)
            return resp

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
            ctx, _t = self._begin(authorization, "/shadow")
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
            resp = self._response(composed, fmt)
            self._end(ctx, "/shadow", _t)
            return resp

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
            ctx, _t = self._begin(authorization, "/sticker")
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
            if ctx.get("is_demo"):
                out = self._downscale_max(out, DEMO_MAX_DIM)
            resp = self._response(out, fmt)
            self._end(ctx, "/sticker", _t)
            return resp

        @web.post("/outline")
        def outline_endpoint(
            file: UploadFile = File(...),
            outline_color: str = Form("#000000"),
            outline_width: int = Form(4),
            format: str = Form("png"),
            authorization: Optional[str] = Header(default=None),
        ):
            """Subject on transparent bg with a thin configurable outline."""
            ctx, _t = self._begin(authorization, "/outline")
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
            resp = self._response(out, fmt)
            self._end(ctx, "/outline", _t)
            return resp

        @web.post("/silhouette")
        def silhouette_endpoint(
            file: UploadFile = File(...),
            subject_color: str = Form("#7C3AED"),
            bg_color: str = Form("#FFFFFF"),
            format: str = Form("png"),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Two-tone silhouette portrait — subject filled with one solid color,
            background filled with another. Apple Music / Spotify avatar style.

            Use for stylized profile pictures, podcast cover art, anonymized
            portraits, or branding placeholders. Reuses BiRefNet for the mask
            then composites with two flat colors — no extra model load.
            """
            ctx, _t = self._begin(authorization, "/silhouette")
            fmt = self._check_format(format, allowed=frozenset({"png", "webp", "jpg"}))
            data = file.file.read()
            image_obj = self._open_image(data)
            rgb, mask = self._get_mask(image_obj)

            bg_rgb = Image.new("RGB", rgb.size, self._parse_color(bg_color))
            subject_rgb = Image.new("RGB", rgb.size, self._parse_color(subject_color))
            bg_rgb.paste(subject_rgb, (0, 0), mask)

            self._bump_counter()
            resp = self._response(bg_rgb, fmt)
            self._end(ctx, "/silhouette", _t)
            return resp

        @web.post("/inpaint")
        def inpaint_endpoint(
            file: UploadFile = File(...),
            mask: Optional[UploadFile] = File(default=None),
            x: Optional[int] = Form(default=None),
            y: Optional[int] = Form(default=None),
            w: Optional[int] = Form(default=None),
            h: Optional[int] = Form(default=None),
            dilation: int = Form(8),
            format: str = Form("png"),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            LaMa-based image inpainting. Three modes, auto-detected:

            1. `mask` field present → user-supplied mask (white = inpaint, black = keep)
            2. `x,y,w,h` fields present → bbox mode (rectangular region)
            3. Neither → auto-subject mode (BiRefNet derives subject mask, inverts it)

            `dilation` (default 8) expands the mask by N pixels before inpainting.
            Higher values reduce ghost outlines from tight masks; max 32.
            """
            ctx, _t = self._begin(authorization, "/inpaint")
            fmt = self._check_format(format, allowed=frozenset({"png", "webp", "jpg"}))
            if not (0 <= int(dilation) <= 32):
                raise HTTPException(400, "dilation must be in range 0..32")

            data = file.file.read()
            image_obj = self._open_image(data)
            img_w, img_h = image_obj.size

            # Determine mode + build mask
            mode: str
            if mask is not None:
                # Mode: user-supplied mask
                mode = "mask"
                mask_data = mask.file.read()
                try:
                    mask_pil = Image.open(io.BytesIO(mask_data)).convert("L")
                except (UnidentifiedImageError, OSError):
                    raise HTTPException(400, "Invalid or unsupported mask image")
                if mask_pil.size != (img_w, img_h):
                    mask_pil = mask_pil.resize((img_w, img_h), Image.NEAREST)
            elif any(v is not None for v in (x, y, w, h)):
                # Mode: bbox
                if any(v is None for v in (x, y, w, h)):
                    raise HTTPException(400, "bbox mode requires all four of x, y, w, h")
                if w <= 0 or h <= 0:
                    raise HTTPException(400, "bbox w and h must be > 0")
                if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                    raise HTTPException(400, "bbox extends outside the image")
                mode = "bbox"
                mask_pil = Image.new("L", (img_w, img_h), 0)
                ImageDraw.Draw(mask_pil).rectangle([x, y, x + w - 1, y + h - 1], fill=255)
            else:
                # Mode: auto-subject (BiRefNet → invert)
                mode = "auto-subject"
                # Region-only use (inverted + dilated below) — skip edge refinement.
                _rgb, subject_mask = self._get_mask(image_obj, refine=False)
                mask_arr = np.asarray(subject_mask.convert("L"), dtype=np.uint8)
                if mask_arr.max() == 0:
                    raise HTTPException(422, "No subject detected. Send mask or bbox.")
                inverted_arr = 255 - mask_arr
                mask_pil = Image.fromarray(inverted_arr, mode="L")

            # Reject empty masks
            mask_arr_check = np.asarray(mask_pil.convert("L"), dtype=np.uint8)
            if mask_arr_check.max() == 0:
                raise HTTPException(400, "Mask has no pixels to inpaint.")

            # Warn (don't reject) if mask covers >50% of image
            warn_header = None
            white_frac = (mask_arr_check > 127).sum() / mask_arr_check.size
            if white_frac > 0.5:
                warn_header = f"mask covers {white_frac:.0%} of the image; LaMa quality degrades on very large masks"

            try:
                inpainted = self._inpaint(image_obj, mask_pil, dilation)
            except RuntimeError as e:
                raise HTTPException(500, f"inpaint failed: {e}")

            self._bump_counter()
            content = self._encode(inpainted, fmt)
            headers = {
                "x-knockout-model": "big-lama",
                "x-knockout-mode": mode,
            }
            if warn_header:
                headers["x-knockout-warning"] = warn_header
            resp = Response(content=content, media_type=self._FORMAT_TO_MEDIA[fmt], headers=headers)
            self._end(ctx, "/inpaint", _t)
            return resp

        @web.post("/studio-shot")
        def studio_shot_endpoint(
            file: UploadFile = File(...),
            bg_color: str = Form("#FFFFFF"),
            aspect: str = Form("1:1"),
            padding: int = Form(48),
            shadow: bool = Form(True),
            transparent: bool = Form(False),
            enhance: bool = Form(False),
            enhance_strength: float = Form(0.15),
            format: str = Form("jpg"),
            quality: Optional[int] = Form(None),
            max_dim: Optional[int] = Form(None),
            width: Optional[int] = Form(None),
            height: Optional[int] = Form(None),
            despill: Optional[float] = Form(None),
            watermark: Optional[str] = Form(None),
            watermark_opacity: float = Form(0.5),
            preset: Optional[str] = Form(None),
            detect: str = Form("standard"),
            decontaminate: bool = Form(False),
            engine: str = Form("standard"),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            E-commerce preset — cutout → tight crop → centered on bg with shadow → standard aspect.

            `aspect`: "1:1", "4:5", "16:9", "3:2", or "W:H" (ints). Default 1:1.
            `transparent`: if true, keep a transparent background (bg_color and
                shadow are ignored). Output is forced to PNG when a non-alpha
                format (jpg) is requested, since jpg can't carry transparency.
            `enhance`: if true, apply a subtle brightness + saturation lift to
                the subject for ecommerce-ready output. Default off.
            `enhance_strength`: 0.0–0.5 lift amount. Default 0.15 (subtle).
            """
            ctx, _t = self._begin(authorization, "/studio-shot")
            status_code = 200
            try:
                detect = self._check_detect(detect)
                engine = self._check_engine(engine)
                if detect == "high_recall":
                    self._require_paid_compute(ctx, "detect=high_recall")
                if engine in ("product-v1", "auto"):
                    self._require_paid_compute(ctx, f"engine={engine}")
                if despill is not None or watermark or preset:
                    self._require_pro(ctx, "Premium output (despill, watermark, presets)")
                if preset:
                    p = self._apply_preset(ctx, preset, {
                        "quality": quality, "max_dim": max_dim, "width": width,
                        "height": height, "despill": despill, "watermark": watermark,
                    })
                    quality, max_dim, width, height, despill, watermark = (
                        p["quality"], p["max_dim"], p["width"], p["height"],
                        p["despill"], p["watermark"],
                    )
                fmt = self._check_format(format, allowed=frozenset({"png", "webp", "jpg"}))
                if transparent and fmt == "jpg":
                    fmt = "png"  # jpg can't carry alpha — coerce to a lossless alpha format
                data = file.file.read()
                image_obj = self._open_image(data)
                if engine == "product-v1":
                    rgb, mask = self._product_mask(image_obj)
                    used = "product-v1"
                elif engine == "auto":
                    rgb, mask, used = self._acquire_mask_auto(image_obj, detect, decontaminate)
                else:
                    rgb, mask = self._acquire_mask(image_obj, detect=detect, decontaminate=decontaminate)
                    used = "standard"

                try:
                    aw_str, ah_str = aspect.split(":")
                    aw, ah = int(aw_str), int(ah_str)
                    if aw <= 0 or ah <= 0:
                        raise ValueError()
                except Exception:
                    raise HTTPException(400, "aspect must be in 'W:H' format, e.g. '1:1' or '4:5'")

                # Clamp aspect ratio — block extreme stretches that blow up canvas size.
                if not (0.2 <= aw / ah <= 5.0):
                    raise HTTPException(400, "aspect ratio must be between 1:5 and 5:1")

                bbox = self._bounding_box(mask)
                if bbox is None:
                    raise HTTPException(400, "No subject detected in image")
                left, top, right, bottom = bbox

                clean_rgb = self._clean_foreground(rgb, mask, strength=self._despill_strength(despill))

                # Ecommerce-ready pre-pass: subtle brightness + saturation lift
                # on the subject. Applied before alpha so it never bleeds into
                # transparency. Saturation gets ~2x the lift of brightness since
                # Color is perceptually gentler.
                if enhance:
                    strength = max(0.0, min(float(enhance_strength), 0.5))
                    clean_rgb = ImageEnhance.Brightness(clean_rgb).enhance(1.0 + strength)
                    clean_rgb = ImageEnhance.Color(clean_rgb).enhance(1.0 + strength * 2)

                cutout = clean_rgb.convert("RGBA")
                cutout.putalpha(mask)

                subject_w = right - left
                subject_h = bottom - top
                pad = max(0, min(int(padding), 2000))  # cap padding — prevents oversized canvas

                # Target canvas: subject + 2*padding, padded out to aspect ratio
                base_w = subject_w + pad * 2
                base_h = subject_h + pad * 2
                target_w = max(base_w, int(round(base_h * aw / ah)))
                target_h = max(base_h, int(round(target_w * ah / aw)))
                # Re-check W after H adjustment (keeps ratio exact)
                if round(target_w * ah / aw) != target_h:
                    target_w = int(round(target_h * aw / ah))

                # Backstop: cap final canvas dimensions — a skinny subject + wide
                # aspect can still multiply out past the ratio/padding clamps.
                if target_w > 8000 or target_h > 8000:
                    raise HTTPException(400, "Resulting canvas too large; reduce padding or use a less extreme aspect ratio")

                bg_rgb = Image.new("RGB", (target_w, target_h), self._parse_color(bg_color))

                subject_cut = cutout.crop((left, top, right, bottom))
                subject_mask = mask.crop((left, top, right, bottom))

                paste_x = (target_w - subject_w) // 2
                paste_y = (target_h - subject_h) // 2

                if transparent:
                    # Transparent preset — centered cutout on a fully transparent
                    # canvas. No bg fill, no shadow (a shadow needs an opaque bg).
                    # Unmasked paste: using subject_cut as its own paste mask
                    # multiplies alpha into itself (128 -> 64) — Codex 009 P1.
                    composed = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
                    composed.paste(subject_cut, (paste_x, paste_y))
                elif shadow:
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

                resp = self._finalize_response(
                    composed, fmt, quality=quality, max_dim=max_dim, width=width,
                    height=height, watermark=watermark, watermark_opacity=watermark_opacity,
                    headers=self._engine_headers(used),
                )
                return resp
            except HTTPException as e:
                status_code = e.status_code
                raise
            except Exception:
                status_code = 500
                raise
            finally:
                self._end(ctx, "/studio-shot", _t, status_code)

        @web.post("/collage")
        def collage_endpoint(
            files: List[UploadFile] = File(...),
            main_index: int = Form(0),
            main_position: str = Form("BR"),
            bg_color: str = Form("#FFFFFF"),
            aspect: str = Form("1:1"),
            padding: int = Form(24),
            format: str = Form("jpg"),
            quality: Optional[int] = Form(None),
            max_dim: Optional[int] = Form(None),
            despill: Optional[float] = Form(None),
            watermark: Optional[str] = Form(None),
            watermark_opacity: float = Form(0.5),
            preset: Optional[str] = Form(None),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Product collage — N photos (2-9), each background-removed and
            tight-cropped, laid out around a main image on a solid canvas.

            `files`: 2-9 images. `main_index` picks the hero (default 0 = first).
            `main_position`: TL, T, TR, L, C, R, BL, B, BR (default BR). The main
            image takes ~65% of the canvas at that anchor; the rest fill the
            remaining space in equal cells.

            Paid tiers only. Billed at N base-image units (each photo is a full
            model pass).
            """
            ctx, _t = self._begin(authorization, "/collage")
            status_code = 200
            n = 1
            try:
                if despill is not None or watermark or preset:
                    self._require_pro(ctx, "Premium output (despill, watermark, presets)")
                if preset:
                    p = self._apply_preset(ctx, preset, {
                        "quality": quality, "max_dim": max_dim,
                        "despill": despill, "watermark": watermark,
                    })
                    quality, max_dim, despill, watermark = (
                        p["quality"], p["max_dim"], p["despill"], p["watermark"],
                    )
                n = len(files)
                if not (2 <= n <= 9):
                    raise HTTPException(400, "collage requires 2-9 images")
                if not (0 <= main_index < n):
                    raise HTTPException(400, f"main_index must be 0..{n - 1}")
                pos = main_position.strip().upper()
                if pos not in self._COLLAGE_POSITIONS:
                    raise HTTPException(
                        400, "main_position must be one of TL, T, TR, L, C, R, BL, B, BR")
                fmt = self._check_format(format, allowed=frozenset({"png", "webp", "jpg"}))

                try:
                    aw_str, ah_str = aspect.split(":")
                    aw, ah = int(aw_str), int(ah_str)
                    if aw <= 0 or ah <= 0:
                        raise ValueError()
                except Exception:
                    raise HTTPException(400, "aspect must be in 'W:H' format, e.g. '1:1' or '4:5'")
                if not (0.2 <= aw / ah <= 5.0):
                    raise HTTPException(400, "aspect ratio must be between 1:5 and 5:1")

                # Canvas: 1600px on the long side at the requested aspect.
                if aw >= ah:
                    W, H = 1600, max(1, int(round(1600 * ah / aw)))
                else:
                    H, W = 1600, max(1, int(round(1600 * aw / ah)))
                pad = max(0, min(int(padding), 200))

                cutouts = []
                for i, f in enumerate(files):
                    img = self._open_image(f.file.read())
                    rgb, mask = self._get_mask(img)
                    bbox = self._bounding_box(mask)
                    if bbox is None:
                        raise HTTPException(400, f"No subject detected in image {i + 1} of {n}")
                    clean = self._clean_foreground(
                        rgb, mask, strength=self._despill_strength(despill))
                    cut = clean.convert("RGBA")
                    cut.putalpha(mask)
                    cutouts.append(cut.crop(bbox))

                main_cut = cutouts.pop(main_index)
                main_rect, cells = self._collage_rects(W, H, len(cutouts), pos)
                canvas = Image.new("RGBA", (W, H), self._parse_color(bg_color) + (255,))
                fitted, at = self._fit_in_cell(main_cut, main_rect, pad)
                canvas.paste(fitted, at, fitted)
                for cut, cell in zip(cutouts, cells):
                    fitted, at = self._fit_in_cell(cut, cell, pad)
                    canvas.paste(fitted, at, fitted)

                return self._finalize_response(
                    canvas.convert("RGB"), fmt, quality=quality, max_dim=max_dim,
                    watermark=watermark, watermark_opacity=watermark_opacity,
                )
            except HTTPException as e:
                status_code = e.status_code
                raise
            except Exception:
                status_code = 500
                raise
            finally:
                # Billed at N units — each input is a full model pass.
                self._end(ctx, "/collage", _t, status_code, units=n)

        @web.post("/video/remove")
        def video_remove_endpoint(
            file: UploadFile = File(...),
            format: str = Form("prores4444"),
            bg_color: Optional[str] = Form(None),
            bg_image: Optional[UploadFile] = File(None),
            bg_blur: int = Form(0),
            speed: float = Form(1.0),
            smoothing: int = Form(30),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Video background removal — async. Returns a job_id immediately;
            poll GET /jobs/{job_id} for progress and the result URL.

            - `format`: prores4444 (MOV, real 10-bit alpha) | webm (VP9 alpha)
              | mp4 (H.264 — needs an opaque background, no alpha).
            - Backgrounds (precedence: image > blur > color > transparent):
              `bg_image` composite onto an uploaded image; `bg_blur` 1-100
              composite onto a blurred copy of the source (portrait look);
              `bg_color` solid hex.
            - `speed`: 0.25-4.0 output speed multiplier. Audio is tempo-shifted
              to match. Billing is on OUTPUT seconds, so 2x speed halves cost.
            - `smoothing`: 0-100 temporal alpha smoothing (kills matte flicker;
              values above 95 are clamped — lower it for fast-moving subjects).

            Caps: 15s max, 30fps, 200MB upload, 1080p processing. Paid tiers
            only. Billed at $0.10 per output second ($0.08 on Knockout Plus).
            """
            ctx, _t = self._begin(authorization, "/video/remove")
            fmt = format.strip().lower()
            if fmt not in VIDEO_FORMATS:
                raise HTTPException(400, "format must be prores4444, webm, or mp4")
            bg_blur = max(0, min(int(bg_blur), 100))
            if not (0.25 <= float(speed) <= 4.0):
                raise HTTPException(400, "speed must be between 0.25 and 4.0")
            has_bg = bool(bg_color or bg_image or bg_blur > 0)
            if fmt == "mp4" and not has_bg:
                raise HTTPException(400, "mp4 cannot carry alpha — set bg_color/bg_image/bg_blur or use prores4444/webm")
            if bg_color:
                self._parse_color(bg_color)  # validate early
            bg_png = None
            if bg_image is not None:
                # Validate + normalize the backdrop to PNG once, at submit.
                bg_obj = self._open_image(bg_image.file.read())
                buf = io.BytesIO()
                bg_obj.convert("RGB").save(buf, format="PNG")
                bg_png = buf.getvalue()
            ext = (file.filename or "").rsplit(".", 1)[-1].lower()
            if ext not in VIDEO_INPUT_EXTS:
                raise HTTPException(400, f"unsupported container .{ext} — use {', '.join(sorted(VIDEO_INPUT_EXTS))}")
            data = file.file.read()
            if len(data) > VIDEO_MAX_BYTES:
                raise HTTPException(413, "video too large (200MB max)")

            # Probe duration server-side before accepting the job.
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
                f.write(data)
                probe_path = f.name
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "json", probe_path],
                    capture_output=True, text=True, timeout=60)
                try:
                    seconds = float(json.loads(probe.stdout)["format"]["duration"])
                except Exception:
                    raise HTTPException(400, "could not read video duration — is the file valid?")
            finally:
                os.unlink(probe_path)
            if seconds > VIDEO_MAX_SECONDS:
                raise HTTPException(400, f"clip is {seconds:.1f}s — {VIDEO_MAX_SECONDS}s max for now")
            # Slow motion LENGTHENS the output — cap the OUTPUT length too, or a
            # 15s clip at 0.25x becomes a 60s render (4x the bill and a ProRes
            # big enough to blow the storage limit).
            if seconds / float(speed) > VIDEO_MAX_SECONDS:
                raise HTTPException(
                    400,
                    f"output would be {seconds / float(speed):.1f}s at speed={speed} — "
                    f"max {VIDEO_MAX_SECONDS}s output; use a shorter clip or higher speed")

            job_id = str(uuid.uuid4())
            self._storage_upload(f"{job_id}/in.{ext}", data, file.content_type or "video/mp4")
            if bg_png is not None:
                self._storage_upload(f"{job_id}/bg.png", bg_png, "image/png")
            status, _ = self._supabase_request(
                "POST", "/rest/v1/video_jobs",
                body={
                    "id": job_id,
                    "user_id": ctx.get("user_id"),
                    "token_id": ctx.get("token_id"),
                    "status": "queued",
                    "seconds": round(seconds, 2),
                    "params": {"format": fmt, "bg_color": bg_color,
                               "bg_blur": bg_blur, "speed": float(speed),
                               "has_bg_image": bg_png is not None,
                               "smoothing": smoothing, "in_ext": ext},
                },
                prefer="return=minimal")
            if status not in (200, 201):
                raise HTTPException(503, "could not create job")
            Knockout().process_video_job.spawn(job_id)

            # Billing is on OUTPUT seconds — speed shortens the output and the bill.
            est_seconds = max(1, math.ceil(seconds / float(speed)))
            # Usage row for the submit; billing happens in the worker on success.
            self._end(ctx, "/video/remove", _t, skip_meter=True)
            return {
                "job_id": job_id,
                "status": "queued",
                "seconds": round(seconds, 2),
                "output_seconds": round(seconds / float(speed), 2),
                "estimated_cost_usd": round(est_seconds * 0.10, 2),
                "poll": f"/jobs/{job_id}",
            }

        @web.get("/jobs/{job_id}")
        def job_status_endpoint(
            job_id: str,
            authorization: Optional[str] = Header(default=None),
        ):
            """Poll a video job. Returns status/progress; result_url when done (1h signed)."""
            ctx = self._check_auth(authorization)
            job = self._job_get(job_id)
            if not job:
                raise HTTPException(404, "job not found")
            if not ctx.get("is_legacy") and job.get("user_id") != ctx.get("user_id"):
                raise HTTPException(404, "job not found")
            resp = {
                "job_id": job_id,
                "status": job.get("status"),
                "progress": job.get("progress", 0),
                "seconds": job.get("seconds"),
                "frames": job.get("frames"),
            }
            if job.get("status") == "done" and job.get("result_path"):
                resp["result_url"] = self._storage_signed_url(job["result_path"])
            if job.get("status") == "error":
                resp["error"] = job.get("error")
            return resp

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
            ctx, _t = self._begin(authorization, "/compare")
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

            if ctx.get("is_demo"):
                canvas = self._downscale_max(canvas, DEMO_MAX_DIM)
            resp = self._response(canvas, fmt)
            self._end(ctx, "/compare", _t)
            return resp

        @web.post("/remove-batch-url")
        def remove_batch_url_endpoint(
            body: BatchUrlBody,
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Remove backgrounds from up to 10 remote images in one call.

            Body: {"urls": ["https://...", ...], "format": "png" | "webp"}
            """
            ctx, _t = self._begin(authorization, "/remove-batch-url")
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

            out_resp = {"count": len(results), "format": fmt, "results": results}
            self._end(ctx, "/remove-batch-url", _t)
            return out_resp

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
            model: str = Form("realesrgan"),
            face_enhance: bool = Form(False),
            format: str = Form("png"),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            Super-resolution. Two backends:

            - `model=realesrgan` (default, v0.9.0+): Real-ESRGAN x4plus. Restores /
              invents plausible detail — best "wow" on low-res or degraded photos,
              and faster (single pass, no tiling). Slight painted look on already-
              high-quality inputs.
            - `model=swin2sr`: SwinV2 transformer. Faithful, no invented detail —
              sharpens what's there without artifacts. Prefer for accuracy-sensitive
              (product / archival) work. Conservative on heavily degraded inputs.

            `scale` 2 or 4. `face_enhance=true` routes through GFPGAN (Real-ESRGAN
            backend only — kept for backwards compatibility).
            """
            ctx, _t = self._begin(authorization, "/upscale")
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
            resp = self._response(output_pil, fmt)
            self._end(ctx, "/upscale", _t)
            return resp

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
            ctx, _t = self._begin(authorization, "/face-restore")
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
            resp = self._response(self._bgr_to_pil(output_bgr), fmt)
            self._end(ctx, "/face-restore", _t)
            return resp

        @web.post("/colorize")
        def colorize_endpoint(
            file: UploadFile = File(...),
            format: str = Form("png"),
            authorization: Optional[str] = Header(default=None),
        ):
            """
            DDColor — colorize black-and-white or grayscale photos.

            Apache-2.0 licensed. ConvNeXt-Large backbone predicts ab channels
            in LAB color space. Single feed-forward (no diffusion sampling),
            ~500ms warm on L4. Works on any RGB input — color images are
            processed too (treated as grayscale internally).
            """
            ctx, _t = self._begin(authorization, "/colorize")
            fmt = self._check_format(format, allowed=frozenset({"png", "webp", "jpg"}))

            data = file.file.read()
            image_obj = self._open_image(data)

            # DDColor expects BGR uint8 (OpenCV convention). Same flow as Real-ESRGAN.
            bgr = self._pil_to_bgr(image_obj)
            try:
                result = self.colorizer(bgr)
                output_bgr = result.get(OutputKeys.OUTPUT_IMG)
            except Exception as e:
                raise HTTPException(500, f"colorize failed: {e}")

            if output_bgr is None:
                raise HTTPException(500, "colorize produced no output")

            self._bump_counter()
            resp = self._response(self._bgr_to_pil(output_bgr), fmt)
            self._end(ctx, "/colorize", _t)
            return resp

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
            ctx, _t = self._begin(authorization, "/preview")
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
            resp = self._response(result, fmt)
            self._end(ctx, "/preview", _t)
            return resp

        @web.post("/estimate")
        def estimate_endpoint(body: EstimateBody):
            """
            Predict latency + cost for a given endpoint and image size.

            No GPU work — pure lookup against measured baselines. Intended
            for client-side progress UI and pre-flight billing checks.
            """
            LATENCY_MS_BASE = {
                "remove": 200, "remove-url": 250, "replace-bg": 220,
                "mask": 150, "smart-crop": 180, "shadow": 230, "psd": 260,
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
                "est_cost_usd": 0.02,
                "free_during_beta": True,
                "note": "warm = container already running; cold = first request after scaledown",
            }

        @web.post("/headshot")
        def headshot_endpoint(
            file: UploadFile = File(...),
            bg_color: str = Form("#FFFFFF"),
            bg_blur: str = Form("false"),
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
            copy of the original. `bg_blur` accepts true/false OR an intensity
            2-100 (same scale as /video/remove; numeric overrides blur_radius).
            `1` stays a legacy alias for true (uses blur_radius, default 20),
            so the intensity scale starts at 2 — use 2 for the faintest blur.
            `head_top_ratio` controls how much empty space sits above the
            subject (default 18% of canvas).
            """
            ctx, _t = self._begin(authorization, "/headshot")
            # bg_blur: bool-like ("true"/"false") for back-compat, or 0-100
            # intensity matching the /video/remove scale.
            # Truthy/falsy sets must cover everything pydantic's bool coercion
            # accepted before this param became a string, or old callers 400.
            _blur_raw = (bg_blur or "false").strip().lower()
            if _blur_raw in ("true", "t", "yes", "y", "on", "1"):
                blur_on, blur_r = True, max(1, min(int(blur_radius), 80))
            elif _blur_raw in ("false", "f", "no", "n", "off", "", "0", "none"):
                blur_on, blur_r = False, 0
            else:
                try:
                    _n = int(_blur_raw)
                except ValueError:
                    raise HTTPException(400, "bg_blur must be true/false or an intensity 0-100")
                if not 0 <= _n <= 100:
                    raise HTTPException(400, "bg_blur intensity must be 0-100")
                blur_on = _n > 0
                blur_r = int(round(2 + (_n / 100.0) * 38))  # same mapping as /video/remove
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

            if blur_on:
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
            resp = self._response(composed, fmt)
            self._end(ctx, "/headshot", _t)
            return resp

        return web
