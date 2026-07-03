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
import os
import secrets
import time
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
DEMO_MAX_DIM = 512                 # demo output capped to this longest side
DEMO_DAILY_CAP_DEFAULT = 500       # global anonymous calls/day; DEMO_DAILY_CAP overrides

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
                    "useknockout.com/signin — 10 images/month free, no card, "
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

        row = rows[0]
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
                        "own key — 10 full-quality images/month free across the "
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
                        "own key — 10 images/month free, no card, available now."
                    ),
                )
            d[key] = used + 1
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
        """Free tier: 10 images/month. Paid tiers: no monthly cap."""
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
                    if used >= 10:
                        raise HTTPException(
                            status_code=402,
                            detail="Free tier monthly quota (10) exhausted. Upgrade at useknockout.com/pricing.",
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
                },
                prefer="return=minimal",
            )
        except Exception:
            pass

        # 2. Stripe meter event for paid tiers (pro pays base per-image too —
        #    the $10/mo unlocks features, image volume is still metered).
        if 200 <= status < 300 and not skip_meter and ctx.get("tier") in {"payg", "volume", "pro"}:
            self._report_meter(ctx, units=units, event_name=meter_event)

    def _begin(self, authorization: Optional[str], endpoint: str) -> Tuple[dict, float]:
        """One call → auth + scope + quota + start timer. Use at top of each handler."""
        ctx = self._check_auth(authorization)
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

    def _get_mask(self, image_obj, refine: bool = True):
        """Run BiRefNet on an RGB image, return (rgb_image, mask_pil).

        Pads the image to a square BEFORE the model's fixed 1024x1024 resize so
        the aspect ratio is preserved. Resizing a non-square image straight to
        1024x1024 squishes it, which wrecks thin, low-contrast boundaries (water
        reflections, fine hair, etc). We pad to a square, infer, then crop the
        mask back to the original frame.

        refine=False skips the guided-filter edge refinement — for callers that
        only need a region (e.g. /inpaint), where the full-res float32 filter
        buffers are pure wasted memory + latency.
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

    def _remove(self, image_obj, despill=None):
        rgb, mask = self._get_mask(image_obj)
        clean_rgb = self._clean_foreground(rgb, mask, strength=self._despill_strength(despill))
        result = clean_rgb.convert("RGBA")
        result.putalpha(mask)
        return result

    def _composite_on_bg(self, image_obj, bg_image_or_color, despill=None):
        """Composite foreground onto a solid color or image background."""
        rgb, mask = self._get_mask(image_obj)
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

    def _response(self, image_out, fmt: str, quality: Optional[int] = None):
        content = self._encode(image_out, fmt, quality=quality)
        return Response(content=content, media_type=self._FORMAT_TO_MEDIA[fmt])

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

    def _finalize_response(self, image_obj, fmt: str, **kw):
        content = self._finalize(image_obj, fmt, **kw)
        return Response(content=content, media_type=self._FORMAT_TO_MEDIA[fmt])

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

    @modal.asgi_app(label="api")
    def fastapi_app(self):
        web = FastAPI(
            title="useknockout",
            description="State-of-the-art background removal + upscaling + colorization API.",
            version="0.10.0",
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
                "version": "0.8.0",
                "endpoints": [
                    "POST /remove",
                    "POST /remove-url",
                    "POST /psd",
                    "POST /replace-bg",
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
            watermark: Optional[str] = Form(None),
            watermark_opacity: float = Form(0.5),
            preset: Optional[str] = Form(None),
            authorization: Optional[str] = Header(default=None),
        ):
            ctx, _t = self._begin(authorization, "/remove")
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
            data = file.file.read()
            image_obj = self._open_image(data)
            result = self._remove(image_obj, despill=despill)
            if ctx.get("is_demo"):
                result = self._downscale_max(result, DEMO_MAX_DIM)
                # Resize params could upscale right past the demo cap — drop them.
                max_dim = width = height = None
            resp = self._finalize_response(
                result, fmt, quality=quality, max_dim=max_dim, width=width,
                height=height, watermark=watermark, watermark_opacity=watermark_opacity,
            )
            self._end(ctx, "/remove", _t)
            return resp

        @web.post("/remove-url")
        def remove_url_endpoint(
            body: UrlBody,
            authorization: Optional[str] = Header(default=None),
        ):
            ctx, _t = self._begin(authorization, "/remove-url")
            fmt = self._check_format(body.format)

            try:
                resp = requests.get(str(body.url), timeout=15)
                resp.raise_for_status()
            except requests.RequestException as e:
                raise HTTPException(400, f"Could not fetch image: {e}")

            image_obj = self._open_image(resp.content)
            result = self._remove(image_obj)
            out_resp = self._response(result, fmt)
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
            authorization: Optional[str] = Header(default=None),
        ):
            """Return just the alpha mask as a grayscale PNG/WebP (0 = bg, 255 = subject)."""
            ctx, _t = self._begin(authorization, "/mask")
            fmt = self._check_format(format)
            data = file.file.read()
            image_obj = self._open_image(data)
            _, mask = self._get_mask(image_obj)
            out = mask.convert("L")
            if ctx.get("is_demo"):
                out = self._downscale_max(out, DEMO_MAX_DIM)
            resp = self._response(out, fmt)
            self._end(ctx, "/mask", _t)
            return resp

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
            ctx, _t = self._begin(authorization, "/smart-crop")
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

            resp = self._response(cropped, fmt)
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
                rgb, mask = self._get_mask(image_obj)

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
                    composed = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
                    composed.paste(subject_cut, (paste_x, paste_y), subject_cut)
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
            ctx, _t = self._begin(authorization, "/headshot")
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
            resp = self._response(composed, fmt)
            self._end(ctx, "/headshot", _t)
            return resp

        return web
