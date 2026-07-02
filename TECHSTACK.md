---
project: projects/useknockout-api
type: techstack
---

# useknockout-api — Tech Stack

A single-file Python service (`main.py`) that runs a stack of computer-vision models on GPU and exposes them as an HTTP API. There are no LLMs in this project — every model is an image model (segmentation, super-resolution, face restoration, colorization, inpainting).

## Language & Runtime

| Item | Version | Notes |
|---|---|---|
| Python | 3.11 | Pinned in the Modal image (`debian_slim(python_version="3.11")`). |
| Modal | `>=0.64.0` | Only dependency in `requirements.txt`. The deploy/orchestration layer — everything else is installed *inside* the Modal container image, not locally. |

The local `requirements.txt` deliberately contains only `modal` — all ML/image dependencies are declared in `main.py` via `modal.Image.pip_install(...)` and baked into the container at build time.

## Serverless / GPU Platform

| Tool | Purpose |
|---|---|
| **Modal** | Serverless GPU platform. Hosts the app as a Modal `App` with an `@app.cls` GPU class (`gpu="L4"`), exposes the FastAPI app via `@modal.asgi_app`, manages scale-to-zero, secrets, and a `modal.Dict` (`knockout-stats`) used as a lightweight cross-container counter / demo-rate-limit store. |

Modal config of note (in `main.py`): `gpu="L4"`, `scaledown_window=300` (5 min warm), `timeout=600`, `max_containers=10`, secret `knockout-secrets`.

## Web Framework

| Library | Version | Purpose |
|---|---|---|
| **FastAPI** | `0.115.0` (`fastapi[standard]`) | HTTP API framework. Serves all ~22 endpoints, auto-generates OpenAPI docs at `/docs`, handles multipart/JSON bodies. Mounted on Modal via `@modal.asgi_app`. |
| **python-multipart** | `0.0.9` | Multipart form parsing for file uploads (`UploadFile`). |
| **pydantic** | `2.9.2` | Request body models (`UrlBody`, `BatchUrlBody`, `EstimateBody`) and validation (`HttpUrl`). |
| **CORS middleware** | (FastAPI built-in) | `allow_origins=["*"]`, methods `POST`/`GET` — lets the browser playground call the API directly. |

## Core ML / Vision Models & Libraries

| Library / Model | Version | License | Purpose |
|---|---|---|---|
| **PyTorch (torch)** | `2.4.0` | BSD | Inference engine for all GPU models. Run in `.half()` (fp16) on CUDA. |
| **torchvision** | `0.19.0` | BSD | Image transforms (resize / normalize / ToTensor) feeding BiRefNet. |
| **transformers** | `4.44.2` | Apache-2.0 | Loads BiRefNet (`AutoModelForImageSegmentation`, `trust_remote_code=True`) and Swin2SR (`Swin2SRForImageSuperResolution` + `AutoImageProcessor`). |
| **BiRefNet** (`ZhengPeng7/BiRefNet`) | HF weights | MIT | **The core model** — SOTA dichotomous image segmentation / salient-object detection. Produces the alpha matte for background removal. Input size 1024×1024. Drives every `/remove`-family + preset endpoint. |
| **Swin2SR** (`caidas/swin2SR-*`) | HF weights | Apache-2.0 | Default super-resolution model for `/upscale` (x2 classical, x4 real-world BSRGAN-PSNR). SwinV2 transformer; better natural texture on real photos than Real-ESRGAN. Tiled inference with linear-blend overlap implemented by hand. |
| **Real-ESRGAN (realesrgan)** | `0.3.0` | BSD-3 | Alternative `/upscale` backend (`RRDBNet` x4plus weights). Better on anime/illustration. Also the optional bg upsampler for face restore. |
| **GFPGAN** | `1.3.8` | Apache-2.0 | Portrait/face restoration (`/face-restore`, and `/upscale?face_enhance=true`). Uses GFPGANv1.4 weights. Two configured restorers: bg-preserving and full (bg upscaled via Real-ESRGAN). |
| **facexlib** | `0.3.0` | (Apache-2.0) | Face detection + parsing models (ResNet50 detection, parsenet parsing) used by GFPGAN. Weights pre-baked into the image. |
| **basicsr** | `1.4.2` | Apache-2.0 | Backbone arch lib for Real-ESRGAN/GFPGAN (`RRDBNet`). Note: build patches its `torchvision.transforms.functional_tensor` import (removed in torchvision 0.17+) via sed. |
| **DDColor** (`damo/cv_ddcolor_image-colorization`) | ModelScope snapshot | Apache-2.0 | Photo colorization (`/colorize`). ConvNeXt-Large backbone predicting ab channels in LAB; single feed-forward, no diffusion. |
| **ModelScope (modelscope)** | `1.18.1` | Apache-2.0 | Pipeline registry used to load + run DDColor (keeps its basicsr fork isolated from the main basicsr). |
| **simple-lama-inpainting** | `0.1.2` | Apache-2.0 | Wrapper around **LaMa** (big-lama) for `/inpaint`. Resolution-robust large-mask inpainting, deterministic, no prompts. |
| **pymatting** | `1.1.12` | MIT | Closed-form / ML foreground estimation (`estimate_foreground_cf` / `_ml`) to remove color spill & halos at mask edges (`_clean_foreground`). |
| **Pillow (PIL)** | `10.4.0` | HPND/MIT | All image decode/encode + compositing, masks, shadows, outlines, EXIF-orientation handling (`ImageOps.exif_transpose`), checkerboard previews, drop shadows. |
| **OpenCV (opencv-python-headless)** | `4.10.0.84` | Apache-2.0 | Required by Real-ESRGAN / GFPGAN / DDColor pipelines (BGR array convention). |
| **NumPy** | `1.26.4` | BSD | Array math throughout (masks, bounding boxes, tiled upscale blending). Pinned to 1.26.4 — basicsr/modelscope try to bump it to 2.x. |
| **timm** | `1.0.9` | Apache-2.0 | Backbone building blocks required by BiRefNet. |
| **kornia** | `0.7.3` | Apache-2.0 | Differentiable CV ops (BiRefNet dependency). |
| **einops** | `0.8.0` | MIT | Tensor rearrange ops (model dependency). |
| **huggingface_hub** | `0.24.6` | Apache-2.0 | Pulls model weights from the HF Hub at build time. |
| **requests** | `2.32.3` | Apache-2.0 | Fetching remote images for `/remove-url`, `/replace-bg?bg_url`, batch-url. |

### Supporting deps pulled in for ModelScope
`datasets==2.21.0`, `oss2==2.18.5`, `addict==2.4.0`, `simplejson==3.19.2`, `sortedcontainers==2.4.0` — front-loaded in the image because ModelScope's pipeline base imports them unconditionally.

### System packages
`libgl1`, `libglib2.0-0` (apt) — required by OpenCV in the headless container.

## Build Tooling

| Tool | Purpose |
|---|---|
| `modal.Image` builder | Declarative image build: apt install, layered `pip_install`, a `run_commands` sed-patch for the `functional_tensor` rename, and `run_function(_download_model)` to **bake all model weights into the image** at build time (BiRefNet, Swin2SR x2/x4, Real-ESRGAN, GFPGAN, facexlib detection/parsing, DDColor ~870 MB, LaMa ~200 MB) so cold starts skip downloads. |
| `deploy.sh` | One-command deploy wrapper (`modal deploy main.py`). |
| `modal token new` / `modal secret create` | CLI auth + secret provisioning (see SELFHOSTING.md). |

## Standard-library usage of note
`urllib.request` is used directly (not an SDK) for all outbound HTTP to Supabase REST and the Stripe meter-events API — keeps the container dependency-light.

## Auxiliary scripts
- `scripts/migrate-payg-prices.mjs` — Node.js (ESM) script for migrating pay-as-you-go prices (Stripe-side billing maintenance).
- `eval/run_flowerbox.py` — pure-stdlib eval harness that posts test images through the live `/studio-shot` endpoint.

## External APIs called by the service
- **Supabase REST** (`/rest/v1/...`) — per-user token auth, tier lookup, usage logging, monthly-quota view.
- **Stripe Billing Meter Events API** (`api.stripe.com/v1/billing/meter_events`) — usage-based billing for paid tiers.
- **Hugging Face Hub** / **ModelScope** / **GitHub releases** — model-weight downloads (build time only).

## TypeScript SDK (separate package)
The README references `@useknockout/node` (npm) as the official TypeScript/Node client. It lives in a separate repo/package; this repository is the Python API server only.
