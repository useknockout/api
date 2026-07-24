
 <div align="center">

  # 🥊 useknockout
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/tlorents/useknockout-demo)

  **State-of-the-art background removal API for images and video — open source, self-hostable, 10× cheaper than remove.bg.**

  [![MIT License](https://img.shields.io/badge/license-MIT-3da639)](./LICENSE)
  [![npm version](https://img.shields.io/npm/v/@useknockout/node?color=cb3837)](https://www.npmjs.com/package/@useknockout/node)
  [![npm downloads](https://img.shields.io/npm/dm/@useknockout/node?color=cb3837)](https://www.npmjs.com/package/@useknockout/node)
  [![GitHub stars](https://img.shields.io/github/stars/useknockout/api?style=social)](https://github.com/useknockout/api)
  [![Powered by Modal](https://img.shields.io/badge/powered%20by-Modal-7c3aed)](https://modal.com)
  [![Model: BiRefNet](https://img.shields.io/badge/model-BiRefNet-ff6f00)](https://github.com/ZhengPeng7/BiRefNet)
  [![Python](https://img.shields.io/badge/python-3.11-3776ab?logo=python&logoColor=white)](https://python.org)
  [![TypeScript](https://img.shields.io/badge/SDK-TypeScript-3178c6?logo=typescript&logoColor=white)](https://www.npmjs.com/package/@useknockout/node)

  [**Playground**](https://useknockout.com/playground) · [**Docs**](https://useknockout.com/docs) · [**Quick Start**](#quick-start) · [**API Reference**](./APIREFERENCE.md) ·
  [**Self-hosting**](./SELFHOSTING.md)

  <br/>

  <img src="./docs/hero.png" alt="useknockout before/after — background removal demo" width="800"/>

  <br/>

  *Drop an image in. Get a transparent PNG out. ~200ms per call.*

</div>

A production-grade background removal API powered by [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — the current SOTA on DIS5K, HRSOD, and COD benchmarks. Served on Modal's GPU infrastructure with scale-to-zero economics.

- **SOTA quality** — matches or beats remove.bg, Photoroom, and Pixelcut on hair, fur, fine detail
- **Fast** — ~200ms per image on a warm L4 GPU
- **Cheap** — $0.02 per image pay-as-you-go (10x cheaper than remove.bg PAYG)
- **Video too** — async video background removal with ProRes 4444 alpha output ($0.10/output-second)
- **26 endpoints** — cutouts, masks, PSD export, upscale, face restore, inpaint, collage, studio shots, and more
- **MIT licensed** — model weights and code, commercial use OK
- **Self-hostable** — deploy to your own Modal workspace in one command

  
*Works alpha-preserving (PNG with transparent bg) OR opaque (solid color / remote image as new bg).*
   

---

## Table of contents

- [Demo](#demo)
- [Quick start](#quick-start)
- [API reference](./APIREFERENCE.md)
- [Benchmarks](#benchmarks)
- [Self-hosting](./SELFHOSTING.md)
- [Architecture](#architecture)
- [Pricing](#pricing)
- [License](#license)

---

## Demo

**Try it in your browser, no signup:** [useknockout.com/playground](https://useknockout.com/playground)

Drag an image in, pick an operation, get the result back. It runs the live API. Also available as a [Hugging Face Space](https://huggingface.co/spaces/tlorents/useknockout-demo).

**Full docs (endpoints, SDKs, self-hosting):** [useknockout.com/docs](https://useknockout.com/docs)

**Base URL (for code):** `https://useknockout--api.modal.run`

Input → Output:

| Original | After |
|---|---|
| Complex hair | Clean wisps, no halo |
| Fur / pet photos | Soft edges preserved |
| Product shots | Sharp, clean cutout |
| Low-contrast subjects | Accurate separation |

---

## Quick start

### Get your key — 10 images/month free, no card

→ **[useknockout.com/signin](https://useknockout.com/signin)**

Sign in, click to reveal your key on the dashboard, and paste it below. No credit card, ready in seconds.

- **10 full-resolution images/month free, forever** — no card needed
- Pay-as-you-go at **$0.02/image** unlocks all endpoints — AI upscale, face restore, colorize, e-commerce presets, batch, video, and more (10× cheaper than remove.bg's $0.20)
- **$0.003/image** at 100k+/month for volume

```bash
export KNOCKOUT_TOKEN=kno_live_your_key_here
```

The examples below assume `$KNOCKOUT_TOKEN` is set.

### Hit the API in 3 seconds

```bash
curl -X POST "https://useknockout--api.modal.run/remove" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "file=@your-image.jpg" \
  -o out.png
```

You get a PNG with a transparent alpha channel. Done.

### With a URL instead of a file

```bash
curl -X POST "https://useknockout--api.modal.run/remove-url" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/cat.jpg"}' \
  -o out.png
```

### Replace the background with a color or remote image

```bash
# solid color background
curl -X POST "https://useknockout--api.modal.run/replace-bg" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "file=@cat.jpg" \
  -F "bg_color=#FF5733" \
  -F "format=jpg" \
  -o out.jpg

# use a remote image as the new background
curl -X POST "https://useknockout--api.modal.run/replace-bg" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "file=@cat.jpg" \
  -F "bg_url=https://example.com/mountains.jpg" \
  -o out.png
```

### Batch — process up to 10 images in one call

```bash
# multipart batch
curl -X POST "https://useknockout--api.modal.run/remove-batch?format=png" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "files=@a.jpg" -F "files=@b.jpg" -F "files=@c.jpg"

# URL batch — JSON body
curl -X POST "https://useknockout--api.modal.run/remove-batch-url" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"urls":["https://a.jpg","https://b.jpg"], "format":"png"}'
```

Both return JSON: `{ "count": N, "format": "png", "results": [{ "success": true, "data_base64": "..." }, ...] }`.

### More presets (v0.3.0)

```bash
# Sticker — cutout + thick white outline (WhatsApp / iMessage sticker style)
curl -X POST "https://useknockout--api.modal.run/sticker" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "file=@photo.jpg" -F "stroke_width=24" -o sticker.png

# Smart crop — tight bounding box around subject
curl -X POST "https://useknockout--api.modal.run/smart-crop" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "file=@photo.jpg" -F "padding=32" -o cropped.png

# Studio shot — e-commerce preset (white bg + shadow + centered, 1:1 aspect)
curl -X POST "https://useknockout--api.modal.run/studio-shot" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "file=@photo.jpg" -F "aspect=1:1" -F "format=jpg" -o studio.jpg

# Shadow — subject composited onto new bg with a drop shadow
curl -X POST "https://useknockout--api.modal.run/shadow" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "file=@photo.jpg" -F "bg_color=#F3F4F6" -o shadow.png

# Compare — before/after side-by-side for marketing/social
curl -X POST "https://useknockout--api.modal.run/compare" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "file=@photo.jpg" -o compare.png

# Mask — just the black/white mask, for your own pipeline
curl -X POST "https://useknockout--api.modal.run/mask" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "file=@photo.jpg" -o mask.png

# Outline — subject on transparent bg with a thin outline
curl -X POST "https://useknockout--api.modal.run/outline" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "file=@photo.jpg" -F "outline_color=#000000" -F "outline_width=4" -o outline.png
```

### New in v0.9–v0.11 — upscale, PSD, collage, video

```bash
# AI upscale — Real-ESRGAN super-resolution (2x/4x)
curl -X POST "https://useknockout--api.modal.run/upscale" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "file=@photo.jpg" -F "scale=4" -o upscaled.png

# PSD export — layered Photoshop file (subject + background layers)
curl -X POST "https://useknockout--api.modal.run/psd" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "file=@photo.jpg" -o out.psd

# Collage — 2-9 photos, each cut out and laid out around a hero image
curl -X POST "https://useknockout--api.modal.run/collage" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "files=@main.jpg" -F "files=@b.jpg" -F "files=@c.jpg" \
  -F "main_position=BR" -F "aspect=1:1" -o collage.jpg
```

### Video background removal (async)

Removes the background from a clip (up to 15s) and returns ProRes 4444 with a real alpha channel — or WebM/MP4 composited onto a color, image, or blur.

```bash
# submit the job
curl -X POST "https://useknockout--api.modal.run/video/remove" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "file=@clip.mp4" -F "format=prores4444"
# -> {"job_id": "..."}

# poll until done, then download
curl "https://useknockout--api.modal.run/jobs/JOB_ID" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN"
```

Billed at $0.10 per output second ($0.08 on Knockout Plus). Paid tiers only.

### Health check

```bash
curl https://useknockout--api.modal.run/health
# {"status":"ok","model":"ZhengPeng7/BiRefNet"}
```

---

## API reference

The full endpoint-by-endpoint reference for all 26 endpoints, plus client examples for Python, Node.js, Go, and browser/TypeScript, lives in a dedicated doc:

→ **[API reference & client examples](./APIREFERENCE.md)**

---

## Benchmarks

Measured on Modal `gpu="L4"`, Python 3.11, torch 2.4, batch size 1, 1024×1024 model input.

| Image size | Warm latency (p50) | Cold start | Output format |
|---|---|---|---|
| 512×512 | 180 ms | ~25 s | PNG / WebP |
| 1024×1024 | 220 ms | ~25 s | PNG / WebP |
| 2048×2048 | 310 ms | ~25 s | PNG / WebP |
| 4000×4000 | 520 ms | ~25 s | PNG / WebP |

### Quality vs. competitors

BiRefNet (the model we serve) consistently ranks first or second on public benchmarks:

- **DIS5K** (Dichotomous Image Segmentation): #1 F-measure as of 2024
- **HRSOD** (High-Resolution Salient Object Detection): #1 MAE
- **COD10K** (Camouflaged Object Detection): #1 or #2 depending on metric

See the [BiRefNet paper](https://arxiv.org/abs/2401.03407) and [leaderboards](https://paperswithcode.com/task/dichotomous-image-segmentation) for details.

---

## Self-hosting

Run your own instance on Modal in one command. Full prerequisites, deploy steps, and tuning guide:

→ **[Self-hosting guide](./SELFHOSTING.md)**

---

## Architecture

```
┌────────────┐      HTTPS       ┌───────────────────────────┐
│   Client   │ ───────────────▶ │  Modal ASGI (FastAPI)     │
│ (any lang) │                  │  ┌─────────────────────┐  │
└────────────┘                  │  │ Auth (bearer)       │  │
                                │  │ Validation          │  │
                                │  │ Image decode (PIL)  │  │
                                │  │ BiRefNet on L4 GPU  │  │
                                │  │ Encode (PNG/WebP)   │  │
                                │  └─────────────────────┘  │
                                │  Scale-to-zero, auto-HTTPS │
                                └───────────────────────────┘
```

- **One file** (`main.py`), single Modal class, 26 endpoints + auto-generated docs
- **Weights baked into image** at build time — cold starts are just image pull + GPU model load (~25 s)
- **FastAPI** handles multipart, JSON, CORS, OpenAPI schema generation

---

## Pricing

Sign up at **[useknockout.com/signin](https://useknockout.com/signin)** — 10 images/month free, no card.

| Tier | Price | Best for |
|---|---|---|
| **Free** | 10 images / month, no card | Personal, eval, open source |
| **Pay-as-you-go** | $0.02 / image | Side projects, early startups |
| **Knockout Plus** | $10 / month, 250 images included, then $0.02 | Regular use + premium features |
| **Volume** | $0.003 / image at 100k+/mo | Production workloads |
| **Enterprise** | Custom, private endpoints | Compliance, BYO-cloud |

Add-ons: **PSD export** $0.10/image · **Video** $0.10 per output second ($0.08 on Plus).

For reference — the same image on remove.bg is **$0.20** at their PAYG rate.

**Free tier includes** the core background-removal endpoints:

`/remove` · `/remove-url` · `/mask` · `/compare` · `/preview`

**Paid endpoints** (any paid tier) add edits, AI enhancement, e-commerce presets, batch, and video:

`/replace-bg` · `/smart-crop` · `/outline` · `/sticker` · `/shadow` · `/silhouette` · `/studio-shot` · `/headshot` · `/collage` · `/upscale` · `/face-restore` · `/colorize` · `/inpaint` · `/psd` · `/remove-batch` · `/remove-batch-url` · `/video/remove`

**Knockout Plus** additionally unlocks layered PSD at a flat rate, edge despill, saved presets (`/presets`), custom watermarks, resizing, and compression control across all endpoints.

Try the API in your browser with no key at the [playground](https://useknockout.com/playground). `/estimate` (pricing calculator) is free for everyone.

---

## Contact

- **GitHub Issues:** https://github.com/useknockout/api/issues
- **Twitter / X:** [@useknockout](https://x.com/useknockout)

---

## License

MIT License — see [LICENSE](./LICENSE). Model weights ([BiRefNet](https://github.com/ZhengPeng7/BiRefNet)) are also MIT. Commercial use is allowed for both.

---

<p align="center">
  Built in a few hours because someone said it couldn't be done.
</p>
