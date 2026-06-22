 <div align="center">

  # 🥊 useknockout
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/tlorents/useknockout-demo)

  **State-of-the-art background removal API — open source, self-hostable, 40× cheaper than remove.bg.**

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
- **Cheap** — (Currently FREE) Starting June 1 ~$0.02 per image raw compute cost (40x cheaper than remove.bg PAYG)
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

### Try it right now — no signup

The shared demo key works instantly, no account needed:

```bash
curl -X POST "https://useknockout--api.modal.run/remove" \
  -H "Authorization: Bearer kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f" \
  -F "file=@your-image.jpg" \
  -o out.png
```

The demo key is deliberately limited: **`/remove` only, low-res output, shared daily cap.** Enough to judge the quality — sign up below for full resolution and every other endpoint.

### Get your own key — 20 images/month free, no card

→ **[useknockout.com/signin](https://useknockout.com/signin)**

- **20 full-resolution images/month free, forever** — no card needed
- **All endpoints unlocked** — AI upscale, face restore, colorize, e-commerce presets, batch, and more
- Then **$0.02/image** pay-as-you-go (20× cheaper than remove.bg's $0.20)
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

### Health check

```bash
curl https://useknockout--api.modal.run/health
# {"status":"ok","model":"ZhengPeng7/BiRefNet"}
```

---

## API reference

The full endpoint-by-endpoint reference for all 20 endpoints, plus client examples for Python, Node.js, Go, and browser/TypeScript, lives in a dedicated doc:

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

- **One file** (`main.py`), single Modal class, two endpoints + health + docs
- **Weights baked into image** at build time — cold starts are just image pull + GPU model load (~25 s)
- **FastAPI** handles multipart, JSON, CORS, OpenAPI schema generation

---

## Pricing

Sign up at **[useknockout.com/signin](https://useknockout.com/signin)** — 20 images/month free, no card.

| Tier | Price | Best for |
|---|---|---|
| **Free** | 20 images / month, no card | Personal, eval, open source |
| **Pay-as-you-go** | $0.005 / image | Side projects, early startups |
| **Volume** | $0.003 / image at 100k+/mo | Production workloads |
| **Enterprise** | Custom, private endpoints | Compliance, BYO-cloud |

For reference — the same image on remove.bg is **$0.20** at their PAYG rate.

**Free tier includes** all core background-removal + basic-edit endpoints:

`/remove` · `/remove-url` · `/replace-bg` · `/mask` · `/smart-crop` · `/outline` · `/sticker` · `/compare` · `/preview`

**Paid endpoints** (any paid tier) add AI enhancement, e-commerce presets, and batch:

`/studio-shot` · `/headshot` · `/shadow` · `/silhouette` · `/upscale` · `/face-restore` · `/colorize` · `/inpaint` · `/remove-batch` · `/remove-batch-url`

The anonymous demo key (no signup) is `/remove` only, low-res, with a shared daily cap. `/estimate` (pricing calculator) is free for everyone.

Credits never expire. No subscriptions. You only pay for what you use.

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
