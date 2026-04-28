 <div align="center">

  # 🥊 useknockout

  **State-of-the-art background removal API — open source, self-hostable, 40× cheaper than remove.bg.**

  [![MIT License](https://img.shields.io/badge/license-MIT-3da639)](./LICENSE)
  [![npm version](https://img.shields.io/npm/v/@useknockout/node?color=cb3837)](https://www.npmjs.com/package/@useknockout/node)
  [![npm downloads](https://img.shields.io/npm/dm/@useknockout/node?color=cb3837)](https://www.npmjs.com/package/@useknockout/node)
  [![GitHub stars](https://img.shields.io/github/stars/useknockout/api?style=social)](https://github.com/useknockout/api)
  [![Powered by Modal](https://img.shields.io/badge/powered%20by-Modal-7c3aed)](https://modal.com)
  [![Model: BiRefNet](https://img.shields.io/badge/model-BiRefNet-ff6f00)](https://github.com/ZhengPeng7/BiRefNet)
  [![Python](https://img.shields.io/badge/python-3.11-3776ab?logo=python&logoColor=white)](https://python.org)
  [![TypeScript](https://img.shields.io/badge/SDK-TypeScript-3178c6?logo=typescript&logoColor=white)](https://www.npmjs.com/package/@useknockout/node)

  [**Live API**](https://useknockout--api.modal.run) · [**Docs**](https://useknockout--api.modal.run/docs) · [**Quick Start**](#quick-start) · [**API Reference**](#api-reference) ·
  [**Self-hosting**](#self-hosting)

  <br/>

  <img src="./docs/hero.png" alt="useknockout before/after — background removal demo" width="800"/>

  <br/>

  *Drop an image in. Get a transparent PNG out. ~200ms per call.*

</div>


A production-grade background removal API powered by [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — the current SOTA on DIS5K, HRSOD, and COD benchmarks. Served on Modal's GPU infrastructure with scale-to-zero economics.

- **SOTA quality** — matches or beats remove.bg, Photoroom, and Pixelcut on hair, fur, fine detail
- **Fast** — ~200ms per image on a warm L4 GPU
- **Cheap** — ~$0.00005 per image raw compute cost (4,000x cheaper than remove.bg PAYG)
- **MIT licensed** — model weights and code, commercial use OK
- **Self-hostable** — deploy to your own Modal workspace in one command

  
*Works alpha-preserving (PNG with transparent bg) OR opaque (solid color / remote image as new bg).*
   

---

## Table of contents

- [Demo](#demo)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Client examples](#client-examples)
- [Benchmarks](#benchmarks)
- [Self-hosting](#self-hosting)
- [Architecture](#architecture)
- [Pricing](#pricing)
- [License](#license)

---

## Demo

**Live endpoint:** `https://useknockout--api.modal.run`

**Interactive docs:** `https://useknockout--api.modal.run/docs`

Input → Output:

| Original | After |
|---|---|
| Complex hair | Clean wisps, no halo |
| Fur / pet photos | Soft edges preserved |
| Product shots | Sharp, clean cutout |
| Low-contrast subjects | Accurate separation |

---

## Quick start

### Public beta token — copy, paste, try it right now

During public beta, everyone shares this bearer token:

```
kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f
```

No signup. Just use it. We're free during beta. Paid tier launches soon — need your own key or higher limits? DM [@useknockout](https://x.com/useknockout).

### Hit the API in 3 seconds

```bash
curl -X POST "https://useknockout--api.modal.run/remove" \
  -H "Authorization: Bearer kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f" \
  -F "file=@your-image.jpg" \
  -o out.png
```

You get a PNG with a transparent alpha channel. Done.

### With a URL instead of a file

```bash
curl -X POST "https://useknockout--api.modal.run/remove-url" \
  -H "Authorization: Bearer kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/cat.jpg"}' \
  -o out.png
```

### Replace the background with a color or remote image

```bash
# solid color background
curl -X POST "https://useknockout--api.modal.run/replace-bg" \
  -H "Authorization: Bearer kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f" \
  -F "file=@cat.jpg" \
  -F "bg_color=#FF5733" \
  -F "format=jpg" \
  -o out.jpg

# use a remote image as the new background
curl -X POST "https://useknockout--api.modal.run/replace-bg" \
  -H "Authorization: Bearer kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f" \
  -F "file=@cat.jpg" \
  -F "bg_url=https://example.com/mountains.jpg" \
  -o out.png
```

### Batch — process up to 10 images in one call

```bash
# multipart batch
curl -X POST "https://useknockout--api.modal.run/remove-batch?format=png" \
  -H "Authorization: Bearer kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f" \
  -F "files=@a.jpg" -F "files=@b.jpg" -F "files=@c.jpg"

# URL batch — JSON body
curl -X POST "https://useknockout--api.modal.run/remove-batch-url" \
  -H "Authorization: Bearer kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f" \
  -H "Content-Type: application/json" \
  -d '{"urls":["https://a.jpg","https://b.jpg"], "format":"png"}'
```

Both return JSON: `{ "count": N, "format": "png", "results": [{ "success": true, "data_base64": "..." }, ...] }`.

### More presets (v0.3.0)

```bash
# Sticker — cutout + thick white outline (WhatsApp / iMessage sticker style)
curl -X POST "https://useknockout--api.modal.run/sticker" \
  -H "Authorization: Bearer kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f" \
  -F "file=@photo.jpg" -F "stroke_width=24" -o sticker.png

# Smart crop — tight bounding box around subject
curl -X POST "https://useknockout--api.modal.run/smart-crop" \
  -H "Authorization: Bearer kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f" \
  -F "file=@photo.jpg" -F "padding=32" -o cropped.png

# Studio shot — e-commerce preset (white bg + shadow + centered, 1:1 aspect)
curl -X POST "https://useknockout--api.modal.run/studio-shot" \
  -H "Authorization: Bearer kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f" \
  -F "file=@photo.jpg" -F "aspect=1:1" -F "format=jpg" -o studio.jpg

# Shadow — subject composited onto new bg with a drop shadow
curl -X POST "https://useknockout--api.modal.run/shadow" \
  -H "Authorization: Bearer kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f" \
  -F "file=@photo.jpg" -F "bg_color=#F3F4F6" -o shadow.png

# Compare — before/after side-by-side for marketing/social
curl -X POST "https://useknockout--api.modal.run/compare" \
  -H "Authorization: Bearer kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f" \
  -F "file=@photo.jpg" -o compare.png

# Mask — just the black/white mask, for your own pipeline
curl -X POST "https://useknockout--api.modal.run/mask" \
  -H "Authorization: Bearer kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f" \
  -F "file=@photo.jpg" -o mask.png

# Outline — subject on transparent bg with a thin outline
curl -X POST "https://useknockout--api.modal.run/outline" \
  -H "Authorization: Bearer kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f" \
  -F "file=@photo.jpg" -F "outline_color=#000000" -F "outline_width=4" -o outline.png
```

### Health check

```bash
curl https://useknockout--api.modal.run/health
# {"status":"ok","model":"ZhengPeng7/BiRefNet"}
```

---

## API reference

Base URL: `https://useknockout--api.modal.run`

### `POST /remove`

Remove the background from an uploaded image.

**Headers**

| Header | Required | Description |
|---|---|---|
| `Authorization` | Yes | `Bearer <API_TOKEN>` |
| `Content-Type` | Auto | `multipart/form-data` (set by your client) |

**Body** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | binary | Yes | Image to process (JPEG, PNG, WebP). Max 25 MB. |

**Query params**

| Param | Type | Default | Description |
|---|---|---|---|
| `format` | string | `png` | `png` (default) or `webp`. Both include alpha. |

**Response** — `image/png` or `image/webp` with a transparent background.

### `POST /remove-url`

Fetch an image from a URL and remove its background.

**Headers**

| Header | Required | Description |
|---|---|---|
| `Authorization` | Yes | `Bearer <API_TOKEN>` |
| `Content-Type` | Yes | `application/json` |

**Body** — JSON

```json
{
  "url": "https://example.com/image.jpg",
  "format": "png"
}
```

**Response** — same as `/remove`.

### `POST /replace-bg`

Remove the background and composite the subject onto a new background — solid color or a remote image.

**Headers**

| Header | Required | Description |
|---|---|---|
| `Authorization` | Yes | `Bearer <API_TOKEN>` |
| `Content-Type` | Auto | `multipart/form-data` (set by your client) |

**Body** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | binary | Yes | Foreground image to process. Max 25 MB. |
| `bg_color` | string | No (default `#FFFFFF`) | Hex color for the new background. Examples: `#000000`, `#ff5733`, `#1a73e8`. |
| `bg_url` | string | No | Remote URL of a background image. Takes precedence over `bg_color`. |
| `format` | string | No (default `png`) | Output format: `png`, `webp`, or `jpg` (smallest, opaque only). |

**Response** — `image/png`, `image/webp`, or `image/jpeg` with the subject composited onto the new background. Edges are cleaned via closed-form foreground matting (no color spill, no halo).

### `POST /remove-batch`

Remove backgrounds from up to 10 images in one call.

**Headers**

| Header | Required | Description |
|---|---|---|
| `Authorization` | Yes | `Bearer <API_TOKEN>` |
| `Content-Type` | Auto | `multipart/form-data` |

**Body** — `multipart/form-data` with repeated `files` fields.

**Query params**

| Param | Type | Default | Description |
|---|---|---|---|
| `format` | string | `png` | `png` or `webp`. Applies to every result. |

**Response** — JSON:

```json
{
  "count": 3,
  "format": "png",
  "results": [
    { "filename": "a.jpg", "success": true, "format": "png", "size_bytes": 124503, "data_base64": "..." },
    { "filename": "b.jpg", "success": true, "format": "png", "size_bytes": 98321, "data_base64": "..." },
    { "filename": "c.jpg", "success": false, "error": "Invalid or unsupported image" }
  ]
}
```

Each `data_base64` decodes to PNG/WebP bytes with a transparent background.

### `POST /remove-batch-url`

Same as `/remove-batch` but takes a JSON array of remote URLs.

**Body** — JSON:

```json
{
  "urls": ["https://example.com/a.jpg", "https://example.com/b.jpg"],
  "format": "png"
}
```

**Response** — same JSON shape as `/remove-batch`, with `url` in place of `filename`.

### `POST /mask`

Return just the black/white alpha mask as a grayscale PNG/WebP. Useful for chaining into your own compositing pipeline (Photoshop actions, `ffmpeg` keying, custom workflows).

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | binary | required | Foreground image. |
| `format` | string | `png` | `png` or `webp`. |

**Response** — grayscale image (`0` = background, `255` = subject).

### `POST /smart-crop`

Auto-crop to the subject's tight bounding box + padding.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | binary | required | Foreground image. |
| `padding` | int | `24` | Pixels of padding around the bbox. |
| `transparent` | bool | `true` | `true` → cropped cutout with transparent bg. `false` → cropped region from the original image (bg preserved). |
| `format` | string | `png` | `png`, `webp`, or `jpg` (when `transparent=false`). |

**Response** — cropped image.

### `POST /shadow`

Composite the subject onto a new background with a configurable drop shadow.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | binary | required | Foreground image. |
| `bg_color` | string | `#FFFFFF` | Hex color for the new background. |
| `bg_url` | string | — | Optional remote URL. Takes precedence over `bg_color`. |
| `shadow_color` | string | `#000000` | Hex color for the shadow. |
| `shadow_offset_x` | int | `8` | Shadow offset in pixels (X). |
| `shadow_offset_y` | int | `12` | Shadow offset in pixels (Y). |
| `shadow_blur` | int | `14` | Gaussian blur radius in pixels. |
| `shadow_opacity` | float | `0.45` | 0.0–1.0. |
| `format` | string | `png` | `png`, `webp`, or `jpg`. |

### `POST /sticker`

Subject with a thick outline on a transparent background — iMessage / WhatsApp / Telegram sticker style.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | binary | required | Foreground image. |
| `stroke_color` | string | `#FFFFFF` | Outline color. |
| `stroke_width` | int | `20` | Outline width in pixels (capped at 80). |
| `format` | string | `png` | `png` or `webp`. |

### `POST /outline`

Subject on transparent background with a thin outline.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | binary | required | Foreground image. |
| `outline_color` | string | `#000000` | Outline color. |
| `outline_width` | int | `4` | Outline width in pixels (capped at 60). |
| `format` | string | `png` | `png` or `webp`. |

### `POST /studio-shot`

E-commerce preset: remove background → tight crop → center on solid-color canvas → optional drop shadow → standardized aspect ratio.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | binary | required | Foreground image. |
| `bg_color` | string | `#FFFFFF` | Canvas color. |
| `aspect` | string | `1:1` | `W:H` format. Examples: `1:1`, `4:5`, `16:9`, `3:2`. |
| `padding` | int | `48` | Padding around the subject in pixels. |
| `shadow` | bool | `true` | Include a soft drop shadow. |
| `format` | string | `jpg` | `png`, `webp`, or `jpg`. |

### `POST /compare`

Before/after side-by-side preview — original on the left, transparent cutout (on a checkerboard) on the right. Great for marketing / social media screenshots.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | binary | required | Foreground image. |
| `format` | string | `png` | `png` or `webp`. |

### `POST /headshot` (v0.4.0)

Studio-quality professional headshot — background removed, neutral studio backdrop, optional soft shadow, smart crop to bust framing. One call.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | binary | required | Source portrait. |
| `bg_color` | string | `#f5f5f5` | Studio backdrop hex. |
| `add_shadow` | bool | `true` | Soft drop shadow. |
| `crop` | string | `bust` | `bust`, `head`, or `full`. |
| `format` | string | `png` | `png`, `webp`, or `jpg`. |

### `POST /preview` (v0.4.0)

Cheap, fast low-res preview — 512px max, watermark optional. Use for thumbnail UI before user pays for full-res. Returns in ~1.5s.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | binary | required | Source image. |
| `max_size` | int | `512` | Max edge length. |
| `watermark` | bool | `false` | Add `useknockout` watermark. |

### `POST /estimate` (v0.4.0)

Returns expected processing time + output size **without running the model**. Use to show users "this'll take ~3s, ~1.2 MB" before they hit submit.

```bash
curl -X POST "https://useknockout--api.modal.run/estimate" \
  -H "Content-Type: application/json" \
  -d '{"width": 2048, "height": 1536, "endpoint": "remove"}'
```

Response: `{"estimated_seconds": 2.4, "estimated_output_kb": 1180, "warm": true}`

### `GET /stats` (v0.4.0)

Public stats — total images processed, last-24h count, last-7d trend. Powered by Modal Dict cross-container counter. No auth required.

```bash
curl https://useknockout--api.modal.run/stats
```

### `POST /upscale` (v0.5.0)

**Real-ESRGAN x2/x4 super-resolution.** Takes blurry/small images, outputs 2x or 4x larger with AI-restored detail. Not pixel stretching — invents plausible texture. Tile-based so handles big inputs without OOM.

```bash
curl -X POST "https://useknockout--api.modal.run/upscale" \
  -H "Authorization: Bearer kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f" \
  -F "file=@small.jpg" \
  -F "scale=4" \
  -o upscaled.png
```

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | binary | required | Source image. |
| `scale` | int | `4` | `2` or `4`. |
| `format` | string | `png` | `png`, `webp`, or `jpg`. |

**Use cases:** restore old photos, enlarge product shots, fix low-res screenshots, upscale AI-generated thumbnails.

### `POST /face-restore` (v0.5.0)

**GFPGAN v1.4 face restoration.** Detects faces, restores blurred/compressed/damaged ones while preserving identity. Background also upscaled via Real-ESRGAN. Multi-face safe.

```bash
curl -X POST "https://useknockout--api.modal.run/face-restore" \
  -H "Authorization: Bearer kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f" \
  -F "file=@blurry-portrait.jpg" \
  -o restored.png
```

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | binary | required | Source image with one or more faces. |
| `format` | string | `png` | `png`, `webp`, or `jpg`. |

**Use cases:** old family photos, Zoom screenshots, dating app pics, restore CCTV stills.

### `GET /health`

Returns `{"status":"ok","model":"ZhengPeng7/BiRefNet"}`. No auth required.

### `GET /docs`

Interactive OpenAPI (Swagger) UI.

### Errors

| Code | Meaning |
|---|---|
| `400` | Invalid image, missing field, malformed URL, invalid hex color, or batch > 10 items |
| `401` | Missing `Authorization` header |
| `403` | Invalid bearer token |
| `413` | Image exceeds 25 MB limit |
| `500` | Server error (check dashboard logs) |

### Edge quality

All endpoints apply closed-form foreground matting (via [pymatting](https://github.com/pymatting/pymatting)) after mask prediction. This estimates pure foreground color at soft edges, eliminating color spill from the original background. Result: no halos, no fringing, even on backgrounds that differ sharply from the subject.

---

## Client examples

### Python

```python
import requests

URL = "https://useknockout--api.modal.run/remove"
TOKEN = "kno_public_beta_4d7e9f1a3c5b2e8d6a9f7c1b3e5d8a2f"  # public beta token

with open("input.jpg", "rb") as f:
    resp = requests.post(
        URL,
        headers={"Authorization": f"Bearer {TOKEN}"},
        files={"file": f},
    )
resp.raise_for_status()

with open("output.png", "wb") as f:
    f.write(resp.content)
```

### Node.js SDK (recommended)

```bash
npm i @useknockout/node
```

```ts
import { writeFile } from "node:fs/promises";
import { Knockout } from "@useknockout/node";

const client = new Knockout({ token: process.env.KNOCKOUT_TOKEN! });

// 1. Remove background → transparent PNG
const png = await client.remove({ file: "./input.jpg" });
await writeFile("out.png", png);

// 2. Replace background with a color
const jpg = await client.replaceBackground({
  file: "./input.jpg",
  bgColor: "#FF5733",
  format: "jpg",
});
await writeFile("out.jpg", jpg);

// 3. Replace background with a remote image
const composed = await client.replaceBackground({
  file: "./input.jpg",
  bgUrl: "https://example.com/mountains.jpg",
});

// 4. Batch — process 10 URLs in one call
const batch = await client.removeBatchUrl({
  urls: ["https://example.com/a.jpg", "https://example.com/b.jpg"],
});
for (const r of batch.results) {
  if (r.success) await writeFile(`out-${r.url}.png`, Buffer.from(r.data_base64!, "base64"));
}
```

### Node.js (raw fetch, no SDK)

```js
import { readFile, writeFile } from "node:fs/promises";

const URL = "https://useknockout--api.modal.run/remove";
const TOKEN = process.env.KNOCKOUT_TOKEN;

const buf = await readFile("input.jpg");
const form = new FormData();
form.set("file", new Blob([buf]), "input.jpg");

const res = await fetch(URL, {
  method: "POST",
  headers: { Authorization: `Bearer ${TOKEN}` },
  body: form,
});
if (!res.ok) throw new Error(await res.text());

await writeFile("output.png", Buffer.from(await res.arrayBuffer()));
```

### TypeScript (browser / Next.js)

```ts
export async function removeBackground(file: File, token: string) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch("https://useknockout--api.modal.run/remove", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });

  if (!res.ok) throw new Error(`knockout error: ${res.status}`);
  return await res.blob(); // PNG with alpha
}
```

### Go

```go
package main

import (
    "bytes"
    "io"
    "mime/multipart"
    "net/http"
    "os"
)

func removeBG(path, token string) ([]byte, error) {
    f, err := os.Open(path)
    if err != nil { return nil, err }
    defer f.Close()

    body := &bytes.Buffer{}
    w := multipart.NewWriter(body)
    part, _ := w.CreateFormFile("file", path)
    io.Copy(part, f)
    w.Close()

    req, _ := http.NewRequest("POST",
        "https://useknockout--api.modal.run/remove", body)
    req.Header.Set("Authorization", "Bearer "+token)
    req.Header.Set("Content-Type", w.FormDataContentType())

    resp, err := http.DefaultClient.Do(req)
    if err != nil { return nil, err }
    defer resp.Body.Close()
    return io.ReadAll(resp.Body)
}
```

### cURL — WebP output (smaller files)

```bash
curl -X POST "https://useknockout--api.modal.run/remove?format=webp" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@input.jpg" \
  -o output.webp
```

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

Want to run your own instance? One command after Modal setup.

### Prerequisites

```bash
pip install modal
modal token new
```

### Clone & deploy

```bash
git clone https://github.com/useknockout/api.git
cd api

# create your bearer-token secret
modal secret create knockout-secrets API_TOKEN=$(openssl rand -hex 32)

# deploy
modal deploy main.py
```

Modal prints your live HTTPS URL. First deploy takes ~5 min (image build + weight bake). Subsequent deploys take seconds.

### Tune for your workload

Edit `main.py`:

```python
@app.cls(
    gpu="L4",              # or "A10", "A100", "H100"
    scaledown_window=60,   # seconds of idle before scale-to-zero
    max_containers=10,     # max concurrent containers
)
```

- **Latency-critical?** Keep one warm: `min_containers=1` (costs ~$0.80/hr 24/7).
- **Throughput-critical?** Bump `max_containers` and use `@modal.concurrent(max_inputs=4)` to batch.
- **Higher quality?** Change `MODEL_INPUT_SIZE` to `(2048, 2048)` — 4x slower, sharper edges.

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

The hosted API at `useknockout--api.modal.run` is in **closed beta** while we validate quality. Request an API key: [contact](#contact).

When the paid tier goes live:

| Tier | Price | Best for |
|---|---|---|
| **Free** | 50 images / month, no card | Personal, eval, open source |
| **Pay-as-you-go** | $0.005 / image | Side projects, early startups |
| **Volume** | $0.003 / image at 100k+/mo | Production workloads |
| **Enterprise** | Custom, private endpoints | Compliance, BYO-cloud |

For reference — the same image on remove.bg is **$0.20** at their PAYG rate.

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
