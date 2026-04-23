# useknockout

> State-of-the-art background removal. Open source. 40x cheaper than remove.bg.

A production-grade background removal API powered by [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — the current SOTA on DIS5K, HRSOD, and COD benchmarks. Served on Modal's GPU infrastructure with scale-to-zero economics.

- **SOTA quality** — matches or beats remove.bg, Photoroom, and Pixelcut on hair, fur, fine detail
- **Fast** — ~200ms per image on a warm L4 GPU
- **Cheap** — ~$0.00005 per image raw compute cost (4,000x cheaper than remove.bg PAYG)
- **MIT licensed** — model weights and code, commercial use OK
- **Self-hostable** — deploy to your own Modal workspace in one command

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
- [Roadmap](#roadmap)
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

### Hit the API in 3 seconds

```bash
curl -X POST "https://useknockout--api.modal.run/remove" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@your-image.jpg" \
  -o out.png
```

You get a PNG with a transparent alpha channel. Done.

### With a URL instead of a file

```bash
curl -X POST "https://useknockout--api.modal.run/remove-url" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/cat.jpg"}' \
  -o out.png
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

### `GET /health`

Returns `{"status":"ok","model":"ZhengPeng7/BiRefNet"}`. No auth required.

### `GET /docs`

Interactive OpenAPI (Swagger) UI.

### Errors

| Code | Meaning |
|---|---|
| `400` | Invalid image, missing field, or malformed URL |
| `401` | Missing `Authorization` header |
| `403` | Invalid bearer token |
| `413` | Image exceeds 25 MB limit |
| `500` | Server error (check dashboard logs) |

---

## Client examples

### Python

```python
import requests

URL = "https://useknockout--api.modal.run/remove"
TOKEN = "YOUR_TOKEN"

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

### Node.js (fetch)

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

## Roadmap

- [x] BiRefNet on L4 with FastAPI ASGI (shipped)
- [ ] Optional post-processing: matting refinement, edge smoothing
- [ ] SAM 2 prompt-driven mode (click to keep/remove a region)
- [ ] Batch endpoint (`POST /remove-batch` with N images)
- [ ] Webhook callbacks for async processing
- [ ] CDN cache for repeat images (content-hash keyed)
- [ ] Official SDKs: `@useknockout/node`, `useknockout` (PyPI)
- [ ] Self-serve dashboard + Stripe billing

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
