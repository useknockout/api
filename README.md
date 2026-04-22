# useknockout

State-of-the-art background removal API. Powered by [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (MIT license) on Modal GPUs.

Drop an image in, get a PNG with a transparent background out. Accurate edges on hair, fur, glass, and complex subjects — comparable to remove.bg / Photoroom, self-hostable, and 40x cheaper to run.

- **Model:** BiRefNet (SOTA on DIS5K, HRSOD, COD benchmarks)
- **Runtime:** Modal, L4 GPU, scale-to-zero
- **Latency:** ~200ms per image warm, ~10s cold start
- **License:** MIT (model + code)

## Quick start

### 1. Install prerequisites

```bash
pip install modal
modal token new   # opens browser, links your Modal account
```

### 2. (Optional) Set an API token

```bash
modal secret create knockout-secrets API_TOKEN=$(openssl rand -hex 32)
```

If no token is set, the API is open (fine for dev, **not** for production).

### 3. Deploy

```bash
modal deploy main.py
```

Modal prints your live URL, e.g.:
```
https://useknockout--api-api.modal.run
```

### 4. Call it

**File upload:**
```bash
curl -X POST "$URL/remove" \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "file=@cat.jpg" \
  -o cat-nobg.png
```

**Remote URL:**
```bash
curl -X POST "$URL/remove-url" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/cat.jpg"}' \
  -o cat-nobg.png
```

**WebP output (smaller files):**
```bash
curl -X POST "$URL/remove?format=webp" \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "file=@cat.jpg" \
  -o cat-nobg.webp
```

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/remove` | multipart `file` | `image/png` or `image/webp` w/ alpha |
| POST | `/remove-url` | JSON `{"url": "...", "format": "png"}` | `image/png` or `image/webp` w/ alpha |
| GET | `/health` | — | `{"status": "ok", "model": "..."}` |
| GET | `/docs` | — | OpenAPI/Swagger UI |

## Cost

Modal L4: ~$0.80/hr while warm, **$0 when idle** (scales to zero).

At 200ms per image, one warm container serves ~18,000 images/hr for $0.80 → **~$0.000045/image raw cost**. That's ~4,500x cheaper than remove.bg's PAYG pricing.

## Architecture

```
Client → Modal HTTPS endpoint → BiRefNet on L4 GPU → PNG w/ alpha
```

Single file. One decorator per endpoint. Weights baked into the container image at build time — cold starts are fast (image pull + model-to-GPU, no HF download).

## License

MIT — code and model weights. Commercial use OK.
