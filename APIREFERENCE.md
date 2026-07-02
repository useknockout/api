# useknockout API reference

[← Back to the README](./README.md)

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
| `quality` | int | — | Compression quality 1–100 for `webp`. Ignored for `png` (lossless). |
| `max_dim` | int | — | Resize so the longest side is ≤ this many px (aspect preserved). |
| `width` / `height` | int | — | Exact resize. One alone preserves aspect; both set an exact box. |
| `despill` | float | — | **Knockout Plus.** Edge color decontamination strength `0`–`100`. Default behavior is full despill; lower to preserve original edge color. |
| `watermark` | string | — | **Knockout Plus.** Text watermark, bottom-right, auto-scaled. |
| `watermark_opacity` | float | `0.5` | Watermark opacity `0.0`–`1.0`. |
| `preset` | string | — | **Knockout Plus.** Apply a saved preset by name (see `/presets`). Explicit params override preset values. |

Params marked **Knockout Plus** require a `pro`-tier key; other callers receive `402`. `quality`, `max_dim`, `width`, `height` are available on all paid tiers.

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

### `POST /psd`

Background removal exported as a **layered Photoshop `.psd`** — the cutout sits on its own transparent layer ("Cutout"), ready to edit in Photoshop, Affinity, or Photopea.

Paid endpoint. Billed as a **$0.10/image add-on** (its own meter, independent of your base per-image rate). **Included free with Knockout Plus.**

**Headers**

| Header | Required | Description |
|---|---|---|
| `Authorization` | Yes | `Bearer <API_TOKEN>` (paid tier) |
| `Content-Type` | Auto | `multipart/form-data` |

**Body** — `multipart/form-data`

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | binary | — | Image to process. Max 25 MB. |
| `max_dim` / `width` / `height` | int | — | Resize the output (see `/remove`). |
| `despill` | float | — | **Knockout Plus.** Edge decontamination `0`–`100`. |
| `watermark` | string | — | **Knockout Plus.** Text watermark. |
| `preset` | string | — | **Knockout Plus.** Apply a saved preset. |

**Response** — `image/vnd.adobe.photoshop` (a `.psd` with a transparent cutout layer).

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
| `quality` | int | No | Compression quality 1–100 (`jpg`/`webp`). |
| `max_dim` / `width` / `height` | int | No | Resize the output (see `/remove`). |
| `despill` | float | No | **Knockout Plus.** Edge decontamination `0`–`100`. |
| `watermark` / `watermark_opacity` | string / float | No | **Knockout Plus.** Text watermark + opacity. |
| `preset` | string | No | **Knockout Plus.** Apply a saved preset. |

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
| `transparent` | bool | `false` | Keep a transparent background. `bg_color` and `shadow` are ignored; output is forced to PNG. |
| `enhance` | bool | `false` | Off by default. Set `true` for a subtle brightness + saturation lift (ecommerce-ready). Leave off for true-to-life color. |
| `enhance_strength` | float | `0.15` | Lift amount, `0.0`–`0.5`. Only applies when `enhance=true`. |
| `format` | string | `jpg` | `png`, `webp`, or `jpg`. |
| `quality` | int | — | Compression quality 1–100 (`jpg`/`webp`). |
| `max_dim` / `width` / `height` | int | — | Resize the output (see `/remove`). |
| `despill` | float | — | **Knockout Plus.** Edge decontamination `0`–`100`. |
| `watermark` / `watermark_opacity` | string / float | — | **Knockout Plus.** Text watermark + opacity. |
| `preset` | string | — | **Knockout Plus.** Apply a saved preset. |

### `POST /presets` · `GET /presets` · `DELETE /presets/{name}`

**Knockout Plus.** Save reusable output configs and apply them by name with `preset=<name>` on `/remove`, `/replace-bg`, `/studio-shot`, and `/psd`. A preset sets defaults; any explicit param on the request overrides it.

- **`POST /presets`** — create/update. JSON body `{ "name": "web-thumb", "config": { "max_dim": 800, "quality": 75, "despill": 60, "watermark": "© Brand" } }`. Config keys: `quality`, `max_dim`, `width`, `height`, `despill`, `watermark` (unknown keys dropped).
- **`GET /presets`** — list your presets.
- **`DELETE /presets/{name}`** — delete one.

Presets are per-user (tied to your API key's account). Requires a `pro`-tier key.

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

### `POST /upscale` (v0.6.0)

**Swin2SR / Real-ESRGAN x2/x4 super-resolution.** Takes blurry/small images, outputs 2x or 4x larger with AI-restored detail. Not pixel stretching — invents plausible texture.

**v0.6.0** — default backend switched to **Swin2SR** (SwinV2 Transformer, successor to SwinIR). Sharper detail and more natural texture on real photos compared to Real-ESRGAN, which tends to produce a painted / plastic look on faces. Real-ESRGAN remains available via `model=realesrgan` and is still the better choice for anime / illustrations.

```bash
# default — Swin2SR (best for real photos)
curl -X POST "https://useknockout--api.modal.run/upscale" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "file=@small.jpg" \
  -F "scale=4" \
  -o upscaled.png

# legacy — Real-ESRGAN (best for anime / illustrations)
curl -X POST "https://useknockout--api.modal.run/upscale" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
  -F "file=@art.png" \
  -F "scale=4" \
  -F "model=realesrgan" \
  -o upscaled.png
```

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | binary | required | Source image. |
| `scale` | int | `4` | `2` or `4`. |
| `model` | string | `realesrgan` | `realesrgan` (default) restores detail — best on low-res/degraded photos. `swin2sr` is faithful (no invented detail) for accuracy-sensitive work. |
| `face_enhance` | bool | `false` | Route through GFPGAN for facial detail. Implies Real-ESRGAN backend. |
| `format` | string | `png` | `png`, `webp`, or `jpg`. |

**Use cases:** restore old photos, enlarge product shots, fix low-res screenshots, upscale AI-generated thumbnails.

### `POST /face-restore` (v0.5.0)

**GFPGAN v1.4 face restoration.** Detects faces, restores blurred/compressed/damaged ones while preserving identity. Background also upscaled via Real-ESRGAN. Multi-face safe.

```bash
curl -X POST "https://useknockout--api.modal.run/face-restore" \
  -H "Authorization: Bearer $KNOCKOUT_TOKEN" \
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
import os
import requests

URL = "https://useknockout--api.modal.run/remove"
TOKEN = os.environ["KNOCKOUT_TOKEN"]  # get yours at useknockout.com/signin

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
