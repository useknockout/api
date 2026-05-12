# `/inpaint` — v0.8.0 design

**Date:** 2026-05-12
**Author:** Code agent
**Status:** Approved (via /background mode)

## Goal

Add a deterministic, prompt-free inpainting endpoint that lets users erase regions of an image and have the model fill the hole with plausible background. Matches existing brand: one endpoint per operation, no prompts, opinionated defaults, MIT-friendly.

## Endpoint contract

```
POST /inpaint
Content-Type: multipart/form-data
Authorization: Bearer <token>

Required:
  file: image       (the photo to inpaint)

Mode is auto-detected:
  (auto-subject)    no mask, no bbox       → BiRefNet derives subject mask, inverts it
  mask: image       a PNG/JPEG mask        → white pixels = inpaint, black = keep
  x,y,w,h: int      bbox in image pixels   → synthesized into a white-on-black mask

Optional:
  dilation: int     default 8, range 0..32 (the escape hatch)
  format: str       default "png", one of png|webp|jpg

Returns:
  Content-Type: image/{format}
  Body: full-resolution inpainted image

Headers on response:
  x-knockout-model: big-lama
  x-knockout-mode:  auto-subject | mask | bbox
```

**Mode precedence** when multiple fields present: `mask` > `bbox` > auto-subject. Documented in the GET-info handler.

## Architecture

### Image build

```python
.pip_install("simple-lama-inpainting==0.1.2")
```

### `_download_model` addition

Bake LaMa weights into the Modal image so cold starts skip the ~200 MB download:

```python
from simple_lama_inpainting import SimpleLama
SimpleLama()  # triggers one-time weight download into cache
```

### `Knockout.load()` addition

```python
self.inpainter = SimpleLama()
```

### `_inpaint` helper

Single private method on the `Knockout` class:

```
_inpaint(image: PIL.Image, mask: PIL.Image, dilation: int) -> PIL.Image:
  1. Dilate mask by `dilation` px (cv2.dilate, ELLIPSE kernel)
  2. Compute scale: target_max = 1024 / max(image.size)
  3. If scale < 1: downscale image + mask to 1024 max-edge (Lanczos for image, NEAREST for mask)
  4. inpainted_small = self.inpainter(image_small, mask_small)
  5. Upscale inpainted_small back to original size (Lanczos)
  6. Composite: only replace pixels where original (un-resized) mask was non-zero
  7. Return PIL.Image at original full resolution
```

Step 6 keeps unmasked pixels byte-identical to input. This is the trick that makes the result feel high-resolution even though LaMa ran at 1024px.

### Endpoint flow (`POST /inpaint`)

```
1. _begin(authorization, "/inpaint") → ctx, t (existing auth + logging)
2. _check_format(format, allowed={png, webp, jpg})
3. Parse multipart: file (required), mask (optional), x, y, w, h (optional)
4. Open image with _open_image(file)
5. Determine mode:
     a. mask present → open mask, validate size, resize NEAREST to image size if needed, mode="mask"
     b. x,y,w,h present → validate bounds, synthesize mask (PIL.Image.new + ImageDraw rectangle), mode="bbox"
     c. otherwise → run _get_mask(image) for BiRefNet, invert (255 - mask), mode="auto-subject"
6. Reject empty masks (no white pixels) → 400
7. inpainted = self._inpaint(image, mask, dilation)
8. _bump_counter()
9. Return Response with body + x-knockout-model and x-knockout-mode headers
10. _end(ctx, "/inpaint", t)
```

## Error handling

| Failure | Status | Response body |
|:---|:---:|:---|
| Empty mask (no white pixels) | 400 | `"Mask has no pixels to inpaint."` |
| Mask >50% of image | proceed | Warn via `x-knockout-warning` header |
| bbox out of bounds | 400 | `"bbox extends outside the image"` |
| Mask dims don't match image | auto-fix | Resize mask NEAREST to image dims |
| Auto-subject mode but BiRefNet finds nothing | 422 | `"No subject detected. Send mask or bbox."` |
| Image > 4096px on either axis | 413 | Matches existing endpoint convention |
| LaMa runtime error | 500 | `"inpaint failed: {detail}"` |
| Invalid dilation value | 400 | `"dilation must be 0..32"` |

## Testing

**Smoke tests** (all against `examples/group-on-mountains.png`):

1. **auto-subject mode** — POST file only → 200, output PNG with people removed
2. **mask mode** — POST file + `examples/mask-example.png` → 200, masked region replaced
3. **bbox mode** — POST file + `x=100&y=100&w=300&h=400` → 200, rectangular region replaced

**Local syntax check** via `python -m py_compile main.py`.

**Production verification** via curl after `modal deploy main.py`:
- `GET /inpaint` returns 200 with info JSON (not 405)
- `POST /inpaint` with multipart file returns 200 with image bytes
- `GET /` listing includes `POST /inpaint`
- Version bumped to `0.8.0`

## Ship sequence

Mirrors the /colorize and /silhouette ships:

1. Commit endpoint code → push → `modal deploy main.py`
2. Update memory + POSTS.md (22 → 23 endpoints)
3. Bump 4 SDKs 0.1.1 → 0.2.0 (minor — new method, not patch)
4. Add `client.inpaint()` to Node SDK, `useInpaint()` to React, `inpaint` subcommand to CLI, `client.inpaint()` + async to Python
5. Publish all 4 to npm/PyPI (user runs — needs tokens)
6. Tag `v0.8.0` on API repo, `v0.2.0` on each SDK repo
7. Append item to `AGENT_COMMS.md` for Desktop to add `/inpaint` to landing page

## Versioning rationale

- **API v0.8.0** (minor): new model (LaMa), new endpoint, behaves differently from anything that existed before. Earns the minor bump.
- **SDKs 0.1.1 → 0.2.0**: new public method on a class is a minor change in semver.

## What's out of scope

- Multiple-region bbox in one call (just call /inpaint multiple times)
- OCR-based watermark auto-detection (user sends mask)
- Stable Diffusion inpainting (generative, prompt-based, off-brand)
- Per-region quality tuning (one global `dilation`, no per-region knobs)
- Post-inpaint Swin2SR cleanup (composable; user can chain `/inpaint` then `/upscale`)
