# Spec: Video + GIF Background Removal (staged)

Status: design only. Not built inline with the 2026-06-25 image-feature batch
(quality, resize, watermark, despill, PSD, presets). Video is a different
architecture and a different billing model — it gets its own build + deploy.

## Why separate

The image endpoints are one model pass per request. Video is N frames per
request, each a full BiRefNet pass, plus demux/remux. A 10s clip at 30fps =
300 model passes = 300x the GPU time and cost of one image. Throwing this on
the existing synchronous request path would block a container for minutes and
blow request timeouts. It needs async jobs.

## Architecture

1. Upload -> object storage (S3/R2/Supabase Storage), return a `job_id`.
2. Async worker (Modal function, not the ASGI request) processes the job:
   - `ffmpeg` demux to frames (PNG) + extract audio track.
   - Batch frames through BiRefNet (reuse `_get_mask` / `_clean_foreground`;
     batch on GPU for throughput, not one-at-a-time).
   - Optional temporal smoothing of the alpha across frames to kill flicker
     (EMA or guided filter on the mask sequence — single-frame masks jitter).
   - Composite per frame: transparent, solid color, or `--bg-image`.
   - `ffmpeg` remux. Output codecs:
     - **ProRes 4444** (`-c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le`)
       for true alpha (the pro/editing path LocalBG markets).
     - WebM/VP9 with alpha (`-pix_fmt yuva420p`) for web.
     - MP4 (H.264) when compositing onto an opaque bg (no alpha).
   - Re-attach audio.
3. Poll `GET /jobs/{job_id}` for status + result URL. Webhook optional.

GIF: same pipeline, frames are the GIF frames; output animated GIF (1-bit alpha)
or WebP (better alpha). GIF alpha is binary — matte hard, no soft edges.

## Endpoints

- `POST /video/remove` (multipart or url) -> `{job_id}`.
  Params: `format` (prores4444|webm|mp4|gif|webp), `bg_color`, `bg_image`,
  `smoothing` (0-100 temporal), `fps_cap`, `max_seconds`.
- `GET /jobs/{job_id}` -> `{status, progress, result_url?, error?}`.
- Reuse the image despill/quality knobs per frame where they apply.

## Billing

Not per-image. Options:
- Per output-second, or
- Per processed frame (frames = duration * min(fps, fps_cap)).
Needs a new Stripe metered price + a per-job meter event sized by frame count.
Set a hard `max_seconds` and `fps_cap` on free/payg to bound worst-case GPU
spend. Gate behind paid tier only (no free video — too expensive).

## Image/runtime deps to add

- `ffmpeg` via `apt_install("ffmpeg")`.
- ProRes needs an ffmpeg build with prores_ks (debian ffmpeg has it).
- Storage client (boto3 for S3/R2, or Supabase Storage REST).
- A Modal async function + a jobs table (id, user_id, status, input_url,
  result_url, frames, created_at).

## Open questions

- Storage backend: R2 (cheap egress) vs Supabase Storage (already in stack).
- Max clip length on payg before it gets cost-prohibitive.
- Temporal smoothing method — start simple (alpha EMA), measure flicker.
- Whether to expose a sync path for very short clips (< ~2s) to skip the job
  dance, or always async for consistency. Lean: always async.

## Effort

High. ~1 to 2 focused sessions: storage + jobs table + async worker + ffmpeg
demux/remux + ProRes/WebM encode + temporal smoothing + billing meter + SDK
surface. Build after the image batch ships and is validated.
