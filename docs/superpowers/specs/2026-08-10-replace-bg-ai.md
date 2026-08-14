# Spec: `/replace-bg-ai` — AI-generated backgrounds behind a real cutout

Status: DRAFT, not implemented. Written 2026-08-10.
Owner: useknockout-api (backend).
Purpose: give the channel partner (Tomek/Kravento) an AI scene-generation
feature to test, using a design where the product pixels are never regenerated.

## Why this shape, and not `/edit`

The obvious ask was an image-editing endpoint: send a photo + prompt to a
generative model, get an edited photo back. Three things kill that for this
customer:

1. **The gateway available for free evaluation cannot do it.** The ASU AIML
   gateway strips image content blocks before they reach the model
   (`agent-os/backend/app/llm/providers/asu_gateway.py:47-50`, verified by
   Troy's own earlier probes). It is text-to-image only. So image-to-image
   evaluation there is impossible, not merely awkward.
2. **Generative editing regenerates the product.** Colour and texture drift.
   For a customer selling tissue paper by exact shade, that is a defect, not a
   feature.
3. **Resolution.** Their source is 3072x4080 (12.5MP). Current image models cap
   output around 1-2MP, so a "cleaned up" photo comes back materially smaller
   than the original.

Splitting the job avoids all three:

```
product photo ──> BiRefNet cutout (ours, full res, pixels untouched)
                                                       │
prompt ────────> text-to-image background ─────────────┴──> composite
```

The model never sees the product, so it cannot alter it, and text-to-image is
exactly what the free evaluation gateway supports. The marketing claim this
enables is one generic editors cannot make: **"new background, and we can prove
the product is untouched."**

## Endpoint

```
POST /replace-bg-ai        multipart/form-data
```

| Field | Type | Default | Notes |
| :--- | :--- | :--- | :--- |
| `file` | file | required | The product photo |
| `prompt` | str | required | Scene description, 3-500 chars |
| `model` | str | `auto` | `auto` \| `flux2-pro` \| `mai-image-2e` \| `gpt-image-2` \| `nano-banana` |
| `format` | str | `jpg` | png/webp/jpg — output is opaque |
| `detect` | str | `standard` | Passed to `_acquire_mask`, same as /remove |
| `decontaminate` | bool | `false` | Passed to `_acquire_mask`, same as /remove |
| `quality`, `max_dim`, `width`, `height`, `despill`, `watermark`, `watermark_opacity`, `preset` | | | Same semantics as `/replace-bg` |

`model=auto` resolves to whichever provider Troy's evaluation picks as default.
The explicit values exist so the partner can A/B them and report a preference;
the param is a research instrument first and a product feature second.

## Implementation sketch

Reuses existing machinery. The only new code is background acquisition.

```python
@web.post("/replace-bg-ai")
def replace_bg_ai_endpoint(...):
    ctx, _t = self._begin(authorization, "/replace-bg-ai")
    self._require_ai_bg(ctx)                     # allowlist gate, see below
    prompt = self._check_prompt(prompt)          # length + emptiness
    model  = self._check_bg_model(model)         # enum -> 400 on unknown
    detect = self._check_detect(detect)
    if detect == "high_recall":
        self._require_paid_compute(ctx, "detect=high_recall")

    fg = self._open_image(file.file.read())
    bg = self._generate_background(prompt, model, size=fg.size)   # NEW
    composited = self._composite_on_bg(fg, bg, despill=despill,
                                       detect=detect,
                                       decontaminate=decontaminate)
    ...
```

`_composite_on_bg` currently calls `_get_mask` directly (main.py:1113-1121). It
should be switched to `_acquire_mask` so `detect`/`decontaminate` flow through
here the same way they now do for `/remove`, `/smart-crop`, `/studio-shot`.
That is a one-line change plus two kwargs, and it is a prerequisite.

### `_generate_background(prompt, model, size)`

1. Map `model` -> provider + deployment name.
2. Call the provider's text-to-image endpoint asking for the nearest supported
   aspect ratio to `size`.
3. Decode the returned b64/URL to a PIL image.
4. Upscale/cover-crop to exactly `size` (LANCZOS). The background is a backdrop,
   so mild upscaling is acceptable — this is why generating at 1-2MP does not
   cap our output resolution the way generative *editing* would.
5. On any provider error: raise 502 `"background generation failed"`. Do NOT
   silently fall back to a solid colour — the caller paid for a scene.

Cache key `sha256(prompt + model + aspect)` in a `modal.Dict` with a short TTL,
so a partner running the same prompt across 20 product shots pays for one
generation instead of 20. This is the single biggest cost lever in the feature.

### Provider routing

| `model` | Provider | Endpoint shape |
| :--- | :--- | :--- |
| `flux2-pro` | Azure AI Foundry | `POST {AZURE_FOUNDRY_ENDPOINT}/openai/deployments/{dep}/images/generations?api-version=2025-04-01-preview`, header `Api-Key`. FLUX.2's native param route is `https://{resource}.api.cognitive.microsoft.com/providers/blackforestlabs/v1/flux-2-flex?api-version=preview` if `guidance`/`steps` control is wanted later. |
| `mai-image-2e` | Azure AI Foundry | same deployments/images/generations shape |
| `gpt-image-2` | Azure AI Foundry | same |
| `nano-banana` | GCP Gemini | `GOOGLE_API_KEY` |

Confirmed available on resource `ai-agents-os` (queried 2026-08-08):
`FLUX.2-pro` GA, `FLUX.1-Kontext-pro` GA, `gpt-image-2` GA,
`MAI-Image-2e-2026-04-09` preview, `MAI-Image-2.5-Pro-2026-06-19` preview.
**Open item: deployment NAMES are unverified** — the deployments-list route
404s on this resource, so the names must be read from the Foundry portal
(ai.azure.com -> project -> Deployments) before wiring.

### Secrets

New Modal secret, separate from `knockout-secrets` so the experiment can be
deleted in one command without touching production credentials:

```
modal secret create knockout-ai-bg \
  AZURE_FOUNDRY_API_KEY=... \
  AZURE_FOUNDRY_ENDPOINT=https://ai-agents-os.openai.azure.com \
  GOOGLE_API_KEY=...
```

Values live in `agent-os/.env` already. Troy creates the secret; the assistant
never reads the values.

Do NOT wire the ASU gateway token into the deployed API. ASU is for Troy's own
model evaluation only, not for serving a commercial partner's traffic.

## Gating

```python
AI_BG_ALLOWLIST = frozenset(filter(None,
    os.environ.get("AI_BG_ALLOWLIST", "").split(",")))
```

Empty set = feature off for everyone (default). Add the partner's Supabase
`user_id` to enable. `is_legacy` (Troy's own token) always passes. This makes
"turn it off" a secret edit, not a deploy, and guarantees no customer can come
to depend on it during the experiment.

Also:
- NOT in `FREE_TIER_ENDPOINTS`, NOT in `DEMO_ENDPOINTS`.
- Daily cap (`AI_BG_DAILY_CAP`, default 50) on the same `modal.Dict` counter
  pattern as `_enforce_demo_limit`, because each call spends real money against
  Troy's provider account.

## Billing

Provider cost is ~$0.02-0.19 per generated background depending on model, which
is at or above the $0.02/image list price. So this cannot bill on the existing
`images.processed` meter at the normal rate.

Options, decision needed before any public launch:
1. Own Stripe meter `ai.backgrounds` at a price that covers the worst model.
2. Bill 1 image unit + a flat surcharge.
3. During the experiment: allowlist-only, absorb the cost, meter nothing beyond
   the normal image unit. **Recommended for the 20-test evaluation.**

The prompt cache above materially changes the economics of options 1 and 2 for
batch users; measure the hit rate during the experiment.

## Evaluation plan (Troy, before the partner sees it)

Free, on the ASU gateway, since it is text-to-image and that is all this needs:
`nano_banana_pro`, `gpt_image2`, `geminiflash2_5_image`.

Generate the same 5 prompts across the 3 models, composite each behind an
existing cutout from `eval/cases/`, and judge: does the backdrop read as a
photographed surface, is the lighting direction plausible against the product,
does it tile/repeat, does it produce text artifacts. Pick a default for
`model=auto`; wire only the winner plus one alternate to Azure.

The partner then runs their own tests against the Azure-backed endpoint. Do not
pre-run the partner's tests — they want their own read.

## Prerequisites

1. Azure deployment names read from the Foundry portal (blocking).
2. `modal secret create knockout-ai-bg` (Troy).
3. `_composite_on_bg` switched from `_get_mask` to `_acquire_mask`.

## Non-goals

- No image-to-image editing. If that is ever wanted it is a different endpoint
  with different guarantees, and it cannot be evaluated on the ASU gateway.
- No SDK propagation until the experiment concludes and the endpoint is
  promoted from allowlist to general availability.
- No prompt-safety filtering beyond what the providers apply themselves; the
  allowlist is the containment mechanism during the experiment.
