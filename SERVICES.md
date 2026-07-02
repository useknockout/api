---
project: projects/useknockout-api
type: services
---

# Services — useknockout-api

Hosted / third-party services this GPU image API depends on.

## Hosting / compute

| Service | Role |
|---|---|
| **Modal** | Serverless GPU platform. Hosts the FastAPI app as a Modal `App` (`@app.cls`, `gpu="L4"`, `@modal.asgi_app`), with scale-to-zero, secrets (`knockout-secrets`), and a `modal.Dict` (`knockout-stats`) used as a cross-container counter / demo rate-limit store. Deployed via `modal deploy main.py`. |

## Called at runtime

| Service | Role | Auth |
|---|---|---|
| **Supabase REST** (`/rest/v1/...`) | Per-user token auth, tier lookup, usage logging, monthly-quota view. | per-user token |
| **Stripe Billing — Meter Events API** (`/v1/billing/meter_events`) | Usage-based billing for paid tiers. | Stripe key |

## Called at build time (model weights)

**Hugging Face Hub**, **ModelScope**, and **GitHub releases** — pull BiRefNet, Swin2SR, Real-ESRGAN, GFPGAN, facexlib, DDColor, and LaMa weights, baked into the Modal image so cold starts skip downloads.

No LLM providers — every model is a computer-vision/image model.
