# Problem brief: white halo on lightbox flat-lays

Written 2026-08-17 by the useknockout-api backend session, for review by other
AI sessions / engineers. **Self-contained: no prior context needed.**

We are asking for critique and better ideas. We are explicitly open to swapping
models, adding open-source libraries or SDKs, changing the architecture, or
throwing out the approach below entirely. Nothing here is committed.

**If you name a model, library or paper, please make sure it actually exists.**
Every model named in this document was verified against Hugging Face directly.
Confidently-worded suggestions that turn out to be invented cost us real time.
Where you are unsure, say so.

---

## 1. What useknockout is

A commercial background-removal API (~28 endpoints) running on Modal GPUs (L4),
live at `https://useknockout--api.modal.run`, version 0.13.0. Real paying
customers. Python + FastAPI, single `main.py`.

## 2. What we run today (verified against the code)

```
input image
  -> BiRefNet inference               (ZhengPeng7/BiRefNet, MIT, fixed 1024x1024 transform)
  -> _refine_alpha                    (grayscale guided filter, radius 8, eps 1e-3, guide = image)
  -> optional detect=high_recall      (2nd pass on a brightness-boosted copy, union of raw alphas)
  -> optional decontaminate           (global Lab + k-means band classifier)
  -> _clean_foreground                (PyMatting estimate_foreground_cf / _ml, AFTER alpha is final)
  -> composite / output
```

Relevant helpers in `main.py`: `_get_mask` (954), `_refine_alpha` (992),
`_acquire_mask` (1332), `_clean_foreground` (1099/1357).

Note: PyMatting is installed but only its **foreground estimators** are used. No
alpha estimator (`estimate_alpha_cf`) is wired in anywhere today.

Roughly a dozen endpoints call `_get_mask` or `_acquire_mask`: `/remove`,
`/remove-url`, `/mask`, `/replace-bg`, `/smart-crop`, `/studio-shot`, `/sticker`,
`/outline`, `/silhouette`, `/shadow`, `/psd`, `/headshot`, `/video/remove`, and
others. Only `/remove`, `/remove-url`, `/smart-crop`, `/studio-shot` currently
expose the `detect` / `decontaminate` params.

## 3. The problem

A channel partner's client photographs **packaging film (tissue paper sheets)**
lying flat inside a **white lightbox**. Product and background wall are at nearly
identical brightness. Source photos are 3072x4080 (12.5 MP).

On these shots our cutout leaves a **wavy white halo along the product edge**,
10-20 px wide at 1280 px height (so ~30-60 px at full res). Photoroom, on the
same source files, produces a dead-straight clean edge. The client previously used
Photoroom desktop and was satisfied.

Measured on the client's own originals: the alpha boundary overruns the physical
film edge by **9-46 px**, and the overrun width is *consistent row to row*
(e.g. median 40 px, p90 44 px on one film). So the mask is including a strip of the
white base sheet the film rests on. This is a **boundary-localization error**, not
colour contamination.

Why the existing knob does not solve it: `decontaminate` reassigns colour in a band
around the edge. On **coloured** film there is a colour difference to exploit and it
helps. On **white film on a white wall** there is no colour signal, so it erodes into
the product instead, producing pinholes and a serrated edge. There is no single
setting that covers both cases.

## 4. What we have already tried, with results

All tested on the client's real full-resolution photos (7 films: white, sage,
tan, salmon, pale pink, dusty rose, plus a beige) and on our existing eval set
(loose hair, foliage, thin jewellery chain, shoe, dog, portrait).

### 4a. Model swaps (single L4 GPU run each, full res)

| Model | License | Result on the flat-lays |
| :--- | :--- | :--- |
| BiRefNet (current) | MIT | The halo. Baseline. |
| `ZhengPeng7/BiRefNet_HR-matting` (2048 inference) | MIT | **Unusable here.** Returns the film as semi-transparent (alpha mean ~30-60 where the film covers ~40% of frame). Physically defensible for thin tissue, commercially wrong. Verified in fp32 — not an inference bug. |
| `PramaLLC/BEN2` (94.6M params, confidence-guided matting refiner) | MIT | Better than baseline on white film, but still leaves a white band on coloured film and includes a lightbox seam strip as product. Worse overall. |
| `feyninc/FeyNobg` (262.8M params, BiRefNet architecture retrained) | Apache-2.0 | **Best of the four.** Halo essentially gone on coloured films; tight straight edge. Still leaves a thin light rim on the pure white film. |

FeyNobg also **passed a complex-edge regression**: foreground coverage within
0.9% of current production on loose hair, foliage, thin chain, shoe and dog, with
no visible silhouette damage. (Coverage %, ours vs FeyNobg: hair 53.6/54.4,
foliage 39.2/39.3, chain 2.9/2.5, shoe 16.0/16.0, dog 48.6/48.5.)

We did NOT evaluate `briaai/RMBG-2.0` because its weights are CC BY-NC
(non-commercial) and this is a commercial product.

### 4b. Post-processing on top of the model alpha (all local CPU, no GPU)

1. **Our existing guided filter** applied to FeyNobg's alpha: helped one film,
   left the other six essentially unchanged. Not sufficient alone.
2. **Smooth + shrink** (blur mask, re-threshold inward, feather): removes the
   wobble, but a residual sliver survives wherever the halo is wider than the
   shrink distance. Blunt instrument.
3. **Global colour snap** (reassign a band by distance to global fg/bg median
   colours): FAILED — the global background median was polluted by the dark
   lightbox floor, so wall-white was classified as foreground.
4. **Local colour snap** (normalised-convolution local colour models, foreground
   sampled from deep interior, remove-only): close to clean, but left a patch
   where a wall shadow gradient made the local distance ratio ambiguous.
5. **`boundary_refine` v1** (per-row snap to strongest luminance gradient within a
   search window): produced visible step artifacts — per-row independence with no
   smoothness coupling.
6. **`boundary_refine` v2** (current best): for each boundary line, sample the
   *local* product colour from ~160 px inside the mask, walk inward until the
   pixel matches that colour, median-filter the resulting offsets along the
   contour, fit a quadratic trend, accept only if mean residual < 9 px, follow the
   fitted trend, remove-only, capped at 70 px, feather 1 px.
   **Result on the films:** sliver removed on pale pink and salmon, much reduced
   on sage, no change on pure white film (nothing to snap to). Mean "near-white
   pixels in the inner edge band" across the 7 films dropped from 0.059 to 0.041.

   **This prototype is NOT safe and must not ship as written.** An external
   reviewer predicted it would destroy legitimate product borders; we tested that
   and confirmed something worse. Alpha mass lost when run over our eval set:

   | Case | Alpha mass lost | Gate residuals |
   | :--- | :--- | :--- |
   | Thin jewellery chain | **38.9%** | 8.2 / 7.1 / 12.9 / 2.7 — passed the gate |
   | Shoe | 0.01% | rejected (residuals 15-19) |
   | Loose hair | 0.02% | rejected |
   | Dog | 0.02% | rejected |
   | Foliage | 0.02% | rejected |

   The chain is visibly thinned along its whole length. The failure is structural:
   the smoothness gate measures how *smoothly the offsets vary*, and for a thin
   object the per-row offsets are small and smooth, so the gate accepts — then the
   inward walk eats most of the object. The complex-edge regression reported in
   section 4a validated **raw FeyNobg only**, not this refiner; those are separate
   release decisions and we conflated them.

   Concrete design faults identified: it scans complete rows and columns instead of
   extracting connected contours, and it fits a single global quadratic per side
   rather than gating per segment. It is therefore not segment-gated in any
   meaningful sense, despite the description.

### 4c. Advice we received and rejected, with reasons

- *"You're applying a hard binary threshold; enable alpha matting / rembg's
  alpha-matting flag."* — We already ship soft alpha and already run PyMatting.
- *"Erode the mask 1-2 px and Gaussian blur it."* — The halo is 30-60 px at full
  res and wavy, not a uniform 1-2 px anti-aliasing fringe. Eroding far enough to
  eat it also eats product.
- *"1 px anti-aliasing blend explains the halo."* — It does not explain a 40 px
  consistent overrun.

## 5. What a reviewer told us to do (not yet built)

An independent review recommended a **geometry-gated, inward-only contour
optimiser** rather than colour clustering: split the contour into simple vs
complex segments by curvature and line-fit residual; on simple segments only,
solve for one coherent boundary path via dynamic programming / shortest path with
an image-gradient term measured normal to the contour plus smoothness and
stay-near-current terms; RANSAC line fitting for overwhelmingly straight runs;
inward-only, displacement-capped, and a no-op when confidence is weak; hair,
foliage, chains, holes and translucent regions never touched. Our v2 above is a
simplified version of this (colour data term, quadratic trend instead of DP).

The same review flagged that a wide erode/dilate trimap plus PyMatting
`estimate_alpha_cf` is a cheap benchmark worth running, but that alpha matting is
mathematically ill-posed where foreground and background are visually
indistinguishable — i.e. it will not save the white-on-white case.

## 5b. Images accompanying this brief

Three review panels are provided. Each is a single JPEG, 2318x794, four columns:

```
1. ORIGINAL PHOTO | 2. FEYNOBG (raw model) | 3. FEYNOBG + BOUNDARY REFINE | 4. ALPHA MASK
```

Columns 2 and 3 are composited on **magenta (#FF00C8)**. That colour is
deliberate: retained white background pixels are unmistakable against it. A
checkerboard backdrop hides this exact defect (white halo on white checks), and
black loses the film's own edge. Column 4 is the raw model alpha with no
post-processing — it is not what we ship.

| Panel | Why it is included |
| :--- | :--- |
| `panel-palepink.jpg` | Best case. A 40 px overrun into the white base sheet, removed by the prototype. |
| `panel-white.jpg` | **Unsolved case.** White film on a white wall. Columns 2 and 3 look nearly identical because there is no colour signal to snap to. |
| `panel-sage.jpg` | Honest middle. Improved but not clean; a teal-tinged fringe survives where the snap stopped short. |

Four more films (`blush`, `beige`, `rose`, `tan`) exist and can be supplied on
request; they repeat the same three outcomes.

Full-resolution source files, our current production output, the FeyNobg
outputs, both alpha masks, and the prototype implementation itself all live in
`eval/cases/kravento-film/` in this repository (see its README). The
`boundary_refine` prototype is `boundary_refine_prototype.py` there — please
read the code rather than trusting the description in section 4b.

## 5c. Ideas already received from reviewers

Recorded so later reviewers do not repeat them.

**Accepted / under evaluation:**

- Ship FeyNobg as an explicit opt-in engine (e.g. `engine=product-v1`) covering
  background-removal and mask endpoints only, with `standard` unchanged. No
  silent auto-routing, because unpredictable output drift makes customer
  complaints impossible to reproduce.
- Load the second model in a **separate narrow Modal class/container** rather than
  alongside everything in the current production container, to isolate cold
  starts, VRAM and rollbacks.
- **Empty-lightbox reference plate.** The client's camera setup is fixed, so have
  them shoot one empty lightbox frame; register it to each product photo, compute
  colour/gradient/texture differences, and use that as a prior alongside the model
  mask, optimising only a narrow boundary band. This supplies information a single
  image genuinely does not contain, which is the crux of the white-on-white case.
  Exposed as a reusable `background_reference` / capture profile, not per-customer
  code. **We consider this the strongest idea received so far.**
- Direct **erase / restore** editing that deterministically overrides alpha, in
  addition to the hint map — hints as suggestions to a matting solver are weaker
  than hard user overrides.
- SAM 2 (Apache-2.0) as an optional interactive model for propagating sparse
  strokes, loaded lazily and never in the automatic path.
- A release gate built on **hand-corrected full-resolution ground-truth masks** for
  all seven films, with hard negatives: real dark borders, patterned edges, labels
  touching the boundary, multiple objects, curved products, glass, hair, foliage.
  Measure boundary displacement, halo pixels, removed alpha mass, topology, thin
  feature retention, latency, VRAM, cold start.

**Rejected:**

- **Routing hard cases to PhotoRoom's API.** Non-starter on two grounds: PhotoRoom
  is a direct competitor and this product exists to compete with them, and it
  would ship a customer's unreleased product photography to that competitor.
  Please do not propose reselling or proxying a competitor's API. Suggestions that
  improve our own pipeline are what we are looking for.

## 6. Open questions we want opinions on

1. **Better model?** Is there a background-removal or matting model with a
   commercially usable licence (MIT / Apache / BSD) that handles low-contrast
   product-on-white better than FeyNobg? We know of BiRefNet variants, BEN2,
   RMBG (non-commercial), and the Mask-Guided-Matting family. What are we missing
   as of late 2026?
2. **Better boundary algorithm?** Is the DP/shortest-path contour optimiser the
   right call, or is there a stronger standard approach for "snap a coarse mask to
   a straight, low-contrast product edge"? Anything in classical CV (active
   contours, graph cuts with shape priors, edge-linking) or recent learned
   refinement work that fits better?
3. **White-on-white with no colour signal.** When foreground and background are
   genuinely indistinguishable locally, what does a production system do? Is a
   learned domain prior the only answer, or is there a principled geometric one?
   How does Photoroom plausibly do it?
4. **User-supplied hint.** The client's own suggestion: a pencil tool to mark
   background, like Photoroom desktop. Our sketch is an optional second file on
   `/remove` — an RGBA scribble map where alpha 0 = no hint, opaque black =
   definite background, opaque white = definite foreground — merged into a trimap
   and solved with an alpha estimator in the unknown region only. Because their
   lightbox camera is fixed, one rough hint could be reused across an entire batch.
   Is that the right interface? Better encodings?
5. **Scope of any model change.** If FeyNobg replaces or supplements BiRefNet,
   should that apply to every mask-consuming endpoint, or only the
   background-removal ones? We currently have evidence only for plain cutout
   quality; we have NOT evaluated the difference on `/studio-shot`, `/sticker`,
   `/outline`, `/silhouette`, `/shadow`, `/psd`, `/headshot`, or `/video/remove`.
   How would you scope and stage that rollout?
6. **Two models or one?** Is running two models (route by image characteristics,
   or let the caller pick) a reasonable production design, or an operational
   mistake — double the weights, double the cold-start surface, double the
   regression matrix?

## 7. Constraints on any answer

- Commercial use. Non-commercial licences (e.g. CC BY-NC) are unusable.
- Runs on Modal, single L4 GPU per container, cold starts already 120-180 s;
  we cannot afford a large increase in model load time.
- Full resolution matters: customers get 3072x4080 output today, and any solution
  must not cap that.
- Backward compatibility is a hard rule: existing callers' output must not change.
  New behaviour ships default-off behind an explicit parameter.
- No hand-tuning per customer. Whatever ships has to be safe on hair, fur,
  foliage, jewellery and translucent edges, or must provably no-op on them.
