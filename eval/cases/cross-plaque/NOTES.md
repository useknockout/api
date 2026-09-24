# cross-plaque

Reported by Tomek via Telegram, 2026-09-18. Prometheus memorial plaque: matte
black plate, cross-shaped CUTOUT (white background shows through it), gold rose
applique, gold/black fretwork book behind. Shot on white. Native 1396x1974.

## The failure

`/remove` keeps ONLY the cross and the gold rose. The entire black plate and the
book are dropped as background.

The cross is not even an object: it is negative space, a hole with the white
backdrop showing through. The model keeps those background pixels because the
shape reads as a figure, and discards the actual subject around them.

Cause is salient-object detection doing what it is built to do. The bright cross
and gold rose are high-contrast and read as salient; the plate is large, matte,
dark, low-contrast against white, and bleeds off two frame edges.

## Reproduces on EVERY path

Measured as share of pixels with alpha > 200 (genuinely solid subject):

| variant | alpha>200 |
| :--- | ---: |
| full token, native 1974 | 5.6% |
| demo token, native | 5.6% |
| demo token, 1536 WebP (exact playground path) | 5.7% |
| full token, 1536 | 5.5% |
| full token, 1024 | 5.6% |
| engine=product-v1, native | 5.6% |

Six paths, one number. ~5.6% == cross + rose. No salient-object configuration
retains the plate.

## Fix: engine=backdrop (2026-09-19)

Flat-colour key, zero GPU. Backdrop colour = median of the frame border;
alpha = distance from it on a 12..40 ramp; enclosed transparent islands under
0.2% of the frame are refilled (specular flecks on the gold), larger ones stay
open (the cross); then the same guided-filter refine as every other engine.
Refuses with 400 when the border is not a flat colour.

Result on this image: 70.4% solid, plate + book + rose kept, cross hole and
fretwork gaps transparent, rose speculars intact. `ours-backdrop.png`.
Opt-in only; the standard path is untouched (byte-identical before/after).

Regression run 2026-09-20, prod vs dev standard output, md5: kravento-film
blush/sage/tan/beige/palepink/rose/white, plant-foliage, flowerbox-1,
flower-pale-leaves. All ten IDENTICAL.

## Update 2026-09-24: engine=auto does this itself

`auto` now detects the plaque mistake (default mask dropped a large area of
pixels far from a flat backdrop colour: 57.6% of frame, 95% of strong pixels)
and switches to the backdrop key. Fires on this image only across 25 eval
images. Customers on engine=auto get the fix with no new parameter. Deployed
and verified on prod 2026-09-24 (remove / studio-shot / smart-crop).

## Where NOT to use it

flower-pale-leaves passes the flatness guard (white backdrop) and comes out
worse: cast shadows under each bloom are kept as solid grey subject, because
a colour key cannot tell shadow from petal. Standard 36.1% solid, backdrop
46.4%, the difference is shadow. See
eval/cases/flower-pale-leaves/backdrop-engine-worse.png. Do not recommend
backdrop for the film sheets or soft-lit organics; it is for hard-edged
flat-lit products with real holes.

`evidence-all-variants-magenta.png` is all six composited over magenta so the
transparent region is unambiguous. They are visually identical.

## Method warning (cost a lot of time on 2026-09-18)

Do NOT judge these outputs by viewing the RGBA PNG over a white background. The
dropped plate leaves a 4-6% partial-alpha ghost that renders as a convincing
solid dark plate over white, and produced a completely fabricated conclusion
that resolution controlled the outcome (it does not - see the table above).
Composite over a saturated colour, or measure alpha directly. Opaque coverage
for a retained plate would be 40-50%, not 5.6%.

## Also broken here, separately

Interior holes. Even if the plate were retained, correct output needs a
transparent cross-shaped hole plus transparent gaps through the book fretwork.
Masks currently fill interior holes solid. Same class as jewelry-thin-chain.

## Ruled out

- Playground path. Caps input at 1536, re-encodes WebP q0.92
  (Playground.tsx:9-63). Replicated exactly; same failure.
- Demo output cap. DEMO_MAX_DIM runs AFTER _remove (main.py:3074-3077), cannot
  affect the mask. Demo and full-token masks are identical.
- Input resolution. No threshold exists.
- Cold start. The 119997ms in the user screenshot was GPU cold start
  (route.ts maxDuration=180); warm runs fail identically.
- engine/detect/decontaminate. product-v1 fails the same;
  detect/decontaminate are ignored on that path by design (main.py:1860).
