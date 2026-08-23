# Kravento lightbox film case

Client of our channel partner (Kravento) photographs packaging film — tissue
paper sheets — lying flat inside a white lightbox. Product and background wall
sit at nearly identical brightness. This is the case behind the white-halo
problem described in `docs/superpowers/specs/2026-08-17-white-halo-problem.md`.

All images are the client's own full-resolution originals, 3072x4080.

## Files

| Suffix | What it is |
| :--- | :--- |
| `<film>-original.jpg` | Untouched client photo, straight from the camera |
| `<film>-current-prod.jpg` | Current production output (BiRefNet + guided filter, default params), composited on teal |
| `<film>-feynobg.jpg` | `feyninc/FeyNobg` raw model alpha, no post-processing at all, composited on teal |
| `<film>-feynobg-refined.jpg` | FeyNobg + prototype `boundary_refine` v2 (product-colour snap, inward-only, trend-gated) |
| `<film>-feynobg-alpha.png` | The raw FeyNobg alpha mask, grayscale, full res |

Films: `blush`, `sage`, `tan`, `beige`, `palepink`, `rose`, `white`.

Teal (`#00AAA5`) is used as the composite background because that is what the
partner used in the comparison screenshots they sent us — it makes any retained
white background pixels obvious.

## What to look at

The defect is a **white halo along the product edge**, worst on the left edges.
Compare `-current-prod` against `-feynobg` on `tan` and `white`, then look at
`-feynobg` vs `-feynobg-refined` on `palepink`, `beige` and `sage` — that pair
shows what the boundary-refinement prototype removes.

`white` is the unsolved case: with a white film on a white wall there is no
colour signal to snap to, and every automatic approach we have tested leaves a
faint rim.

Note on `tan-feynobg`: the soft bottom lip is **not** a cutout defect. The front
edge of the film curls up off the table and is genuinely out of focus in the
source photograph.

## Coverage caveat

`-current-prod` exists only for `tan` and `white` so far. The other five would
each need a fresh production API call to generate. Every other file is present
for all seven films.

Generated 2026-08-17. Nothing here is deployed; `boundary_refine` is a local
prototype, and no model has been swapped in production.
