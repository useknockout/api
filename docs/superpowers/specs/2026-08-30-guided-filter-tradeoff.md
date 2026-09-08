# Guided-filter refinement: a real trade, not a bug

**Status:** investigated, nothing shipped. Prod unchanged. Dev has an experiment knob.
**Date:** 2026-08-30/31
**Trigger:** Troy compared a bike and a cocktail glass against April 2026 outputs and
found today's results visibly worse.

---

## The one-paragraph version

`_refine_alpha` — a guided filter, radius 8 — runs on every default-engine mask.
It was added 2026-07-02 in `78f97ff` (v0.9.0) and did not exist in April. On
subjects with **thin structures** (bike spokes, glass rims, jewelry chain) it
smears them into fog. On subjects with **low-contrast product edges** (Kravento's
sage film on a lightbox) it *reduces* white halo. So it helps the exact defect
that started the product-v1 engine track, and hurts fine detail. Turning it off
globally would trade one customer complaint for another.

## Evidence

Guided filter OFF reproduces April exactly. Same crop, same scale, bike rear wheel:
April and `refine=off` are the same image (mean alpha difference **0.009**, RGB
differing ~1 level in 255). Today's prod output is the fogged one.

Bike, whole image:

| mode | foreground | soft-edge px |
| :--- | ---: | ---: |
| standard (prod today) | 12.82% | **14.37%** |
| low (radius 4) | 12.92% | 9.08% |
| off | **13.48%** | **4.03%** |

Foreground *increases* with the filter off — it was erasing thin structure, not
merely softening it.

Five-image sweep (all `engine=standard`, dev):

| image | soft% standard -> off | halo standard -> off |
| :--- | :--- | :--- |
| glass | 1.84 -> 0.74 (April 0.64) | 0.0031 -> 0.0032 |
| palepink film | 0.80 -> 0.54 | 0.0632 -> 0.0624 |
| hair | 8.52 -> 8.00 | 0 -> 0 |
| chain | 2.06 -> 1.12 | 0.1901 -> **0.2131** |
| flower | 2.47 -> 1.80 | 0.0012 -> 0.0016 |

Foreground area was unchanged on all five.

## The counterexample that stopped the change

**`sage` — a real Kravento film — gets WORSE with the filter off:**

| sage | foreground | soft% | halo |
| :--- | ---: | ---: | ---: |
| prod behaviour | 40.82% | 0.88% | **0.1568** |
| refine=off | 40.84% | 0.59% | **0.1698** |

~8% more white fringe, on the one metric the paying customer cares about.

An earlier "six for six, off is better everywhere" conclusion was WRONG: it used
`palepink` as the film case, and palepink escalates to product-v1, so it never
exercised the standard path at all. Sage is the correct standard-path film test.

## Why Kravento is mostly insulated

Escalation measured on prod, `engine=auto`, their real images:

- beige, blush, palepink, rose, tan, white -> **product-v1**
- sage -> **standard**

`ProductEngine.cutout` (S3OD + CascadePSP) contains no guided filter, so 6 of 7
films are structurally immune to any refine change. Sage is not.

## Options (none taken)

1. **Leave prod as-is.** Zero risk. Thin-structure subjects stay degraded.
2. **Expose `refine` as an opt-in param**, default unchanged. Nobody's existing
   output moves; anyone shooting spokes/glassware/chains can turn it off.
   Currently built on dev for `/remove` only (`standard` | `low` | `off`),
   uncommitted, undocumented.
3. **Make it automatic** from the same shape signal `engine=auto` already
   computes (perimeter^2/area, thin_loss): skip the filter on thin/complex
   subjects, keep it on flat low-contrast ones. Best outcome, needs its own eval.

Recommendation: 2 now, 3 later. Do not flip the default.

## Separate finding: `edge=hard` is destructive on its own

With the filter already off, `edge=hard` still shreds bike spokes (foreground
13.48% -> 10.61%). `_harden_alpha` clips mid-alpha to zero then erodes 1px, and
thin structures live in exactly that band. The workspace help text currently
says "Hard gives a cleaner cut on products", which walks users shooting glass,
bottles, ice, or spoked wheels straight into it. Needs its own fix and its own
copy change.

## Untested hypothesis worth keeping

A filter that spreads alpha into a soft ramp across background pixels is a
mechanism for *producing* white halo. The sage number cuts against this (halo
got worse without it), so if anything the filter is fighting halo rather than
causing it — but the interaction between the filter, `_clean_foreground`, and
the halo metric has not been isolated.

## Repro

```
# dev only; param exists on /remove, values: standard | low | off
curl -sL -o out.png -X POST "https://useknockout-dev--api.modal.run/remove" \
  -H "Authorization: Bearer $API_TOKEN" \
  -F "file=@eval/cases/kravento-film/sage-original.jpg" \
  -F "engine=standard" -F "refine=off"
```

Test images used: `eval/cases/kravento-film/*`, `eval/cases/model-loose-hair/`,
`eval/cases/jewelry-thin-chain/`, `eval/cases/flower-pale-leaves/`, the bike and
glass in `OneDrive/Desktop/2027 - Deployed Sites Brand Guide/useknockout/Remove
background/` (`bike.jpg`, `drinks.jpg`, April references `test-finedetail.png`
and `test-glass.png`).
