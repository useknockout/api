# Knockout Layers: a free, client-side canvas on top of one cutout

**Status:** spec v2, approved in direction by Troy 2026-09-04, reviewed by Codex the same
day (`useknockout-api/.agent-handoff/013-from-codex.md`; its corrections are folded in
below). Build owner: web-app session (portal repo).
**API changes required: none.** Kravento's path is untouched by construction.
**Reference implementation:** `docs/prototypes/knockout-layers.template.html` in this repo.
The live one Troy approved: https://claude.ai/code/artifact/feaff5d5-c8d7-4675-97bd-da12d2330d05
**The prototype is a sketch of the shape, not code to copy.** Section 12 lists what in it
must not ship.

---

## 1. Why

Supabase `public.usage`, snapshot 2026-09-04 19:32 UTC (Codex reproduced; a query
needs its UTC cutoff recorded or the numbers drift within a day):

| endpoint | rows | distinct users |
| :--- | ---: | ---: |
| /remove | 3,812 | 38 |
| /studio-shot | 2,780 | 6 |
| /smart-crop | 853 | 5 |
| /silhouette | 32 | 2 |
| /sticker | 25 | 3 |
| /shadow | 16 | 4 |
| /outline | 14 | 2 |

Top three = 93.4% of all 7,971 rows. The four styling endpoints total 87, and 72 of
those landed in one ten-minute window on 2026-09-03 (17:45 to 17:52 UTC) from two
users, which reads as a test burst rather than organic use; before that day they
totalled 15. These are logged rows, not billable units, and public-beta calls are
not logged at all (`main.py:1059`).

Each styling endpoint acquires its own mask (`main.py` /shadow 3360, /sticker 3404,
/outline 3439, /silhouette 3485), so a user who removes a background and then adds a
shadow pays two inferences for one mask. Client-side effects on the cutout already in
the browser avoid the second one. Honest cost claim: **"no additional inference
charge for editing."** Saves, asset storage and fonts still use network and storage.

The average person gets one reason to open the workspace today: "I have a photo to
cut out." Knockout Layers gives them a canvas to keep playing on at no inference
cost, and turns Kravento's scripted pipeline (one look, many SKUs) into something
anyone can do.

Neither Ditther nor Unicorn Studio can do the one thing this is built on: their
effects apply to a flat image. Ours are **mask-aware**. Halftone the background,
keep the shoe sharp.

## 2. Cost model

| step | inference cost |
| :--- | :--- |
| `/remove` -> cutout PNG | 1 GPU call. The only one. |
| background: original photo / colour / gradient / transparent | 0 |
| every effect, on any layer | 0 |
| shadow, outline, silhouette | 0 (today: paid server calls) |
| move, rotate, flip, resize, text, fonts | 0 |
| collage of N products | N cutout calls total, then 0 |

The API endpoints for shadow/sticker/outline/silhouette stay for SDK users. Nothing
is removed.

## 3. Where it lives in the workspace

The workspace today is one mode, **Batch** (many photos, one tool). Keep it; it is
Kravento's job and it works. Add a second mode, **Canvas** (one composition, many
layers), on the same dark three-column skeleton: layer stack left (replaces the tool
list in this mode), interactive stage centre, properties for the selected layer
right, add-product / templates strip at the bottom where the queue rail is.

Entry points, in priority order:
1. **"Open in Canvas"** on a finished Batch tile, shown only when an alpha-capable
   cutout exists (PNG/WebP with alpha; a flattened JPG or studio composite cannot
   reconstruct the matte). Zero-friction: the cutout is already in memory. Hold a
   Blob reference independent of the Batch preview URL's disposal.
2. **Logged-out landing = Canvas with sample cutouts preloaded.** Play forever on
   samples at zero inference; dropping your own photo spends the demo allowance.
   Troy leans toward this being the forefront of the web app: run it on the
   logged-out state first, measure sample-to-own-upload and successful exports,
   then decide about the signed-in default.
3. Mode switch `Batch | Canvas` in the TopBar. Keep an obvious **Upload photo**
   action in Canvas. Switching modes preserves the in-memory Batch queue.

First-run tour, once, dismissible, replayable from Help, never blocking the first
drag, tolerant of unavailable localStorage. Copy (Codex, no em dashes):
1. "Move your cutout. Drag it anywhere on the canvas."
2. "Style the background. Keep your subject sharp."
3. "Make it yours. Add text and choose a font."
4. "Reuse your look. Save a template for your next photo." Logged out: append
   "Sign in to save." Do not promise batch reuse to free users in the tour; disclose
   it as paid where they meet it.

## 4. Document model

Ordered top -> bottom. Background is always last and cannot be deleted.

### Assets and slots (the part the prototype does not have)
- **Asset manifest**: durable asset id, kind (`original` | `cutout`), the
  original-to-cutout relationship, native width/height after EXIF orientation
  normalisation, cutout alpha bounds, content hash, and a durable storage reference.
  Never persist `blob:` URLs or expiring signed URLs as identity; refresh signed URLs
  on load. Store both original and cutout bytes so reopening never reruns `/remove`.
  Local drafts may use IndexedDB; account saves persist assets alongside the JSON.
- **Slots and bindings**: layers reference a `slotId`, not an asset. A `bindings`
  map resolves slot -> asset pair. Two layers may share one slot (the poster template
  stacks a silhouette under a dithered copy of the same bike: one slot, one cutout).
  The background may reference `{kind:"slot-original", slotId}` or a fixed design
  asset, a colour, a gradient, or `transparent`.
- **Templates** = the document with `bindings` emptied and slots kept. v1 batch
  application supports **one input slot**, possibly referenced by several layers.
  Multi-input collage stays an interactive document until a mapping model is
  defined; a batch never silently binds only the first layer.
- **Geometry**: canvas logical `w`/`h` in document pixels; each layer has an anchor
  (centre), `x`/`y` in document pixels, `rot` in radians, uniform `scale`, flips.
  Subjects carry a **slot frame** with `fit: contain | cover` and an anchor, and fit
  the cutout's **alpha bounds**, not its transparent padding, so a look built on a
  wide shoe places a narrow chain where the designer intended.
- **Rendering params that must be persisted**: sRGB throughout; effect id +
  version + typed params; a seed for `grain` and `glitch`; font family, weight,
  style and asset/version; text alignment, line height and wrapping box; layer
  name, lock, opacity if exposed. `version` exists; define migration and reject
  unknown versions.
- **Export never changes composition**: all document-space distances scale
  together at export (blur, shadow, stroke, halftone cell, glyph cell). Resizing
  the document is a different operation from choosing a smaller export.
- **Persistence envelope** (portal-owned, not image-API): owner, revision with
  optimistic checks, storage ownership, and one definition of the free limit
  covering both documents and templates, enforced atomically.

```jsonc
{
  "version": 2,
  "canvas": { "w": 2000, "h": 1400 },
  "assets": { "a1": { "kind":"original", "w":3072, "h":3072, "hash":"…", "ref":"…" },
              "a2": { "kind":"cutout", "of":"a1", "w":3072, "h":3072, "alphaBounds":[420,610,2610,2500], "hash":"…", "ref":"…" } },
  "bindings": { "product": { "original":"a1", "cutout":"a2" } },
  "background": { "source": { "kind":"slot-original", "slotId":"product" },
                  "fx": { "id":"halftone", "v":1, "cell":7, "ink":"#0f1a14", "paper":"#57C985" } },
  "layers": [
    { "id":"t1", "type":"text", "on":true, "value":"NEW SEASON",
      "font": { "family":"Bebas Neue", "weight":400, "style":"normal", "asset":"gf:bebas-neue@1" },
      "size":132, "color":"#E6EAED", "align":"center", "lineHeight":1.05, "box":{"w":900},
      "x":540, "y":448, "rot":-0.06, "scale":1, "flipX":false },
    { "id":"s1", "type":"subject", "on":true, "slotId":"product",
      "frame": { "w":900, "h":700, "fit":"contain", "anchor":"center" },
      "x":1240, "y":770, "scale":1.05, "rot":-0.12, "flipX":false, "flipY":false,
      "fx": { "id":"none" },
      "shadow": { "on":true, "dx":22, "dy":30, "blur":30, "opacity":0.5 },
      "outline": { "on":false, "width":8, "color":"#ffffff" } }
  ]
}
```

Shadow and outline are properties of a subject, not separate layers.

## 5. Effects: ten, in two groups, typed parameters

Cap at ten (Troy's call). "None" is not an effect. One effect per layer in v1.
Every effect declares its **alpha contract** (section 12 explains why):
- *colour-only* effects (duotone, posterize, grain, glitch, dither) transform RGB and
  **preserve source alpha exactly once**;
- *cell/glyph* effects (halftone, blocks, ascii) may paint outside the source
  silhouette, so their coverage is **intersected with source alpha**;
- shadow and outline intentionally extend past the subject and are composed
  separately.
Background effects render where the background is; subject effects where alpha > 0.

**Photo**
| id | definition | params |
| :--- | :--- | :--- |
| `blur` | gaussian blur; "portrait mode" on the background | radius (document px) |
| `duotone` | luminance -> two-colour ramp | colourA, colourB, strength |
| `grain` | film grain + mild posterize | intensity, seed |
| `posterize` | quantise to N levels | levels (int) |
| `halftone` | dot grid, radius from luminance | cell (document px), ink, paper |

**Graphic**
| id | definition | params |
| :--- | :--- | :--- |
| `dither` | ordered Bayer 4x4 to two colours | cell, ink, paper |
| `blocks` | block average; label "Blocks" | block (document px), `studs: bool` (LEGO look) |
| `ascii` | luminance -> character ramp, mono font | cell, ink |
| `silhouette` | fill alpha with one colour; subject only | colour |
| `glitch` | RGB channel offset + horizontal slice shift | displacement, seed |

Expose only applicable controls per effect. Do not force a single `amount` slider.

## 6. Transforms and interaction

- Drag to move. **Pixel-accurate hit test** on subjects with a mouse (sample cutout
  alpha at the pointer). On touch, use screen-space tolerance so thin chains are
  grabbable, and allow selection from the layer list.
- Rotate handle above the box; Shift snaps to 15 degrees; an on-screen snap/angle
  control for touch (no Shift there).
- Corner handle scales uniformly. **Handle hit regions are sized in CSS pixels
  (~44px)** independent of zoom; visible handles may stay small.
- Track the active `pointerId`; clear gestures on `pointercancel` and
  `lostpointercapture`; define second-finger behaviour (ignore or pinch-zoom).
- Flip H / Flip V / Duplicate / Remove in properties. Arrow keys nudge 1px, Shift
  10px, Delete removes. Shortcuts scoped to the editor; ignore input, textarea,
  select and contenteditable; `preventDefault` only for handled keys.
- **Undo/redo** before destructive template replacement and layer removal.
- Selection chrome lives on a separate overlay canvas, never on the export surface.
- Mobile (<= 980px): stage first; Layers and Properties as drawers/sheets.
- Accessible layer selection and reordering; undo/redo reachable without a mouse.

## 7. Export

- Shared pure renderer for preview and export. Export re-renders once at the
  document's export size; preview at fit-to-viewport.
- Formats: PNG (alpha preserved when the background is `transparent`), WebP, JPEG
  (flattened against an explicit matte the user chose). `toBlob` can return null or a
  different MIME type than requested; handle both.
- Long-side caps, aspect preserved: demo 1536, free 2048, paid 4096.
- **Watermark on demo and free exports**, drawn last, never stored in the document.
  Minimum readable size, not a percentage: start at 120 to 180 output px wide at
  1536 to 2048, proportional safe padding, a contrast backing chosen from the image.
  Show placement in the export preview. Inspect real mobile exports before fixing
  the numbers.
- These caps and the watermark are UI monetisation conventions, not tamper-proof
  enforcement: the pixels and the renderer are already in the browser. Accept that
  for a zero-inference feature. Saved-document ownership and quota are enforced
  portal-side.

## 8. Performance

- Start with a **Worker + OffscreenCanvas** effect pipeline, feature-detected with a
  responsive fallback. Interaction and selection stay on the main thread; send
  immutable render jobs with revision ids and discard stale results.
- WebGL is a measured optimisation for expensive kernels, not a prerequisite. ASCII
  needs a glyph atlas or worker-side font rendering either way.
- Preview buffers at preview resolution; the prototype runs subject effects at
  cutout native size even for a 1000px stage. Export uses full-quality assets.
- Cache keys include asset revision, renderer/effect version, all params, font,
  seed and resolution. Use a byte budget, not an entry count; release ImageBitmaps
  and textures. One 4096^2 RGBA buffer is 64 MiB. Serial batch export with
  cancellation. No full native-size RGBA copies solely for hit testing.

## 9. Tier rules

| capability | demo (logged out) | free | payg | plus |
| :--- | :--- | :--- | :--- | :--- |
| canvas with sample cutouts | yes | yes | yes | yes |
| own photo -> cutout | **10/day per IP** (`DEMO_IP_DAILY_CAP_DEFAULT`, env override) | 30/mo | metered | metered |
| all ten effects, transforms, text, fonts | yes | yes | yes | yes |
| export | 1536, watermark | 2048, watermark | 4096, clean | 4096, clean |
| save documents + templates (one shared limit) | no | 3 | unlimited | unlimited |
| apply template to a batch | no | no | yes | yes |

Confirm the deployed demo cap before publishing quota copy; the source default is
10 and the environment can override it.

## 10. Telemetry

Client events, no image data: `canvas_opened{entry}`, `sample_to_own_upload`,
`effect_applied{id, layer}`, `template_saved`, `template_applied{n}`,
`export{format, size, watermarked, success}`, `tour_step{n, dismissed}`.

## 11. Non-goals

Video; interactive or animated scenes; 3D; shader authoring (Unicorn's product).
Any change to `/remove`, `/studio-shot`, `/smart-crop`, `engine=auto`, the SDKs, or
server presets (keep optional preset work out of this). Replacing Batch mode.

## 12. Prototype caveats: do not copy these into production

The prototype at `docs/prototypes/knockout-layers.template.html` demonstrates the
shape. Codex's source review found four things in it that must not ship:
1. **Alpha leakage in cell effects.** `fxPixel`, `fxHalftone` and `fxAscii` paint
   whole blocks, cells and glyphs using averaged alpha, so they can draw where the
   source alpha is zero and lose thin chain and hair coverage. Production
   intersects coverage with source alpha (section 5). Do not multiply alpha twice
   for effects that already preserve it, or soft edges become alpha-squared.
2. **Photo background retains the original subject.** The prototype paints the full
   original photo, then the cutout on top; moving the cutout reveals the subject
   underneath. `/remove` does not supply a clean plate. Treat "original photo" as an
   explicit design option aligned with the cutout, and use colour, gradient,
   transparent or an independent asset for freely movable compositions.
3. **Text is interpolated into `innerHTML`.** A saved value containing markup would
   execute. Production uses text nodes, validated document fields and safe font and
   asset ids.
4. **Selection chrome is drawn on the render canvas.** Keep it on an overlay.

## 13. Acceptance checks (from Codex, adopted)

Save and reopen with original and cutout assets after reload; one-slot template with
duplicated layers; narrow-chain / wide-shoe template fitting; alpha preservation at
hair and chain edges for every effect; moving a subject over a photo background;
1536 / 2048 / 4096 preview-to-export geometry parity; font readiness before render;
transparent and JPEG export; no selection chrome in export; literal HTML text stays
text; touch cancellation and small-screen handles; bounded-memory export and batch
cancellation; Batch queue preserved across mode changes.

## 14. Coordination

Portal session owns Canvas state, renderer, persistence and mode integration. The
queued visual pass on `codex/workspace-visual-pass` (landing-page worktree, clean at
`dd9be15`) overlaps on TopBar, Workspace, queue/stage placement and mobile drawers:
coordinate those, then port reviewed layout changes selectively. Do not cherry-pick
assuming a common base; the portal is a separate repository.
