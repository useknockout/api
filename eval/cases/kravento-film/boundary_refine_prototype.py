"""boundary_refine v2 — product-color snap with smoothness gate.

For each boundary row (left/right) and column (top/bottom):
  1. sample LOCAL product colour from well inside the mask (adapts to lighting),
  2. walk inward from the current boundary until the pixel matches that colour,
  3. smooth the resulting boundary offsets across the contour (median filter),
  4. accept only where offsets are consistent (low spread) — complex contours no-op,
  5. remove-only, capped displacement, then feather 1px.
"""
import numpy as np
from PIL import Image, ImageFilter

MAXSHIFT = 70      # px cap (scaled for 3072x4080; ~2% of width)
TOL = 26           # colour distance considered "product"
INSET = 160        # how far inside to sample local product colour
SMOOTH = 41        # median window over boundary offsets
RESID_MAX = 9.0    # px; reject segment if offsets deviate from a smooth trend


def _median1d(x, k):
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    out = np.empty_like(x, dtype=np.float64)
    for i in range(x.size):
        out[i] = np.median(xp[i:i + k])
    return out


def _snap_axis(img, fg, axis, side):
    """Return per-line inward offsets (or -1 where not applicable).

    axis=0 scans rows (left/right edges); axis=1 scans columns (top/bottom).
    """
    if axis == 1:
        img_t = np.transpose(img, (1, 0, 2))
        fg_t = fg.T
    else:
        img_t, fg_t = img, fg

    n = fg_t.shape[0]
    offs = np.full(n, -1.0)
    for i in range(n):
        xs = np.where(fg_t[i])[0]
        if xs.size < 40:
            continue
        b = xs.min() if side == 0 else xs.max()
        step = 1 if side == 0 else -1
        deep = b + step * INSET
        if not (0 <= deep < img_t.shape[1]):
            continue
        lo, hi = sorted((deep - step * 30, deep + step * 30))
        ref = np.median(img_t[i, max(lo, 0):hi + 1], axis=0)
        # walk inward
        k = 0
        while k < MAXSHIFT:
            x = b + step * k
            if not (0 <= x < img_t.shape[1]):
                break
            if np.linalg.norm(img_t[i, x] - ref) <= TOL:
                break
            k += 1
        offs[i] = k
    return offs


def boundary_refine(rgb, alpha):
    img = np.asarray(rgb.convert("RGB"), dtype=np.float64)
    a = np.asarray(alpha.convert("L"), dtype=np.float64)
    fg = a > 128
    if fg.sum() == 0:
        return alpha, {}

    new_a = a.copy()
    stats = {}
    for axis in (0, 1):
        for side in (0, 1):
            offs = _snap_axis(img, fg, axis, side)
            valid = offs >= 0
            if valid.sum() < 100:
                continue
            sm = offs.copy()
            sm[valid] = _median1d(offs[valid], SMOOTH)
            # smoothness = residual against a quadratic trend along the contour.
            # A strip that widens gradually along a diagonal edge is smooth;
            # a genuinely complex contour is not.
            idx = np.where(valid)[0].astype(np.float64)
            vals = sm[valid]
            coef = np.polyfit(idx, vals, 2)
            resid = float(np.mean(np.abs(np.polyval(coef, idx) - vals)))
            name = f"{'rows' if axis == 0 else 'cols'}{'lo' if side == 0 else 'hi'}"
            stats[name] = (round(float(np.median(vals)), 1), round(resid, 1))
            if resid > RESID_MAX:
                continue  # erratic -> complex contour -> no-op
            # follow the fitted trend, not the noisy per-line measurement
            sm[valid] = np.clip(np.polyval(coef, idx), 0, MAXSHIFT)
            # apply
            for i in np.where(valid)[0]:
                k = int(round(sm[i]))
                if k <= 0:
                    continue
                if axis == 0:
                    xs = np.where(fg[i])[0]
                    b = xs.min() if side == 0 else xs.max()
                    if side == 0:
                        new_a[i, b:b + k] = 0.0
                    else:
                        new_a[i, max(b - k + 1, 0):b + 1] = 0.0
                else:
                    ys = np.where(fg[:, i])[0]
                    b = ys.min() if side == 0 else ys.max()
                    if side == 0:
                        new_a[b:b + k, i] = 0.0
                    else:
                        new_a[max(b - k + 1, 0):b + 1, i] = 0.0

    out = Image.fromarray(np.clip(new_a, 0, 255).astype(np.uint8), mode="L")
    return out.filter(ImageFilter.GaussianBlur(1.0)), stats


if __name__ == "__main__":
    import glob, os
    SP = os.path.dirname(os.path.abspath(__file__))
    DL = r"C:\Users\Troy\Downloads\drive-download-20260807T200135Z-1-001"
    TEAL = np.array([0, 170, 165], dtype=np.float64)
    for mask_p in sorted(glob.glob(os.path.join(SP, "film_feynobg_*.png"))):
        key = os.path.basename(mask_p).replace("film_feynobg_", "").replace(".png", "")
        orig = Image.open(glob.glob(os.path.join(DL, f"{key}*.jpg"))[0]).convert("RGB")
        raw = Image.open(mask_p).convert("L")
        if raw.size != orig.size:
            raw = raw.resize(orig.size, Image.LANCZOS)
        ref, stats = boundary_refine(orig, raw)
        oa = np.asarray(orig, dtype=np.float64)
        al = np.asarray(ref, dtype=np.float64)[..., None] / 255.0
        Image.fromarray((oa * al + TEAL * (1 - al)).astype(np.uint8)).save(
            os.path.join(SP, f"ref_{key}_bref2.png"))
        print(key, stats)
