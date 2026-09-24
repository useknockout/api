"""Build review sheets from bench_matting.py output. Plain Python, no Modal.

    python eval/bench_matting_review.py

Per image: original | each model on magenta | each model's alpha. Composites
use pymatting foreground estimation, so semi-transparent pixels (veil, lace)
do not carry the original background colour through.
"""
import json
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pymatting import estimate_foreground_ml

OUT = r"C:/Users/Troy/OneDrive/Documents/useknockout_test_images/review-2026-09-24-matting-bench"
H = 420
MAGENTA = np.array([1.0, 0.0, 1.0])


def fit(im):
    return im.resize((int(im.width * H / im.height), H), Image.LANCZOS)


def review(name, src, models, timings):
    rgb = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    s = min(1.0, 1400 / max(rgb.size))
    size = (round(rgb.width * s), round(rgb.height * s))
    small = rgb.resize(size, Image.LANCZOS)
    img = np.asarray(small, np.float64) / 255.0

    top = [("ORIGINAL", fit(small))]
    bottom = [("", Image.new("RGB", fit(small).size, (20, 20, 20)))]
    for key in models:
        a = Image.open(f"{OUT}/{name}__{key}__alpha.png").convert("L").resize(size, Image.BILINEAR)
        a = np.asarray(a, np.float64) / 255.0
        fg = estimate_foreground_ml(img, a)
        comp = fg * a[..., None] + MAGENTA * (1 - a[..., None])
        top.append((f"{key}  {timings[key]}s", fit(Image.fromarray((comp * 255).clip(0, 255).astype(np.uint8)))))
        bottom.append((f"{key} alpha", fit(Image.fromarray((a * 255).astype(np.uint8)).convert("RGB"))))

    sheet = Image.new("RGB", (sum(t.width + 8 for _, t in top), 2 * (H + 30)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for row, items in enumerate((top, bottom)):
        x = 0
        for label, t in items:
            sheet.paste(t, (x, row * (H + 30) + 30))
            d.text((x + 4, row * (H + 30) + 9), f"{name.upper()}  {label}", fill=(255, 255, 120))
            x += t.width + 8
    sheet.save(f"{OUT}/BENCH-{name}.png")


if __name__ == "__main__":
    manifest = json.load(open(f"{OUT}/manifest.json"))
    timings = json.load(open(f"{OUT}/timings.json"))
    for name, src in manifest["images"].items():
        review(name, src, manifest["models"], timings[name])
        print("sheet", name)
