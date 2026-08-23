"""Build 4-up review panels: original | FeyNobg | FeyNobg+boundary_refine | alpha.

Magenta backdrop: retained white background pixels are unmistakable against it,
unlike checkerboard (white-on-white) or black (loses the film's own edge).
Run from this folder:  python make_panels.py
"""
import numpy as np
from PIL import Image, ImageDraw
import os, glob

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = None  # alphas are stored alongside as <film>-feynobg-alpha.png
MAGENTA = np.array([255, 0, 200], dtype=np.float64)
FILMS = ["white", "palepink", "beige", "sage", "blush", "rose", "tan"]
PH = 760  # panel height


def comp(orig, alpha, bg):
    oa = np.asarray(orig.convert("RGB"), dtype=np.float64)
    al = np.asarray(alpha.convert("L"), dtype=np.float64)[..., None] / 255.0
    return Image.fromarray((oa * al + bg * (1 - al)).astype(np.uint8))


def fit(im, h=PH):
    return im.resize((int(im.width * h / im.height), h), Image.LANCZOS)


def label_strip(width, text, color=(255, 255, 120)):
    s = Image.new("RGB", (width, 30), (16, 16, 16))
    ImageDraw.Draw(s).text((8, 9), text, fill=color)
    return s


for film in FILMS:
    orig_p = os.path.join(HERE, f"{film}-original.jpg")
    alpha_p = os.path.join(HERE, f"{film}-feynobg-alpha.png")
    ref_p = os.path.join(HERE, f"{film}-feynobg-refined.jpg")
    if not (os.path.exists(orig_p) and os.path.exists(alpha_p)):
        print("skip", film)
        continue

    orig = Image.open(orig_p).convert("RGB")
    alpha = Image.open(alpha_p).convert("L")
    if alpha.size != orig.size:
        alpha = alpha.resize(orig.size, Image.LANCZOS)

    ralpha_p = os.path.join(HERE, f"{film}-refined-alpha.png")
    ralpha = Image.open(ralpha_p).convert("L") if os.path.exists(ralpha_p) else alpha
    if ralpha.size != orig.size:
        ralpha = ralpha.resize(orig.size, Image.LANCZOS)

    p1 = fit(orig)
    p2 = fit(comp(orig, alpha, MAGENTA))
    p3 = fit(comp(orig, ralpha, MAGENTA))
    p4 = fit(alpha.convert("RGB"))

    labels = ["1. ORIGINAL PHOTO", "2. FEYNOBG (raw model)",
              "3. FEYNOBG + BOUNDARY REFINE", "4. ALPHA MASK"]
    panels = [p1, p2, p3, p4]
    gap = 10
    W = sum(p.width for p in panels) + gap * (len(panels) - 1)
    H = PH + 34
    sheet = Image.new("RGB", (W, H), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    x = 0
    for p, lab in zip(panels, labels):
        sheet.paste(p, (x, 34))
        d.text((x + 6, 11), lab, fill=(255, 255, 120))
        x += p.width + gap
    out = os.path.join(HERE, f"panel-{film}.jpg")
    sheet.save(out, quality=91)
    print("saved", os.path.basename(out), sheet.size)
