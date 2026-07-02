"""Low-res /upscale test: is Swin2SR actually super-resolving, or ~bicubic?

Protocol (decisive):
  1. Downscale a clean sharp photo to ~400px wide (bicubic) -> real low-res source.
  2. POST that to live /upscale (swin2sr x4).
  3. Build a plain bicubic x4 of the same low-res source.
  4. Stitch side-by-side: [bicubic x4 | swin2sr x4]  for eyeball diff.
     If swin2sr reads sharper (text edges) than bicubic -> pipeline works.
     If ~identical -> real bug in the SR path.

Usage: KNOCKOUT_TOKEN=kno_xxx python eval/run_lowres.py [case]
  case defaults to "low-res" (source eval/cases/low-res/city.jpg).
"""
import os
import sys
import time
import urllib.request
from io import BytesIO

from PIL import Image

API = "https://useknockout--api.modal.run/upscale"
CASE = sys.argv[2] if len(sys.argv) > 2 else "low-res"
SRC_DIR = os.path.join(os.path.dirname(__file__), "cases", CASE)
OUT = os.path.join(SRC_DIR, "out")

SMALL_W = 400          # downscale target width
SCALE = 4              # /upscale scale
MODEL = "swin2sr"


def find_source():
    for f in sorted(os.listdir(SRC_DIR)):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and "__" not in f:
            return os.path.join(SRC_DIR, f)
    raise SystemExit(f"no source image in {SRC_DIR}")


def post_upscale(token, png_bytes):
    boundary = "----lowres" + str(int(time.time() * 1000))
    fields = {"scale": str(SCALE), "model": MODEL, "format": "png"}
    parts = []
    for k, v in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        )
    parts.append(
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
         f"filename=\"small.png\"\r\nContent-Type: image/png\r\n\r\n").encode()
    )
    body = b"".join(parts) + png_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        API, method="POST", data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    t = time.time()
    resp = urllib.request.urlopen(req, timeout=180)
    out = resp.read()
    ms = int((time.time() - t) * 1000)
    return out, ms, resp.headers.get("content-type", "")


def main():
    token = os.environ.get("KNOCKOUT_TOKEN") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not token:
        raise SystemExit("No token. Set KNOCKOUT_TOKEN or pass as first arg.")
    os.makedirs(OUT, exist_ok=True)

    src_path = find_source()
    orig = Image.open(src_path).convert("RGB")
    ow, oh = orig.size
    print(f"source {os.path.basename(src_path)} {ow}x{oh}")

    # 1. downscale to real low-res
    sw = SMALL_W
    sh = round(oh * sw / ow)
    small = orig.resize((sw, sh), Image.BICUBIC)
    small.save(os.path.join(OUT, "small.png"))
    buf = BytesIO()
    small.save(buf, format="PNG")
    small_png = buf.getvalue()
    print(f"downscaled -> {sw}x{sh}")

    # 3. bicubic baseline x4
    bic = small.resize((sw * SCALE, sh * SCALE), Image.BICUBIC)
    bic.save(os.path.join(OUT, "bicubic-x4.png"))

    # 2. live swin2sr x4
    out_bytes, ms, ct = post_upscale(token, small_png)
    if "image" not in ct:
        raise SystemExit(f"non-image response {ct}: {out_bytes[:300]!r}")
    swin = Image.open(BytesIO(out_bytes)).convert("RGB")
    swin.save(os.path.join(OUT, "swin2sr-x4.png"))
    print(f"swin2sr x4 -> {swin.size}  {ms}ms  {len(out_bytes)//1024}KB")

    # 4. side-by-side [bicubic | swin2sr], same size
    w = min(bic.size[0], swin.size[0])
    h = min(bic.size[1], swin.size[1])
    bic_c, swin_c = bic.crop((0, 0, w, h)), swin.crop((0, 0, w, h))
    cmp = Image.new("RGB", (w * 2 + 8, h), (0, 0, 0))
    cmp.paste(bic_c, (0, 0))
    cmp.paste(swin_c, (w + 8, 0))
    cmp_path = os.path.join(OUT, "lowres-compare.jpg")
    cmp.save(cmp_path, quality=92)
    print(f"compare (bicubic | swin2sr) -> {cmp_path}")


if __name__ == "__main__":
    main()
