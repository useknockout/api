"""Three-way test of the Kravento pale-leaves case against the LIVE /remove endpoint.

standard      : default BiRefNet path (what the client got)
high_recall   : detect=high_recall (2nd pass on chroma-boosted copy, union)
product-v1    : engine=product-v1 (S3OD + CascadePSP, won the 2026-08-17 lightbox bakeoff)

Usage: python eval/run_pale_leaves.py   (reads API_TOKEN from .env)
"""
import os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import sys as _s
API = "https://useknockout--api.modal.run/" + (_s.argv[1] if len(_s.argv) > 1 else "remove")
CASE = os.path.join(HERE, "cases", "flower-pale-leaves")
SRC = os.path.join(CASE, "original.jpg")
OUT = os.path.join(CASE, "out"); os.makedirs(OUT, exist_ok=True)

EP = _s.argv[1] if len(_s.argv) > 1 else "remove"
BASE = {"format": "png"} if EP == "remove" else {"format": "png", "transparent": "true"}
VARIANTS = [
    (f"{EP}_standard",    {**BASE}),
    (f"{EP}_high_recall", {**BASE, "detect": "high_recall"}),
    (f"{EP}_product_v1",  {**BASE, "engine": "product-v1"}),
]

def token():
    t = os.environ.get("API_TOKEN") or os.environ.get("KNOCKOUT_TOKEN")
    if t: return t
    for line in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
        if line.startswith("API_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("no API_TOKEN")

def post(tok, img_path, fields):
    boundary = "----paleleaves" + str(int(time.time() * 1000))
    data = open(img_path, "rb").read()
    parts = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                  f"filename=\"original.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").encode())
    body = b"".join(parts) + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(API, data=body, method="POST", headers={
        "Authorization": f"Bearer {tok}",
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read(), time.time() - t0, dict(r.headers)

tok = token()
for name, fields in VARIANTS:
    try:
        png, dt, hdr = post(tok, SRC, fields)
        p = os.path.join(OUT, f"{name}.png"); open(p, "wb").write(png)
        print(f"{name:12s} OK  {len(png)/1e6:.1f} MB  {dt:5.1f}s  -> {p}", flush=True)
    except urllib.error.HTTPError as e:
        print(f"{name:12s} HTTP {e.code}: {e.read()[:300]!r}", flush=True)
    except Exception as e:
        print(f"{name:12s} ERR {e!r}", flush=True)
