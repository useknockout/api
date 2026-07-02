"""Run the flowerbox eval images through live /studio-shot in a few variants.
Saves outputs to eval/cases/flowerbox/out/. Token via KNOCKOUT_TOKEN env or arg.
Usage: KNOCKOUT_TOKEN=kno_xxx python eval/run_flowerbox.py
"""
import os
import sys
import time
import urllib.request

API = "https://useknockout--api.modal.run/studio-shot"
SRC = os.path.join(os.path.dirname(__file__), "cases", "flowerbox")
OUT = os.path.join(SRC, "out")

# (label, form fields) — exercise default, transparent, enhance, enhance+transparent
VARIANTS = [
    ("default", {"aspect": "1:1", "format": "jpg"}),
    ("transparent", {"aspect": "1:1", "transparent": "true", "format": "png"}),
    ("enhance", {"aspect": "1:1", "enhance": "true", "format": "jpg"}),
    ("enhance_transparent", {"aspect": "1:1", "enhance": "true", "transparent": "true", "format": "png"}),
]


def post(token, img_path, fields):
    boundary = "----flowerbox" + str(int(time.time() * 1000))
    body = b""
    with open(img_path, "rb") as f:
        data = f.read()
    parts = []
    for k, v in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        )
    parts.append(
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
         f"filename=\"{os.path.basename(img_path)}\"\r\nContent-Type: image/jpeg\r\n\r\n").encode()
    )
    body = b"".join(parts) + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        API, method="POST", data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    t = time.time()
    resp = urllib.request.urlopen(req, timeout=120)
    out = resp.read()
    ms = int((time.time() - t) * 1000)
    return out, ms, resp.headers.get("content-type", "")


def main():
    token = os.environ.get("KNOCKOUT_TOKEN") or (sys.argv[1] if len(sys.argv) > 1 else "")
    if not token:
        print("No token. Set KNOCKOUT_TOKEN or pass as arg.")
        sys.exit(1)
    os.makedirs(OUT, exist_ok=True)
    imgs = sorted(f for f in os.listdir(SRC) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    for img in imgs:
        for label, fields in VARIANTS:
            ext = fields.get("format", "jpg")
            try:
                out, ms, ct = post(token, os.path.join(SRC, img), fields)
                name = f"{os.path.splitext(img)[0]}__{label}.{ext}"
                with open(os.path.join(OUT, name), "wb") as f:
                    f.write(out)
                print(f"OK  {name}  {ms}ms  {len(out)//1024}KB  {ct}")
            except Exception as e:
                print(f"ERR {img} {label}: {e}")


if __name__ == "__main__":
    main()
