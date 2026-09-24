"""Matting benchmark: can we close the veil/lace gap without breaking what works?

Standalone Modal app ("bench-matting"). Never deployed, never touches the `api`
app. Run from the repo root:

    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run eval/bench_matting.py

Models compared, all on the same images:
  birefnet        ZhengPeng7/BiRefNet            (what prod's standard engine runs, raw mask)
  birefnet_mat    ZhengPeng7/BiRefNet-matting    (1024)
  birefnet_hrmat  ZhengPeng7/BiRefNet_HR-matting (2048)
  withoutbg_open  withoutbg OpenWeightsModel     (competitor open model, 448, Apache-2.0)

Every model returns an alpha matte. Composites use pymatting foreground
estimation so semi-transparent pixels (veil, lace) do not carry the original
background colour through; without it a correct matte still looks wrong.

Outputs land in Troy's review folder, not in the repo.
"""
import io
import os
import modal

OUT = r"C:/Users/Troy/OneDrive/Documents/useknockout_test_images/review-2026-09-24-matting-bench"
HARD = r"C:/Users/Troy/OneDrive/Documents/useknockout_test_images/review-2026-09-24-hard-cases"
SHEETS = r"C:/Users/Troy/OneDrive/Documents/useknockout_test_images/review-2026-09-23-3sheets"

IMAGES = {
    "veil": f"{HARD}/veil__0_ORIGINAL.webp",
    "lace": f"{HARD}/lace__0_ORIGINAL.webp",
    "wheel": f"{HARD}/wheel__0_ORIGINAL.webp",
    "fire": f"{HARD}/fire__0_ORIGINAL.webp",
    "tan": f"{SHEETS}/tan__0_ORIGINAL.jpg",
    "salmon": f"{SHEETS}/salmon__0_ORIGINAL.jpg",
    "whitefilm": "eval/cases/kravento-film/white-original.jpg",
    "goldbox": f"{SHEETS}/goldbox__0_ORIGINAL.webp",
    "plaque": "eval/cases/cross-plaque/original.png",
}
MODELS = ["birefnet", "birefnet_mat", "birefnet_hrmat", "withoutbg_open"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0", "torchvision==0.19.0", "transformers==4.44.2",
        "pillow==10.4.0", "timm==1.0.9", "kornia==0.7.3", "einops==0.8.0",
        "huggingface_hub>=0.33.5,<1.0", "numpy==1.26.4",  # withoutbg needs >=0.33.5
        "onnxruntime==1.19.2", "withoutbg==1.1.1",
    )
)
app = modal.App("bench-matting", image=image)

REPOS = {
    "birefnet": ("ZhengPeng7/BiRefNet", 1024),
    "birefnet_mat": ("ZhengPeng7/BiRefNet-matting", 1024),
    "birefnet_hrmat": ("ZhengPeng7/BiRefNet_HR-matting", 2048),
}


@app.cls(gpu="L4", timeout=1800, scaledown_window=120)
class Bench:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForImageSegmentation
        from withoutbg.models import OpenWeightsModel

        torch.set_float32_matmul_precision("high")
        self.nets = {}
        for key, (repo, size) in REPOS.items():
            m = AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True)
            self.nets[key] = (m.to("cuda").eval().half(), size)
        self.wbg = OpenWeightsModel()
        self.wbg.preload()

    def _birefnet(self, key, rgb):
        # Same geometry as prod _get_mask: pad to square, infer, crop back.
        import torch
        from PIL import Image
        from torchvision import transforms

        net, size = self.nets[key]
        w, h = rgb.size
        side = max(w, h)
        sq = Image.new("RGB", (side, side), (0, 0, 0))
        sq.paste(rgb, (0, 0))
        tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        x = tf(sq).unsqueeze(0).to("cuda").half()
        with torch.no_grad():
            pred = net(x)[-1].sigmoid().float().cpu()[0, 0]
        a = transforms.ToPILImage()(pred).resize((side, side), Image.BILINEAR)
        return a.crop((0, 0, w, h))

    @modal.method()
    def run(self, img_bytes: bytes) -> dict:
        import time
        from PIL import Image, ImageOps

        rgb = ImageOps.exif_transpose(Image.open(io.BytesIO(img_bytes))).convert("RGB")
        out = {}
        for key in MODELS:
            t = time.time()
            a = self.wbg.estimate_alpha(rgb) if key == "withoutbg_open" else self._birefnet(key, rgb)
            buf = io.BytesIO()
            a.convert("L").save(buf, format="PNG")
            out[key] = (buf.getvalue(), round(time.time() - t, 2))
        return out


@app.local_entrypoint()
def main():
    # Alphas only. `modal` runs under a Python without pymatting, and the
    # plain Python has no modal, so the sheets are a second step:
    #     python eval/bench_matting_review.py
    import json

    os.makedirs(OUT, exist_ok=True)
    bench = Bench()
    timings = {}
    for name, src in IMAGES.items():
        with open(src, "rb") as f:
            alphas = bench.run.remote(f.read())
        for key, (png, secs) in alphas.items():
            with open(f"{OUT}/{name}__{key}__alpha.png", "wb") as f:
                f.write(png)
        timings[name] = {k: v[1] for k, v in alphas.items()}
        print(name, timings[name])
    with open(f"{OUT}/timings.json", "w") as f:
        json.dump(timings, f, indent=1)
    # IMAGES/MODELS/OUT are also read by bench_matting_review.py via timings.json
    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump({"images": IMAGES, "models": MODELS}, f, indent=1)
