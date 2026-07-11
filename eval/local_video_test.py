"""LOCAL test of the /video/remove worker's composite + remux logic. NO Modal,
NO GPU, $0 — the model is replaced with a fake elliptical mask; everything
else (bg_image / bg_blur / bg_color branches, speed remux + atempo chain,
billing math) is the same code path shape as process_video_job steps 3-5.

Usage: py eval/local_video_test.py <input.mp4>
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFilter

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("TEMP", "/tmp"), "test-clip.mp4")
FPS_CAP = 30


def fake_mask(size):
    """Soft-edged ellipse standing in for BiRefNet."""
    w, h = size
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).ellipse([w * 0.25, h * 0.15, w * 0.75, h * 0.9], fill=255)
    return m.filter(ImageFilter.GaussianBlur(6))


def probe_duration(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "json", path], capture_output=True, text=True, timeout=60)
    return float(json.loads(p.stdout)["format"]["duration"])


def has_audio_stream(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=codec_type", "-of", "json", path],
                       capture_output=True, text=True, timeout=60)
    return bool(json.loads(p.stdout).get("streams"))


def run_case(label, bg_color=None, bg_image=None, bg_blur=0, speed=1.0, fmt="mp4"):
    tmp = tempfile.mkdtemp(prefix="lvt-")
    try:
        # step 3: demux + audio (same commands as the worker)
        frames_dir = os.path.join(tmp, "frames")
        os.makedirs(frames_dir)
        fps = FPS_CAP
        subprocess.run(["ffmpeg", "-y", "-i", SRC, "-vf", f"fps={fps:.3f},scale='min(1920,iw)':-2",
                        os.path.join(frames_dir, "%05d.png")], capture_output=True, timeout=300, check=True)
        audio = os.path.join(tmp, "audio.m4a")
        has_audio = subprocess.run(["ffmpeg", "-y", "-i", SRC, "-vn", "-acodec", "aac", audio],
                                   capture_output=True, timeout=120).returncode == 0
        frame_files = sorted(os.listdir(frames_dir))
        n = len(frame_files)

        # step 4: composite with FAKE mask (model stand-in)
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir)
        bg_img_sized = None
        blur_radius = 2 + (bg_blur / 100.0) * 38 if bg_blur > 0 else 0
        opaque = bool(bg_image or bg_blur > 0 or bg_color is not None)
        for name in frame_files:
            frame = Image.open(os.path.join(frames_dir, name)).convert("RGB")
            mask = fake_mask(frame.size)
            if bg_image is not None:
                if bg_img_sized is None or bg_img_sized.size != frame.size:
                    bg_img_sized = bg_image.resize(frame.size, Image.LANCZOS)
                bg = bg_img_sized
            elif bg_blur > 0:
                bg = frame.filter(ImageFilter.GaussianBlur(blur_radius))
            elif bg_color is not None:
                bg = Image.new("RGB", frame.size, bg_color)
            else:
                out = frame.convert("RGBA")
                out.putalpha(mask)
                out.save(os.path.join(out_dir, name))
                continue
            out = Image.composite(frame, bg, mask)  # plain composite is fine for wiring test
            out.save(os.path.join(out_dir, name))

        # step 5: remux with speed (same command construction as the worker)
        if opaque or fmt == "mp4":
            out_name, vcodec = "out.mp4", ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
        elif fmt == "webm":
            out_name, vcodec = "out.webm", ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "24"]
        else:
            out_name, vcodec = "out.mov", ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"]
        cmd = ["ffmpeg", "-y", "-framerate", f"{fps * speed:.3f}", "-i", os.path.join(out_dir, "%05d.png")]
        if has_audio:
            cmd += ["-i", audio]
            if abs(speed - 1.0) < 1e-6:
                cmd += ["-c:a", "copy"]
            else:
                stages, s = [], speed
                while s > 2.0:
                    stages.append("atempo=2.0"); s /= 2.0
                while s < 0.5:
                    stages.append("atempo=0.5"); s /= 0.5
                stages.append(f"atempo={s:.4f}")
                cmd += ["-filter:a", ",".join(stages), "-c:a", "aac"]
            cmd += ["-shortest"]
        cmd += vcodec + [os.path.join(tmp, out_name)]
        r = subprocess.run(cmd, capture_output=True, timeout=600)
        if r.returncode != 0:
            print(f"[{label}] FFMPEG FAIL: {r.stderr[-300:]}")
            return False

        out_path = os.path.join(tmp, out_name)
        dur = probe_duration(out_path)
        in_dur = n / fps
        expect = in_dur / speed
        units = max(1, math.ceil(in_dur / speed))
        aud = has_audio_stream(out_path)
        ok = abs(dur - expect) < 0.6 and (aud == has_audio)
        keep = os.path.join(os.environ.get("TEMP", "/tmp"), f"lvt-{label}.{out_name.split('.')[-1]}")
        shutil.copy(out_path, keep)
        print(f"[{label}] {'PASS' if ok else 'FAIL'} out={dur:.2f}s expect={expect:.2f}s "
              f"audio={aud} units={units} (${units * 0.10:.2f}) -> {keep}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print(f"source: {SRC} ({probe_duration(SRC):.2f}s)")
    bg = Image.new("RGB", (640, 480))
    for y in range(480):  # gradient backdrop stand-in
        for x in range(0, 640, 8):
            bg.paste((30 + x % 200, 60, 120 + y % 100), [x, y, x + 8, y + 1])
    results = [
        run_case("solid-2x", bg_color=(255, 255, 255), speed=2.0),
        run_case("blur40-1x", bg_blur=40),
        run_case("bgimage-halfspeed", bg_image=bg, speed=0.5),
        run_case("transparent-prores-4x", fmt="prores4444", speed=4.0),
    ]
    print("ALL PASS" if all(results) else "SOME FAILED")
