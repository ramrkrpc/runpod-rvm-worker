# -*- coding: utf-8 -*-
"""RunPod Serverless worker: Robust Video Matting (RVM).
Input:  {"video_url": "<mp4 url>", "downsample_ratio": 0.25, "output": "alpha"|"foreground"}
Output: {"url": "<catbox url of the matte video>"}
"""
import os, sys, time, subprocess, requests, torch, runpod

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# ---- cold start: load RVM once per worker ----
if not os.path.exists("/RVM"):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/PeterL1n/RobustVideoMatting", "/RVM"], check=True)
sys.path.insert(0, "/RVM")

# --- patch RVM's string-rate bug: newer PyAV's to_avrational needs .numerator
#     (int/Fraction have it; the str f'{frame_rate:.4f}' RVM passes does NOT) ---
_iu = "/RVM/inference_utils.py"
try:
    _s = open(_iu, encoding="utf-8").read()
    if "rate=f'{frame_rate:.4f}'" in _s:
        _s = _s.replace("rate=f'{frame_rate:.4f}'", "rate=round(frame_rate)")
        open(_iu, "w", encoding="utf-8").write(_s)
        print("patched RVM string-rate bug")
except Exception as _e:
    print("rate patch warn:", _e)

from inference import convert_video  # noqa: E402

MODEL = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50", trust_repo=True).cuda().eval()


def _catbox(path):
    r = subprocess.run(["curl", "-s", "-A", UA, "-F", "reqtype=fileupload",
                        "-F", f"fileToUpload=@{path}", "https://catbox.moe/user/api.php"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def handler(job):
    inp = job.get("input", {}) or {}
    url = inp.get("video_url")
    if not url:
        return {"error": "missing video_url"}
    dsr = float(inp.get("downsample_ratio", 0.25))
    want = inp.get("output", "alpha")
    last = None
    for _ in range(4):
        try:
            r = requests.get(url, timeout=600, headers={"User-Agent": UA})
            r.raise_for_status()
            with open("in.mp4", "wb") as f:
                f.write(r.content)
            last = None
            break
        except Exception as e:
            last = e
            time.sleep(3)
    if last is not None:
        return {"error": f"download failed: {last}"}
    common = dict(model=MODEL, input_source="in.mp4", output_type="video",
                  downsample_ratio=dsr, seq_chunk=12)
    if want == "foreground":
        convert_video(output_foreground="out.mp4", **common)
        out = "out.mp4"
    else:
        convert_video(output_alpha="alpha.mp4", **common)
        out = "alpha.mp4"
    return {"url": _catbox(out), "output_type": want}


runpod.serverless.start({"handler": handler})
