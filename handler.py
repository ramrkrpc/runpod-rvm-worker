# -*- coding: utf-8 -*-
"""RunPod Serverless worker: Robust Video Matting (RVM).
Input:  {"video_url": "<mp4 url>", "downsample_ratio": 0.25, "output": "alpha"|"foreground"}
Output: {"url": "<catbox url of the matte video>"}
"""
import os, sys, subprocess, requests, torch, runpod

# ---- cold start: load RVM once per worker ----
if not os.path.exists("/RVM"):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/PeterL1n/RobustVideoMatting", "/RVM"], check=True)
sys.path.insert(0, "/RVM")
from inference import convert_video  # noqa: E402

MODEL = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50", trust_repo=True).cuda().eval()


def _catbox(path):
    r = subprocess.run(["curl", "-s", "-F", "reqtype=fileupload",
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
    with open("in.mp4", "wb") as f:
        f.write(requests.get(url, timeout=600).content)
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
