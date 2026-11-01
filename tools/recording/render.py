"""Turns a take from record.py into the posted video: 1920x1080 at 30 fps, with smooth camera moves
(zoom in, pan, zoom out) to each shot's target, and the page's audio normalized for social.

Usage: python3 render.py <take dir> [out.mp4]
"""
import json
import math
import subprocess
import sys

import cv2
import numpy as np

TAKE = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{TAKE}/spark-talk.mp4"
SW, SH = 3840, 2160            # capture
OW, OH = 1920, 1080            # output
FPS = 30
MOVE = 1.1                     # seconds per camera move
MAX_ZOOM = 2.3
DRIFT = 0.025                  # a slow push-in while holding a shot, so it never looks frozen

take = json.load(open(f"{TAKE}/take.json"))
k = SW / take["css"][0]


def target(box):
    """A 16:9 camera rect (cx, cy, w) in capture pixels that frames the box."""
    if not box:
        return (SW / 2, SH / 2, SW)
    x, y, w, h = (v * k for v in box)
    cw = min(SW, max(w, h * 16 / 9, SW / MAX_ZOOM))
    ch = cw * 9 / 16
    cx = min(max(x + w / 2, cw / 2), SW - cw / 2)
    cy = min(max(y + h / 2, ch / 2), SH - ch / 2)
    return (cx, cy, cw)


def ease(u):
    u = min(1, max(0, u))
    return u * u * (3 - 2 * u) if u < 1 else 1.0


shots = [(s["t"], target(s["box"])) for s in take["shots"]]
end = take["end"] - take["video_start"]


def seg(m, t):
    """The camera during shot m: moving from where shot m started toward its target, then a slow push-in."""
    ts, tgt = shots[m]
    cur = blend(starts[m], tgt, ease((t - ts) / MOVE))
    held = t - ts - MOVE
    if held > 0 and tgt[2] < SW:
        cur = (cur[0], cur[1], cur[2] * (1 - DRIFT * min(1, held / 8)))
    return cur


def camera(t):
    m = max([n for n, (ts, _) in enumerate(shots) if ts <= t] or [0])
    return seg(m, t)


def blend(a, b, u):
    # zoom in log space so speed feels even; centers follow
    w = math.exp(math.log(a[2]) + (math.log(b[2]) - math.log(a[2])) * u)
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, w)


starts = [shots[0][1]]           # where each camera move begins: wherever the previous shot had got to
for n in range(1, len(shots)):
    starts.append(seg(n - 1, shots[n][0]))


dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", f"{TAKE}/screen.mp4", "-t", f"{end:.3f}", "-vf", f"fps={FPS}",
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
offset = take["audio_start"] - take["video_start"]
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                        "-itsoffset", f"{offset:.3f}", "-i", f"{TAKE}/audio.webm", "-map", "0:v", "-map", "1:a",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high", "-pix_fmt", "yuv420p",
                        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000", "-c:a", "aac", "-b:a", "192k",
                        "-t", f"{end:.3f}", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
size = SW * SH * 3
n = 0
while True:
    raw = dec.stdout.read(size)
    if len(raw) < size:
        break
    frame = np.frombuffer(raw, np.uint8).reshape(SH, SW, 3)
    cx, cy, cw = camera(n / FPS)
    ch = cw * 9 / 16
    if cw >= SW - 0.5:
        out = cv2.resize(frame, (OW, OH), interpolation=cv2.INTER_AREA)
    else:
        crop = cv2.getRectSubPix(frame, (round(cw), round(ch)), (cx - 0.5, cy - 0.5))
        out = cv2.resize(crop, (OW, OH), interpolation=cv2.INTER_AREA if cw > OW else cv2.INTER_CUBIC)
    enc.stdin.write(out.tobytes())
    n += 1
    if n % 300 == 0:
        print(f"{n / FPS:.0f} s", flush=True)
enc.stdin.close()
enc.wait()
print("wrote", OUT, n, "frames")
