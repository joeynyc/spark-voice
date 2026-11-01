"""Records a live Spark Talk episode for a social video: 4K screen capture + the page's own audio,
plus a shot list (camera targets with timestamps) that render.py turns into zooms.

Needs the GPU display from start-gpu-display.sh (DISPLAY=:1), Playwright (pip install playwright;
playwright install chromium) and ffmpeg with x11grab and h264_nvenc. Usage:
  python3 record.py "<topic>" takes/t1 [minutes]          (SPARK_VOICE_URL defaults to http://127.0.0.1:7799)
"""
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC, OUT = sys.argv[1], sys.argv[2]
MINUTES = int(sys.argv[3]) if len(sys.argv) > 3 else 1
URL = os.environ.get("SPARK_VOICE_URL", "http://127.0.0.1:7799") + "/podcast?hood=flow"
CSS_W, CSS_H, DSF = 1536, 864, 2.5          # 1536x864 layout drawn at 3840x2160
os.makedirs(OUT, exist_ok=True)
os.environ.update(DISPLAY=":1", XAUTHORITY=f"{HERE}/xauth", GOOGLE_API_KEY="no",
                  GOOGLE_DEFAULT_CLIENT_ID="no", GOOGLE_DEFAULT_CLIENT_SECRET="no")

shots = []      # {"t": epoch seconds, "box": [x, y, w, h] in CSS px, or None for the whole screen, "name"}


async def box(pg, *selectors, pad=18):
    """The union of the elements' boxes, in CSS px."""
    rects = await pg.evaluate("""(sels) => sels.map((s) => { const e = document.querySelector(s); if (!e) return null;
        const r = e.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; }).filter(Boolean)""", list(selectors))
    if not rects:
        return None
    x0, y0 = min(r[0] for r in rects) - pad, min(r[1] for r in rects) - pad
    x1, y1 = max(r[2] for r in rects) + pad, max(r[3] for r in rects) + pad
    return [x0, y0, x1 - x0, y1 - y0]


async def shot(name, pg=None, *selectors, pad=18):
    b = await box(pg, *selectors, pad=pad) if selectors else None
    shots.append({"t": time.time(), "box": b, "name": name})
    print(f"{time.strftime('%H:%M:%S')} shot {name} {b and [round(v) for v in b]}", flush=True)


async def main():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(tempfile.mkdtemp(), headless=False, no_viewport=True, args=[
            f"--app={URL}", f"--window-size={CSS_W},{CSS_H}", "--window-position=0,0", f"--force-device-scale-factor={DSF}",
            "--ignore-gpu-blocklist", "--enable-gpu-rasterization", "--autoplay-policy=no-user-gesture-required",
            "--hide-scrollbars", "--noerrdialogs", "--test-type", "--disable-infobars"])
        await asyncio.sleep(2)
        pg = [q for q in ctx.pages if "podcast" in q.url][0]
        await pg.evaluate("localStorage.clear()")
        await pg.reload(); await pg.wait_for_timeout(2500)
        await pg.mouse.move(CSS_W - 2, CSS_H - 2)
        await pg.click(f'#len button[data-m="{MINUTES}"]')
        # The page's audio (voices, music, stings) goes to a recorder as well as the speakers.
        await pg.evaluate("""() => { audio(); const dest = ctx.createMediaStreamDestination(); master.connect(dest);
            const rec = new MediaRecorder(dest.stream, { mimeType: "audio/webm;codecs=opus", audioBitsPerSecond: 320000 });
            window.__chunks = []; rec.ondataavailable = (e) => window.__chunks.push(e.data); window.__rec = rec; }""")

        ff = subprocess.Popen(["ffmpeg", "-hide_banner", "-y", "-f", "x11grab", "-framerate", "30", "-video_size", "3840x2160", "-draw_mouse", "0",
                               "-i", ":1", "-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "15", "-b:v", "0",
                               f"{OUT}/screen.mp4"], stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        start = None
        while start is None:
            line = ff.stderr.readline()
            if m := re.search(r"start: ([0-9.]+)", line):
                start = float(m.group(1))
        log = []
        threading.Thread(target=lambda: log.extend(iter(ff.stderr.readline, "")), daemon=True).start()   # keep ffmpeg's pipe drained
        audio_t0 = await pg.evaluate("() => { window.__rec.start(500); return Date.now() / 1000; }")

        await shot("wide"); await pg.wait_for_timeout(2600)
        await shot("compose", pg, ".compose", pad=24); await pg.wait_for_timeout(900)
        await pg.click("#topic")
        await pg.keyboard.type(TOPIC, delay=38)
        await pg.wait_for_timeout(700)
        await pg.click("#go"); await pg.mouse.move(CSS_W - 2, CSS_H - 2)
        await shot("wide-start")
        await pg.wait_for_timeout(900)
        await shot("flow-search", pg, "#hflow", ".stagebar", pad=10)
        await pg.wait_for_selector(".sources a", timeout=30000); await pg.wait_for_timeout(300)
        await shot("sources", pg, ".sources", ".scripthead", pad=22)
        await pg.wait_for_selector(".ln", timeout=60000); await pg.wait_for_timeout(1400)
        await shot("script", pg, ".sources", ".ln", pad=16)
        await pg.wait_for_selector(".host.talking", timeout=60000)
        await shot("stage", pg, "#stage", pad=8)
        await pg.wait_for_timeout(3500)
        await pg.keyboard.press("2"); await pg.wait_for_timeout(250)
        await shot("rack-spark2", pg, '#hrack [data-n="spark2"]', ".rk-link", pad=26)
        await pg.wait_for_timeout(5500)
        await shot("rack-both", pg, "#hrack", "#hosts", pad=14)
        await pg.wait_for_timeout(3500)
        await pg.keyboard.press("3"); await pg.wait_for_timeout(250)
        await shot("hud-stage", pg, "#stage", pad=8)
        await pg.wait_for_timeout(3800)
        await shot("hud-log", pg, ".hud-tick", "#timeline", pad=20)
        await pg.wait_for_timeout(4200)
        await pg.keyboard.press("1"); await pg.wait_for_timeout(250)
        await shot("wide-flow")
        await pg.wait_for_timeout(5000)
        await shot("hosts", pg, "#hosts", ".caption", pad=10)
        await pg.wait_for_function("() => ep && ep.scriptDone && !play.on && play.i >= ep.lines.length", timeout=300000, polling=200)
        await shot("wide-end")
        await pg.wait_for_timeout(3800)

        data = await pg.evaluate("""() => new Promise((res) => { window.__rec.onstop = async () => {
            const b = new Blob(window.__chunks, { type: "audio/webm" }); const r = new FileReader();
            r.onload = () => res(r.result.split(",")[1]); r.readAsDataURL(b); }; window.__rec.stop(); })""")
        end = time.time()
        ff.stdin.write("q"); ff.stdin.flush(); ff.wait(timeout=60)
        print("ffmpeg:", "".join(log[-2:]).strip()[-200:], flush=True)
        with open(f"{OUT}/audio.webm", "wb") as f:
            f.write(base64.b64decode(data))
        meta = await pg.evaluate("() => ({ id: ep.id, title: ep.title, lines: ep.lines.map((l) => [l.speaker, l.text]), sources: [...document.querySelectorAll('.sources a')].map((a) => a.href) })")
        json.dump({"topic": TOPIC, "video_start": start, "audio_start": audio_t0, "end": end, "css": [CSS_W, CSS_H], "dsf": DSF,
                   "shots": [{**s, "t": s["t"] - start} for s in shots], "episode": meta}, open(f"{OUT}/take.json", "w"), indent=1)
        print("done", round(end - start, 1), "s;", meta["title"], flush=True)
        await ctx.close()

asyncio.run(main())
