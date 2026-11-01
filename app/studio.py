"""The voice studio: /studio and /studio/api/*.

The speech model makes voices up to three ways: from a text description, from a short recording
(the speech-to-text model writes its transcript), or from a built-in voice. Which ways are on depends
on the model's caps. Saved voices are kept as reference clips so they sound the same every time.
"""
import io
import json
import re
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from scipy.signal import resample_poly

from app import config, security
from app.providers import asr, tts

DATA = config.DATA / "studio"
for sub in ("voices", "refs", "takes"):
    (DATA / sub).mkdir(parents=True, exist_ok=True)
INDEX = DATA / "voices.json"

BUILTIN_IDS = {v["id"] for v in tts.builtin}
REF_WORDS = 40          # a kept voice's reference: its opening sentences, about ten seconds
MAX_CHARS = 1200
REF_MAX_S = 15
KEEP_TAKES = 60
KEEP_REFS = 60          # recordings and uploads not (yet) kept as voices
SR = 44100

router = APIRouter()


def wav_bytes(audio, rate=SR):
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def read_audio(src):
    audio, rate = sf.read(src if isinstance(src, Path) else io.BytesIO(src), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, rate


def to_rate(audio, rate, target):
    if rate == target:
        return audio
    g = np.gcd(int(rate), int(target))
    return resample_poly(audio, target // g, rate // g).astype(np.float32)


def trim_silence(audio, rate, thresh=0.01):
    frame = int(rate * 0.02)
    n = len(audio) // frame
    if not n:
        return audio
    rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    loud = np.nonzero(rms > thresh)[0]
    if not len(loud):
        return audio
    pad = int(rate * 0.15)
    return audio[max(0, loud[0] * frame - pad): min(len(audio), (loud[-1] + 1) * frame + pad)]


def split_sentences(text):
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]


def safe_id(s):
    return re.sub(r"[^0-9a-f]", "", s)


def load_voices():
    try:
        return json.loads(INDEX.read_text())
    except (FileNotFoundError, ValueError):
        return seed_cast() if tts.caps["clone"] else []


def seed_cast():
    """A fresh studio starts with the podcast hosts in its cast."""
    voices = []
    for host in config.CFG["podcast"].get("hosts", []):
        src = config.ROOT / host.get("voice", "")
        if host.get("voice") and src.exists():
            vid = uuid.uuid4().hex[:12]
            (DATA / "voices" / f"{vid}.wav").write_bytes(src.read_bytes())
            voices.append({"id": vid, "name": host["name"], "kind": "designed", "description": host.get("voice_description", ""),
                           "ref_text": host["voice_text"], "emoji": "", "hue": host.get("hue", 220), "created": int(time.time())})
    save_voices(voices)
    return voices


def save_voices(voices):
    tmp = INDEX.with_suffix(".tmp")
    tmp.write_text(json.dumps(voices, ensure_ascii=False, indent=1))
    tmp.replace(INDEX)


async def transcribe(audio, rate):
    text = await asr.transcribe(wav_bytes(to_rate(audio, rate, 16000), 16000))
    return text[:1].upper() + text[1:]


def keep_take(wav, text):
    take = uuid.uuid4().hex[:12]
    (DATA / "takes" / f"{take}.wav").write_bytes(wav)
    (DATA / "takes" / f"{take}.txt").write_text(text)
    takes = sorted((DATA / "takes").glob("*.wav"), key=lambda p: p.stat().st_mtime)
    for old in takes[:-KEEP_TAKES]:
        old.unlink(missing_ok=True)
        old.with_suffix(".txt").unlink(missing_ok=True)
    return take


@router.get("/studio")
def page():
    return FileResponse(config.ROOT / "web" / "studio.html", headers={"cache-control": "no-cache"})


@router.get("/studio/api/health")
async def health():
    return {"ok": True, "tts": bool(await tts.health()), "asr": bool(await asr.health())}


@router.get("/studio/api/voices")
def voices():
    return {"builtin": tts.builtin, "saved": load_voices(), "caps": tts.caps}


@router.post("/studio/api/reference")
async def reference(request: Request):
    """A recording or an uploaded clip: trim it, store it, and transcribe it."""
    raw = await security.body(request, limit_mb=20)
    try:
        audio, rate = read_audio(raw)
    except Exception:
        raise HTTPException(400, "That file isn't audio I can read. Try a WAV, MP3 or M4A.")
    audio = trim_silence(audio, rate)[: int(rate * REF_MAX_S)]
    seconds = len(audio) / rate
    if seconds < 1.5:
        raise HTTPException(400, "That clip is too short. Aim for 5 to 10 seconds of talking.")
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio * min(0.9 / peak, 8.0)   # quiet phone mics: bring the level up
    ref = uuid.uuid4().hex[:12]
    sf.write(DATA / "refs" / f"{ref}.wav", to_rate(audio, rate, SR), SR, subtype="PCM_16")
    for old in sorted((DATA / "refs").glob("*.wav"), key=lambda p: p.stat().st_mtime)[:-KEEP_REFS]:
        old.unlink(missing_ok=True)
    try:
        text = await transcribe(audio, rate)
    except httpx.HTTPError:
        text = ""
    return {"ref": ref, "transcript": text, "seconds": round(seconds, 1)}


@router.get("/studio/api/reference/{ref}")
def reference_audio(ref: str):
    path = DATA / "refs" / f"{safe_id(ref)}.wav"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


class Speak(BaseModel):
    mode: str                      # describe | clone | voice
    text: str
    description: str = ""
    voice: str = ""                # built-in id or saved voice id
    ref: str = ""                  # an unsaved recording
    ref_text: str = ""
    emotion: str = ""
    speed: str = ""
    pitch: str = ""
    volume: str = ""
    lang: str = "en"


def clone_source(req):
    """Reference clip and transcript for a recording or a saved voice."""
    if req.mode == "clone":
        path, text = DATA / "refs" / f"{safe_id(req.ref)}.wav", req.ref_text.strip()
    else:
        saved = next((v for v in load_voices() if v["id"] == req.voice), None)
        if not saved:
            raise HTTPException(404, "That voice isn't in your cast any more.")
        path, text = DATA / "voices" / f"{saved['id']}.wav", saved["ref_text"]
    if not path.exists():
        raise HTTPException(404, "Record or upload a voice first.")
    if not text:
        raise HTTPException(400, "Add what was said in the recording, so the clone can line it up.")
    return path.read_bytes(), text


@router.post("/studio/api/speak")
async def speak(req: Speak):
    text = " ".join(req.text.split())[:MAX_CHARS]
    if not text:
        raise HTTPException(400, "Give them something to say.")
    lang = req.lang if req.lang in tts.caps["languages"] else None
    moods = {k: getattr(req, k) for k in ("emotion", "speed", "pitch", "volume")} if tts.caps["mood"] else None
    t0 = time.perf_counter()
    if req.mode == "describe":
        desc = " ".join(req.description.split())[:400]
        if not desc:
            raise HTTPException(400, "Describe the voice you want first.")
        wav = await tts.synth(text, description=desc, mood=moods, language=lang)
    elif req.mode == "voice" and req.voice in BUILTIN_IDS:
        wav = await tts.synth(text, voice=req.voice, mood=moods, language=lang)
    elif req.mode in ("clone", "voice"):
        wav = await tts.synth(text, clone=clone_source(req), mood=moods, language=lang)
    else:
        raise HTTPException(400, "Pick a voice first.")
    gen_ms = round((time.perf_counter() - t0) * 1000)
    audio_ms = round(sf.info(io.BytesIO(wav)).duration * 1000)
    return Response(wav, media_type="audio/wav", headers={
        "x-take": keep_take(wav, text), "x-gen-ms": str(gen_ms), "x-audio-ms": str(audio_ms)})


class SaveVoice(BaseModel):
    name: str
    kind: str                      # designed | cloned
    take: str = ""                 # a described take becomes the voice's reference
    ref: str = ""                  # or a recording
    ref_text: str = ""
    description: str = ""
    emoji: str = ""


@router.post("/studio/api/voices")
def save_voice(req: SaveVoice):
    name = " ".join(req.name.split())[:40] or "Untitled voice"
    if req.take:
        src = DATA / "takes" / f"{safe_id(req.take)}.wav"
        txt = src.with_suffix(".txt")
        ref_text = txt.read_text() if txt.exists() else req.ref_text
    else:
        src = DATA / "refs" / f"{safe_id(req.ref)}.wav"
        ref_text = req.ref_text.strip()
    if not src.exists():
        raise HTTPException(404, "That take has expired. Make it again, then save.")
    if not ref_text:
        raise HTTPException(400, "Add what was said in the recording first.")
    audio, rate = read_audio(src)
    if req.take:
        # Long takes make slow references: keep the opening sentences.
        words = ref_text.split()
        if len(audio) / rate > 12 and len(words) > 8:
            keep = []
            for s in split_sentences(ref_text):
                if keep and len(" ".join(keep + [s]).split()) > REF_WORDS:
                    break
                keep.append(s)
            frac = len(" ".join(keep).split()) / len(words)
            audio, ref_text = trim_silence(audio[: int(len(audio) * min(1, frac + 0.04))], rate), " ".join(keep)
    vid = uuid.uuid4().hex[:12]
    sf.write(DATA / "voices" / f"{vid}.wav", audio, rate, subtype="PCM_16")
    entry = {"id": vid, "name": name, "kind": req.kind if req.kind in ("designed", "cloned") else "cloned",
             "description": " ".join(req.description.split())[:400], "ref_text": ref_text,
             "emoji": req.emoji[:4], "hue": int(vid, 16) % 360, "created": int(time.time())}
    save_voices(load_voices() + [entry])
    return entry


@router.delete("/studio/api/voices/{vid}")
def delete_voice(vid: str):
    voices = load_voices()
    keep = [v for v in voices if v["id"] != vid]
    if len(keep) == len(voices):
        raise HTTPException(404)
    save_voices(keep)
    (DATA / "voices" / f"{safe_id(vid)}.wav").unlink(missing_ok=True)
    return {"ok": True}


@router.get("/studio/api/voices/{vid}/sample")
def voice_sample(vid: str):
    path = DATA / "voices" / f"{safe_id(vid)}.wav"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")
