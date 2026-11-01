"""Ming 3.0 TTS over HTTP.

POST /tts {text, voice}      -> WAV in a built-in voice, fast (the assistant)
POST /synth {...}            -> described, cloned or captioned voices (the studio and the podcast)
POST /v1/audio/speech        -> the OpenAI speech API, built-in voices, for other apps
GET  /health                 -> includes the GPU memory this engine holds
"""
import base64
import functools
import hashlib
import io
import os
import threading
import time
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

import tts_tasks
from ming_tts import MingTTS

from huggingface_hub import snapshot_download

MODEL = os.environ.get("MING_TTS_MODEL", "inclusionAI/Ming-3.0-tts-0.5B")
REVISION = os.environ.get("MING_TTS_REVISION", "7247f4c94a5b2e987404ed28a5cfec41b1a4ca61")   # pinned, like the code
MAX_CHARS = 2000            # one request holds the GPU; keep it bounded
MAX_PROMPT_B64 = 20 * 2**20  # about 15 MB of reference audio
KEEP_PROMPTS = 200

# Flow-matching ODE steps per audio patch (release default 10). The flow head is ~70% of
# step time, so 5 (Ming's own EPSS low-step schedule) roughly halves synthesis time for the
# assistant; outputs stayed word-exact through Ming ASR. The studio asks for 10 per request.
FLOW_STEPS = int(os.environ.get("FLOW_STEPS", "5"))

tts = MingTTS(MODEL if os.path.isdir(MODEL) else snapshot_download(MODEL, revision=REVISION), text_normalization=False)
base_sample = tts.model.flowloss.sample
tts.model.flowloss.sample = functools.partial(base_sample, steps=FLOW_STEPS)
PROMPTS = Path(os.environ.get("PROMPT_DIR", "/tmp/prompts"))
PROMPTS.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()  # one GPU job at a time
# First synthesis after load is ~3x slower (kernel/allocator warm-up); pay it before serving.
tts.synthesize(task="IP", text="Warming up.", controls={"ip": "US_James"}, use_zero_spk_emb=True)
app = FastAPI()
BUILTIN = ["US_James", "US_Stella", "US_Emma", "UK_George", "UK_Charlotte"]


class TTSRequest(BaseModel):
    text: str = Field(max_length=MAX_CHARS)
    voice: str = "US_James"


class SynthRequest(BaseModel):
    """The voice studio's request: one of a description, a reference clip or a caption."""
    text: str = Field(max_length=MAX_CHARS)
    description: Optional[str] = Field(None, max_length=1000)   # freeform voice design (task FREEFORM)
    caption: Optional[dict] = None         # ip / emotion / speed / pitch / volume
    prompt_wav: Optional[str] = Field(None, max_length=MAX_PROMPT_B64)   # base64 WAV of a voice to clone
    prompt_text: Optional[str] = None      # what is said in prompt_wav
    language: Optional[str] = None
    flow_steps: Optional[int] = None


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "sample_rate": tts.sample_rate, "flow_steps": FLOW_STEPS,
            "gpu_gb": round(torch.cuda.memory_reserved() / 2**30, 1)}


def wav_response(wav, t0):
    pcm = wav.squeeze(0).float().cpu().clamp(-1, 1).numpy()
    buf = io.BytesIO()
    sf.write(buf, pcm, tts.sample_rate, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav", headers={
        "X-Synth-Ms": f"{(time.perf_counter() - t0) * 1000:.0f}",
        "X-Audio-Ms": f"{len(pcm) / tts.sample_rate * 1000:.0f}",
    })


@app.post("/tts")
def synthesize(req: TTSRequest):
    t0 = time.perf_counter()
    with lock:
        wav = tts.synthesize(task="IP", text=req.text, controls={"ip": req.voice},
                             use_zero_spk_emb=True)
    return wav_response(wav, t0)


def prompt_file(b64):
    """Reference clips are stored by content hash: MingTTS caches a prompt's latents per path."""
    data = base64.b64decode(b64)
    path = PROMPTS / f"{hashlib.sha1(data).hexdigest()}.wav"
    if not path.exists():
        path.write_bytes(data)
        for old in sorted(PROMPTS.glob("*.wav"), key=lambda p: p.stat().st_mtime)[:-KEEP_PROMPTS]:
            old.unlink(missing_ok=True)
    return str(path)


@app.post("/synth")
def synth(req: SynthRequest):
    caption = {k: v for k, v in (req.caption or {}).items() if v}
    kw = dict(text=req.text, language_id=req.language or None)
    if req.description:
        # Freeform splits long text itself and clones its own first fragment for the rest.
        kw.update(task="FREEFORM", instruction=req.description, use_zero_spk_emb=True)
    else:
        # A mixed caption (voice + mood) isn't one of the single-control tasks, so it goes
        # in as a plain TTS instruction, built with the checkpoint's own caption layout.
        kw.update(task="TTS", instruction=tts_tasks.build_structured_instruction(**caption) if caption else None)
        if req.prompt_wav:
            if not (req.prompt_text or "").strip():
                raise HTTPException(400, "prompt_text is required with prompt_wav")
            kw.update(prompt_wav_path=prompt_file(req.prompt_wav), prompt_text=req.prompt_text, use_spk_emb=True)
        else:
            kw.update(use_zero_spk_emb=True)
    steps = max(2, min(16, req.flow_steps or FLOW_STEPS))
    t0 = time.perf_counter()
    with lock:
        tts.model.flowloss.sample = functools.partial(base_sample, steps=steps)
        try:
            wav = tts.synthesize(**kw)
        except ValueError as e:
            raise HTTPException(400, str(e))
        finally:
            tts.model.flowloss.sample = functools.partial(base_sample, steps=FLOW_STEPS)
    return wav_response(wav, t0)


class SpeechRequest(BaseModel):
    """OpenAI's /v1/audio/speech. Only WAV out; unknown voices fall back to the first built-in."""
    input: str = Field(max_length=MAX_CHARS)
    voice: str = "US_James"
    model: str = "ming-3.0-tts"
    response_format: str = "wav"


@app.post("/v1/audio/speech")
def openai_speech(req: SpeechRequest):
    if req.response_format not in ("wav", "pcm"):
        raise HTTPException(400, "Only wav is supported.")
    return synthesize(TTSRequest(text=req.input, voice=req.voice if req.voice in BUILTIN else BUILTIN[0]))


@app.get("/v1/models")
def openai_models():
    return {"object": "list", "data": [{"id": "ming-3.0-tts", "object": "model", "owned_by": "inclusionAI"}]}
