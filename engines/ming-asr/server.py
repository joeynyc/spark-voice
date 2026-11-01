"""Ming 3.0 ASR (tiny) on CUDA in bf16.

POST /asr <WAV body>               -> {text, language, ms, audio_s}
POST /v1/audio/transcriptions      -> the OpenAI transcription API (multipart "file"), for other apps

The model downloads from Hugging Face on first start.
"""
import io
import os
import sys
import threading
import time

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor

REPO = os.environ.get("ASR_REPO", "inclusionAI/Ming-3.0-asr-tiny")
REVISION = os.environ.get("ASR_REVISION", "ac95e7b1678c08ea6f4a2581a051e432ed63e9b7")
# A plain directory, not the HF cache: the model code imports its sibling files by relative path,
# which the cache's blob symlinks break.
MODEL = os.environ.get("ASR_MODEL", "/models/Ming-3.0-asr-tiny")
LANGUAGE = os.environ.get("ASR_LANGUAGE", "en") or None
# Words it's likely to hear; Context ASR biases toward them. Too many can over-bias, so keep it short.
HOTWORDS = os.environ.get("ASR_HOTWORDS", "DGX Spark,NVIDIA,Nemotron,Ming,GB10,Blackwell")

path = MODEL
if not os.path.exists(os.path.join(path, "config.json")):
    snapshot_download(REPO, revision=REVISION, local_dir=path)
sys.path.insert(0, path)
from asr_request import ASR_INSTRUCTION, context_chunk_frames, decode_transcript, prepare_asr_request  # noqa: E402

PROMPT = ASR_INSTRUCTION + (f"This audio may contains the following words or phrases:{HOTWORDS}." if HOTWORDS else "")
processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16,
                                             device_map="cuda:0", attn_implementation="sdpa").eval()
EOS = getattr(model.config, "text_config", model.config).eos_token_id
lock = threading.Lock()
app = FastAPI()


def transcribe(audio, rate=16000):
    if rate != 16000:
        g = np.gcd(int(rate), 16000)
        audio, rate = resample_poly(audio, 16000 // g, rate // g).astype(np.float32), 16000
    chunks = context_chunk_frames(max(0.1, min(len(audio) / rate, 60)))
    model.config.audio_config.chunk_frames = chunks
    inputs = prepare_asr_request(processor, (audio[None, :], rate), prompt=PROMPT, language_hint=LANGUAGE,
                                 audio_chunk_frames=chunks).to(model.device)
    with lock, torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False, eos_token_id=EOS,
                             pad_token_id=EOS, no_repeat_ngram_size=10)
    return decode_transcript(processor.tokenizer, out[0, inputs.input_ids.shape[1]:].tolist())


# The first call compiles kernels; pay it at startup, not on the first question.
transcribe(np.random.default_rng(0).normal(0, 0.01, 16000 * 2).astype(np.float32))


@app.get("/health")
def health():
    return {"ok": True, "model": REPO, "language": LANGUAGE, "gpu_gb": round(torch.cuda.memory_reserved() / 2**30, 1)}


MAX_BYTES = 20 * 2**20   # about two minutes of 44.1 kHz stereo; it only listens to 60 s anyway


async def read_body(request):
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Audio too large.")
    return bytes(data)


@app.post("/asr")
async def asr(request: Request):
    audio, rate = sf.read(io.BytesIO(await read_body(request)), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t0 = time.perf_counter()
    text = await run_in_threadpool(transcribe, audio, rate)
    return {"text": text, "language": LANGUAGE, "ms": round((time.perf_counter() - t0) * 1000),
            "audio_s": round(len(audio) / rate, 2)}


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(file: UploadFile = File(...)):
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Audio too large.")
    audio, rate = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return {"text": await run_in_threadpool(transcribe, audio, rate)}


@app.get("/v1/models")
def openai_models():
    return {"object": "list", "data": [{"id": "ming-3.0-asr-tiny", "object": "model", "owned_by": "inclusionAI"}]}
