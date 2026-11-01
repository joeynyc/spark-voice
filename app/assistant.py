"""The voice assistant: /talk (WebSocket) and /speech/asr.

The page records you and sends the audio to /speech/asr (the speech-to-text model), then sends the
transcript over /talk. The chat model's reply streams back; it is cut into sentences, each is voiced
by the speech model, and the WAVs go back over the same socket with per-stage timings.
"""
import asyncio
import io
import json
import logging
import re
import time

import httpx
import soundfile as sf
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app import hood, security
from app.config import CFG, where
from app.providers import asr, llm, tts

router = APIRouter()
log = logging.getLogger("spark-voice")
VOICES = {v["id"] for v in tts.builtin}
VOICE = CFG["assistant"].get("voice") if CFG["assistant"].get("voice") in VOICES else tts.builtin[0]["id"]

SYSTEM = (
    f"You are a voice assistant running entirely on {where()}. "
    "Your reply is spoken aloud. By default answer in one to three short conversational sentences. "
    "When the user asks you to teach, explain in depth, or walk them through something, give a "
    "spoken mini-lesson instead: up to about two hundred words in short, plain sentences, building "
    "one idea at a time, ending with a one-sentence recap. "
    "You have no live access to the machines' metrics, so never state speeds, temperatures or other "
    "measurements as if you had measured them. "
    "No markdown, lists, code, emoji or URLs. Write numbers, units and symbols as words "
    "(say 'three point five gigahertz', not '3.5 GHz')."
)

SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
CLAUSE_END = re.compile(r"(?<=[,;:])\s+|\s+(?=—)")
MARKUP = re.compile(r"[*_#`>|~]+")

def ms(t0):
    return round((time.perf_counter() - t0) * 1000)


FIRST_MAX_WORDS = 8           # opener length once no natural break comes soon enough
FIRST_NATURAL_MAX_WORDS = 12  # a comma or full stop within this many words wins
JOINERS = {"at", "across", "on", "in", "with", "which", "that", "and", "but", "so", "to", "for",
           "because", "from", "while", "when", "if", "about", "by", "or", "than"}


def _first_phrase_cut(buf, until):
    """Opener with no clause break: cut before a joining word once there are
    FIRST_MAX_WORDS+ words, so speech starts on a phrase boundary, not mid-phrase."""
    words = list(re.finditer(r"\S+", buf[:until]))
    complete = words[:-1] if until == len(buf) else words  # last word may still be streaming
    if len(complete) <= FIRST_MAX_WORDS:
        return None
    def joiner(w):
        return w.group().lower().strip(",.") in JOINERS

    for i in range(FIRST_MAX_WORDS, 1, -1):  # longest opener first, at least two words
        if joiner(complete[i]) and not joiner(complete[i - 1]):
            return complete[i].start()
    return complete[FIRST_MAX_WORDS - 1].end()


def pop_sentences(buf, force=False, first=False):
    """Split complete sentences off the front of buf; return (sentences, rest).

    first=True cuts the opening chunk at the first clause break instead: TTS time
    scales with audio length, so a short opener gets the first sound out sooner.
    """
    out = []
    if first:
        breaks = [m.end() for m in (CLAUSE_END.search(buf, 15), SENTENCE_END.search(buf)) if m]
        end = min(breaks, default=len(buf))
        if breaks and len(buf[:end].split()) <= FIRST_NATURAL_MAX_WORDS:
            cut = end
        else:
            cut = _first_phrase_cut(buf, until=end)
        if cut:
            out.append(buf[:cut].strip())
            buf = buf[cut:]
    while True:
        m = SENTENCE_END.search(buf)
        if not m:
            break
        out.append(buf[:m.end()].strip())
        buf = buf[m.end():]
    # Very long run-on: cut at the last comma so speech can start.
    if not out and len(buf) > 180 and "," in buf[60:]:
        cut = buf.rindex(",") + 1
        out.append(buf[:cut].strip())
        buf = buf[cut:]
    if force and buf.strip():
        out.append(buf.strip())
        buf = ""
    return [s for s in (MARKUP.sub("", s).strip() for s in out) if s], buf


async def tts_worker(ws, emit, queue, t0, stats, voice):
    idx = 0
    while (text := await queue.get()) is not None:
        t = time.perf_counter()
        async with hood.busy("voice"):
            wav = await tts.speak(text, voice)
        synth_ms = ms(t)
        if idx == 0:
            stats["first_audio_ms"] = ms(t0)
        await emit({
            "type": "audio", "idx": idx, "text": text, "synth_ms": synth_ms,
            "audio_ms": round(sf.info(io.BytesIO(wav)).duration * 1000),
            "roundtrip_ms": ms(t), "at_ms": ms(t0),
        })
        await ws.send_bytes(wav)
        idx += 1


async def run_turn(ws, emit, history, user_text, voice=VOICE):
    t0 = time.perf_counter()
    stats = {}
    history.append({"role": "user", "content": user_text})
    queue = asyncio.Queue()
    worker = asyncio.create_task(tts_worker(ws, emit, queue, t0, stats, voice))
    reply, buf, tokens, ttft, queued = "", "", 0, None, 0
    messages = [{"role": "system", "content": SYSTEM}] + history[-12:]
    try:
        async with hood.busy("write"):
            async for delta in llm.stream(messages, max_tokens=600, temperature=0.6):
                if ttft is None:
                    ttft = ms(t0)
                tokens += 1
                reply += delta
                buf += delta
                await emit({"type": "token", "text": delta})
                sentences, buf = pop_sentences(buf, first=not queued)
                for s in sentences:
                    queue.put_nowait(s)
                queued += len(sentences)
        llm_ms = ms(t0)
        sentences, _ = pop_sentences(buf, force=True, first=not queued)
        for s in sentences:
            queue.put_nowait(s)
        await emit({"type": "llm_done", "ttft_ms": ttft, "llm_ms": llm_ms, "chunks": tokens})
        queue.put_nowait(None)
        await worker
        history.append({"role": "assistant", "content": reply})
        await emit({"type": "done", "total_ms": ms(t0), **stats})
    finally:
        worker.cancel()


async def guarded_turn(ws, turn_id, history, user_text, voice):
    async def emit(obj):
        await ws.send_text(json.dumps({**obj, "turn": turn_id}))

    try:
        await run_turn(ws, emit, history, user_text, voice)
    except asyncio.CancelledError:
        if history and history[-1]["role"] == "user":
            history.pop()  # interrupted turn: drop the unanswered question
        raise
    except Exception as e:  # surface model failures in the page instead of a silent hang; details stay in the log
        log.exception("assistant turn failed")
        await emit({"type": "error", "error": f"A model didn't answer ({type(e).__name__})."})


@router.post("/speech/asr")
async def speech_asr(request: Request):
    """The pages post 16 kHz WAV here; the speech-to-text model may live on another machine."""
    wav = await security.body(request, limit_mb=10)
    t0 = time.perf_counter()
    try:
        async with hood.busy("listen"):
            text = await asr.transcribe(wav)
    except httpx.HTTPError as e:
        log.warning("speech recognition failed: %s", e)
        return JSONResponse({"error": f"Speech recognition isn't answering ({type(e).__name__})."}, status_code=503)
    return {"text": text, "ms": ms(t0), "audio_s": round(sf.info(io.BytesIO(wav)).duration, 2)}


@router.get("/speech/health")
async def speech_health():
    return JSONResponse({}, status_code=200 if await asr.health() else 503)


@router.websocket("/talk")
async def talk(ws: WebSocket):
    await ws.accept()
    history, turn = [], None
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if turn and not turn.done():
                turn.cancel()  # barge-in: a new question or cancel stops the current answer
            if msg["type"] == "reset":
                history.clear()
            elif msg["type"] == "user" and msg.get("text", "").strip():
                voice = msg.get("voice") if msg.get("voice") in VOICES else VOICE
                turn = asyncio.create_task(guarded_turn(ws, msg.get("id"), history, msg["text"].strip(), voice))
    except WebSocketDisconnect:
        if turn:
            turn.cancel()
