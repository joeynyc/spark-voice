"""Text to speech. `ming` is the Ming 3.0 TTS engine in engines/ming-tts: built-in voices, cloning from a
short clip, voices designed from a description, and mood controls. `openai` is any server with the OpenAI
speech API (Kokoro-FastAPI, or the Ming engine's own /v1 route): a fixed set of voices, no cloning.

Each provider says what it can do in `caps`; the pages hide what the model can't.
"""
import base64
import logging

import httpx
from fastapi import HTTPException


log = logging.getLogger("spark-voice")


def _failed(r):
    log.warning("voice engine error %s: %s", r.status_code, r.text[:300])
    return HTTPException(502, f"The voice engine returned an error (HTTP {r.status_code}).")


def _unreachable(name):
    return HTTPException(503, f"The voice engine ({name}) isn't answering.")


class MingTTS:
    caps = {"clone": True, "describe": True, "mood": True, "languages": ["en", "es", "fr", "de", "ja", "ko"]}
    builtin = [
        {"id": "US_James", "name": "James", "tag": "American · warm narrator", "flag": "🇺🇸"},
        {"id": "US_Stella", "name": "Stella", "tag": "American · bright and friendly", "flag": "🇺🇸"},
        {"id": "US_Emma", "name": "Emma", "tag": "American · clear and calm", "flag": "🇺🇸"},
        {"id": "UK_George", "name": "George", "tag": "British · composed", "flag": "🇬🇧"},
        {"id": "UK_Charlotte", "name": "Charlotte", "tag": "British · polished", "flag": "🇬🇧"},
    ]
    # Caption values are the checkpoint's own (Chinese) labels.
    EMOTION = {"happy": "高兴", "sad": "悲伤", "angry": "愤怒", "scared": "恐惧",
               "disgusted": "厌恶", "surprised": "惊讶", "neutral": "中性"}
    LEVEL = {"low": "低", "mid": "中", "high": "高"}
    SPEED = {"slow": "慢速", "mid": "中速", "fast": "快速"}
    # A description can't carry the control fields, so the mood is said in words instead.
    EMOTION_WORDS = {"happy": "Sound happy and upbeat.", "sad": "Sound sad.", "angry": "Sound angry.",
                     "scared": "Sound scared and nervous.", "disgusted": "Sound disgusted.",
                     "surprised": "Sound surprised.", "neutral": "Keep the delivery neutral."}
    SPEED_WORDS = {"slow": "Speak slowly.", "fast": "Speak quickly."}
    PITCH_WORDS = {"low": "Use a low pitch.", "high": "Use a high pitch."}

    def __init__(self, cfg, http):
        self.url = cfg.get("url", "http://127.0.0.1:7798").rstrip("/")
        # Flow-matching steps for studio-quality takes: at the engine's fast default (5), long
        # described voices drift off the script; at 10 they stay word-exact.
        self.quality_steps = int(cfg.get("quality_steps", 10))
        self.http = http

    async def _post(self, path, body):
        try:
            r = await self.http.post(f"{self.url}{path}", json=body)
        except httpx.HTTPError:
            raise _unreachable("Ming TTS")
        if r.status_code == 400:   # e.g. a clip it can't read: the engine's own words are about the input, so pass them on
            try:
                detail = str(r.json().get("detail", ""))[:200]
            except ValueError:
                detail = ""
            raise HTTPException(400, detail or "The voice engine couldn't use that.")
        if r.status_code != 200:
            raise _failed(r)
        return r.content

    async def speak(self, text, voice):
        """Fast path for the assistant: a built-in voice at the engine's default steps."""
        return await self._post("/tts", {"text": text, "voice": voice})

    def _caption(self, mood, ip=None):
        mood = mood or {}
        fields = dict(emotion=self.EMOTION.get(mood.get("emotion")), speed=self.SPEED.get(mood.get("speed")),
                      pitch=self.LEVEL.get(mood.get("pitch")), volume=self.LEVEL.get(mood.get("volume")), ip=ip)
        return {k: v for k, v in fields.items() if v} or None

    async def synth(self, text, *, voice=None, clone=None, description=None, mood=None, language=None):
        """One of: a built-in `voice`, `clone=(wav bytes, what is said in it)`, or a `description`."""
        mood = mood or {}
        body = {"text": text, "language": language if language and language != "en" else None, "flow_steps": self.quality_steps}
        if description:
            body["description"] = " ".join([description.rstrip(".") + "."] + [w for w in (
                self.EMOTION_WORDS.get(mood.get("emotion")), self.SPEED_WORDS.get(mood.get("speed")),
                self.PITCH_WORDS.get(mood.get("pitch"))) if w])
        elif clone:
            body["prompt_wav"], body["prompt_text"] = base64.b64encode(clone[0]).decode(), clone[1]
            body["caption"] = self._caption(mood)
        else:
            body["caption"] = self._caption(mood, ip=voice or self.builtin[0]["id"])
        return await self._post("/synth", body)

    async def health(self):
        try:
            r = await self.http.get(f"{self.url}/health", timeout=3)
            return r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None


class OpenAITTS:
    caps = {"clone": False, "describe": False, "mood": False, "languages": ["en"]}

    def __init__(self, cfg, http):
        self.url = cfg.get("url", "http://127.0.0.1:8880/v1").rstrip("/")
        self.model = cfg.get("model", "tts-1")
        self.headers = {"authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}
        self.builtin = [v if isinstance(v, dict) else {"id": v, "name": v.split("_")[-1].title(), "tag": "", "flag": "🎙️"}
                        for v in cfg.get("voices") or ["alloy"]]
        self.http = http

    async def speak(self, text, voice):
        try:
            r = await self.http.post(f"{self.url}/audio/speech", headers=self.headers, json={
                "model": self.model, "input": text, "voice": voice or self.builtin[0]["id"], "response_format": "wav"})
        except httpx.HTTPError:
            raise _unreachable(self.model)
        if r.status_code != 200:
            raise _failed(r)
        return r.content

    async def synth(self, text, *, voice=None, clone=None, description=None, mood=None, language=None):
        if clone or description:
            raise HTTPException(400, "This speech model can only use its built-in voices.")
        return await self.speak(text, voice)

    async def health(self):
        try:
            r = await self.http.get(f"{self.url}/models", headers=self.headers, timeout=3)
            return {"ok": True} if r.status_code == 200 else None
        except httpx.HTTPError:
            return None


PROVIDERS = {"ming": MingTTS, "openai": OpenAITTS}
