"""Speech to text. `ming` is the Ming 3.0 ASR engine in engines/ming-asr; `openai` is any server with
the OpenAI transcription API (vLLM serving Whisper or Parakeet, faster-whisper-server, Speaches...)."""
import httpx


class MingASR:
    def __init__(self, cfg, http):
        self.url = cfg.get("url", "http://127.0.0.1:7797").rstrip("/")
        self.http = http

    async def transcribe(self, wav):
        r = await self.http.post(f"{self.url}/asr", content=wav, headers={"content-type": "audio/wav"}, timeout=60)
        r.raise_for_status()
        return r.json().get("text", "").strip()

    async def health(self):
        try:
            r = await self.http.get(f"{self.url}/health", timeout=3)
            return r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None


class OpenAIASR:
    def __init__(self, cfg, http):
        self.url = cfg.get("url", "http://127.0.0.1:8000/v1").rstrip("/")
        self.model = cfg.get("model", "whisper-1")
        self.language = cfg.get("language", "en")
        self.headers = {"authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}
        self.http = http

    async def transcribe(self, wav):
        data = {"model": self.model, "response_format": "json"}
        if self.language:
            data["language"] = self.language
        r = await self.http.post(f"{self.url}/audio/transcriptions", data=data, files={"file": ("speech.wav", wav, "audio/wav")},
                                 headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.json().get("text", "").strip()

    async def health(self):
        try:
            r = await self.http.get(f"{self.url}/models", headers=self.headers, timeout=3)
            return {"ok": True} if r.status_code == 200 else None
        except httpx.HTTPError:
            return None


PROVIDERS = {"ming": MingASR, "openai": OpenAIASR}
