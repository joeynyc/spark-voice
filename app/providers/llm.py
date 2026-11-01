"""The chat model: anything that speaks the OpenAI chat API (vLLM, SGLang, llama.cpp, Ollama, LM Studio, a cloud API)."""
import json

import httpx


class ChatModel:
    def __init__(self, cfg, http):
        self.url = cfg.get("base_url", "http://127.0.0.1:8000/v1").rstrip("/")
        self.model = cfg.get("model", "")
        self.extra = cfg.get("extra_body") or {}
        self.headers = {"authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}
        self.http = http

    async def stream(self, messages, **opts):
        """Yields the reply's text as it arrives. Raises httpx.ConnectError if the model is down."""
        body = {"model": self.model, "stream": True, "messages": messages, **self.extra, **opts}
        async with self.http.stream("POST", f"{self.url}/chat/completions", json=body, headers=self.headers) as r:
            if r.status_code != 200:
                raise httpx.HTTPStatusError((await r.aread()).decode()[:200], request=r.request, response=r)
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                choices = json.loads(line[6:]).get("choices") or [{}]
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta

    async def health(self):
        try:
            return (await self.http.get(f"{self.url}/models", headers=self.headers, timeout=3)).status_code == 200
        except httpx.HTTPError:
            return False

    def metrics_url(self):
        """vLLM's Prometheus metrics (token rate for the under-the-hood views); other servers just won't answer."""
        return self.url.removesuffix("/v1") + "/metrics"
