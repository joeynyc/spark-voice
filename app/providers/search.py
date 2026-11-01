"""Web search for the podcast's research step. `searxng` is a self-hosted SearXNG (engines/searxng);
`none` turns research off, and the page hides its "Use the web" switch."""
import httpx


class SearXNG:
    enabled = True

    def __init__(self, cfg, http):
        self.url = cfg.get("url", "http://127.0.0.1:7795").rstrip("/")
        self.http = http

    async def search(self, query, news=False):
        """Results as [{title, url, content, publishedDate}]; [] if the search engine is down."""
        params = {"q": query, "format": "json", **({"categories": "news"} if news else {})}
        try:
            r = await self.http.get(f"{self.url}/search", params=params, timeout=6)
            return r.json().get("results", [])
        except (httpx.HTTPError, ValueError):
            return []

    async def health(self):
        try:
            return (await self.http.get(f"{self.url}/healthz", timeout=3)).status_code == 200
        except httpx.HTTPError:
            return False


class NoSearch:
    enabled = False

    def __init__(self, cfg, http):
        pass

    async def search(self, query, news=False):
        return []

    async def health(self):
        return False


PROVIDERS = {"searxng": SearXNG, "none": NoSearch}
