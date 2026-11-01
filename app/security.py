"""Guards for an app that has no accounts and lives on a home network.

- Optional shared token (APP_TOKEN): open the app once with ?token=... and a cookie keeps you in.
- Cross-site requests are refused: a web page elsewhere can't drive /talk, make voices or delete things.
- Request bodies are capped, and the podcast only fetches public web pages.
"""
import asyncio
import hmac
import ipaddress
import os
import socket
from http.cookies import SimpleCookie
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
from fastapi import HTTPException, Request

TOKEN = os.environ.get("APP_TOKEN", "")
COOKIE = "spark_voice_token"
OPEN = {"/icon.png", "/manifest.webmanifest"}


class Guard:
    """ASGI middleware for both HTTP and WebSocket requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        # Same-origin only for anything that changes state or opens a socket. Browsers always
        # send Origin on cross-site POSTs and WebSockets; tools like curl send none and are let through.
        origin = headers.get("origin")
        changes = scope["type"] == "websocket" or scope["method"] not in ("GET", "HEAD", "OPTIONS")
        if changes and origin and urlparse(origin).netloc != headers.get("host"):
            return await refuse(scope, send, 403, "Cross-site requests aren't allowed.")
        if TOKEN and scope["path"] not in OPEN:
            given = parse_qs(scope.get("query_string", b"").decode()).get("token", [""])[0]
            cookie = SimpleCookie(headers.get("cookie", "")).get(COOKIE)
            ok = any(hmac.compare_digest(t, TOKEN) for t in (given, cookie.value if cookie else "") if t)
            if not ok:
                return await refuse(scope, send, 401, "Open Spark Voice with its link, including ?token=...")
            if given and scope["type"] == "http":
                async def send_with_cookie(message):
                    if message["type"] == "http.response.start":
                        message.setdefault("headers", []).append(
                            (b"set-cookie", f"{COOKIE}={TOKEN}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict".encode()))
                    await send(message)
                return await self.app(scope, receive, send_with_cookie)
        return await self.app(scope, receive, send)


async def refuse(scope, send, status, text):
    if scope["type"] == "websocket":
        return await send({"type": "websocket.close", "code": 1008})
    body = text.encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"text/plain; charset=utf-8"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


async def body(request: Request, limit_mb=20):
    """The request body, refusing anything over limit_mb."""
    limit = limit_mb * 2**20
    if int(request.headers.get("content-length") or 0) > limit:
        raise HTTPException(413, f"That's too big (over {limit_mb} MB).")
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > limit:
            raise HTTPException(413, f"That's too big (over {limit_mb} MB).")
    return bytes(data)


async def public(url):
    """True if the URL is http(s) on a standard port and its host resolves only to public addresses,
    so a search result (or a redirect) can't point the Spark at your router or other machines."""
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname or u.port not in (None, 80, 443):
        return False
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(u.hostname, None, type=socket.SOCK_STREAM)
    except OSError:
        return False
    return bool(infos) and all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


async def fetch_page(http, url, max_bytes=1_500_000, seconds=6, hops=3):
    """A public web page's HTML, at most max_bytes, within `seconds` overall; '' if anything's off."""
    async def go():
        nonlocal url
        for _ in range(hops + 1):
            if not await public(url):
                return ""
            async with http.stream("GET", url, follow_redirects=False, headers={"user-agent": "Mozilla/5.0 SparkTalk"}) as r:
                if r.is_redirect and r.headers.get("location"):
                    url = urljoin(url, r.headers["location"])
                    continue
                if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
                    return ""
                data = bytearray()
                async for chunk in r.aiter_bytes():
                    data += chunk
                    if len(data) >= max_bytes:
                        break
                return data.decode(r.encoding or "utf-8", errors="replace")
        return ""
    try:
        return await asyncio.wait_for(go(), seconds)
    except (httpx.HTTPError, asyncio.TimeoutError, UnicodeError, LookupError):
        return ""
