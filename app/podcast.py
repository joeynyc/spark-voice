"""Spark Talk, the podcast generator: /podcast and /podcast/api/*.

The chat model writes a two-host script from a topic (after a quick web search, when search is on);
each line is voiced by cloning the host's reference clip in voices/, so they sound the same in every
episode. Speech models that can't clone use each host's fallback built-in voice instead.
Episodes are kept (script + one WAV per line) for replay.
"""
import asyncio
import html
import io
import json
import re
import time
import uuid
import wave
from datetime import date
from itertools import zip_longest
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from app import config, hood, security
from app.config import CFG, where
from app.providers import llm, search, tts

EPISODES = config.DATA / "podcast" / "episodes"
EPISODES.mkdir(parents=True, exist_ok=True)
KEEP_EPISODES = 40
WORDS_PER_MIN = 155

# The two hosts, from config.yaml: a name, a personality, and a reference clip to clone.
HOSTS = [{**h, "key": h["key"].upper()} for h in CFG["podcast"].get("hosts", [])][:2]
HOST = {h["key"]: h for h in HOSTS}
if len(HOSTS) != 2:
    raise SystemExit("config.yaml: podcast.hosts needs exactly two hosts")
A, B = HOSTS[0]["key"], HOSTS[1]["key"]


def made_of():
    """How the show is made, in a sentence the hosts can use (true for whatever config.yaml says)."""
    parts = [f"{CFG['llm'].get('label', 'a local chat model')} writes their script",
             f"{CFG['tts'].get('label', 'a local speech model')} gives them their voices"]
    if search.enabled:
        parts.append(f"a private search engine ({CFG['search'].get('label', 'SearXNG')}) looks up the news")
    online = " The only thing that touches the internet is that web search; the AI models themselves never leave the desk." if search.enabled else ""
    return f"{', '.join(parts[:-1])} and {parts[-1]}, all running locally on {where()}, with no cloud AI service.{online}"

SYSTEM = """You write scripts for Spark Talk, a funny, warm, two-host podcast. The hosts:
{a}: {a_bio}
{b}: {b_bio}

How the show is made, which is true: {a_name} and {b_name} are AI hosts. {made_of} If the topic touches on how the show is made, own it proudly and get these facts right. Otherwise don't bring it up.

Write the episode as a real, unscripted-sounding conversation between friends:
- Short turns. Most lines are one or two sentences; none longer than about forty words.
- They react to each other: quick replies like "Right.", "No way.", "Wait, what?", "Ha! Okay.", finishing a thought, gentle teasing.
- A few natural fillers ("uh", "I mean", "like", "okay so") and the odd restart ("it's, it's wild"), but don't overdo it.
- Laughs are written as words, like "Ha!" or "Haha, okay." Never use brackets, stage directions, sound effects, emoji, markdown or lists.
- Write numbers, symbols and units as spoken words. No URLs.
- Open with a quick hook and the show name, explain the topic with concrete examples and one surprising fact, and end with a short sign-off.
- Stay accurate. If something is uncertain, the hosts say so.

Today is {today}.{notes}

Output format, nothing else: first a line "TITLE: <a catchy episode title, max eight words>", then one line per turn, each starting with "{a}:" or "{b}:"."""

NOTES = """

Research notes from a web search just now. Base the facts on these, prefer the newest, and never fill gaps with older events from memory. If the notes are thin or disagree, the hosts say so instead of guessing. Once or twice, credit a source naturally by name, but only one whose site is in brackets below, never any other outlet. Keep it a conversation, not a spec sheet: pick the few facts that matter and react to them. Even though the notes use digits and symbols, say every number in words, like "twenty-eight ninety-nine" for $2,899 or "a hundred twenty-eight gigs" for 128GB.
{notes}"""
NO_NOTES = """

A web search just now found nothing useful on this topic. If it's about recent events, the hosts say up front that they couldn't find fresh news and stick to what's well established, without inventing recent details."""

LINE = re.compile(rf"^\s*\**\s*({A}|{B}|TITLE)\s*\**\s*:\s*(.+)$", re.I)
CLEAN = re.compile(r"\[[^\]]*\]|\([^)]*\)|[*_#`>~]")

http = httpx.AsyncClient(timeout=httpx.Timeout(30, connect=5))   # reading the search results' pages
router = APIRouter()
LIVE = {}   # episodes still being written: lines can be voiced before the script is finished


def episode_dir(eid):
    path = EPISODES / re.sub(r"[^0-9a-f]", "", eid)
    if not path.is_dir():
        raise HTTPException(404, "That episode is gone.")
    return path


def site(url):
    return urlparse(url).netloc.removeprefix("www.")


def page_text(raw, limit=1500):
    """The readable paragraphs of an article, roughly: enough for the hosts to get the facts."""
    raw = re.sub(r"(?is)<(script|style|nav|header|footer|aside)\b.*?</\1>", " ", raw)
    out, size = [], 0
    for para in re.findall(r"(?is)<p\b[^>]*>(.*?)</p>", raw):
        text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", para)).split())
        if len(text) > 60:
            out.append(text)
            size += len(text)
            if size > limit:
                break
    return " ".join(out)[:limit]


STOP = set("""about after also and are because been but can could did does doing explain explained for from get got has have
how into its just like latest make more most new news not now really should tell than that the their them then there these they
this those today very want was were what when where which who why will with would year you your""".split())


async def research(topic):
    """Searches the web (news first, then general) and reads the top pages. Returns (sources, notes).
    Search engines return junk for chatty questions, so results must share the topic's key words."""
    hood.event("search", f"Searching the web for “{topic[:80]}”")
    async with hood.busy("search"):
        sources, notes = await _research(topic)
    hood.event("search", f"{len(sources)} sources read" if sources else "Nothing useful found", sites=[s["site"] for s in sources])
    return sources, notes


async def _research(topic):
    query = re.sub(r"\b(this|current) year\b", str(date.today().year), topic[:160], flags=re.I)
    words = set(re.findall(r"[a-z0-9]{3,}", query.lower())) - STOP
    need = min(2, len(words))
    found, seen = [], set()
    general, news = await asyncio.gather(search.search(query), search.search(query, news=True))
    for res in [x for pair in zip_longest(news, general) for x in pair if x]:
        url = res.get("url", "")
        text = f'{res.get("title", "")} {res.get("content", "")} {url}'.lower()
        if sum(w in text for w in words) < need:
            continue
        if url.startswith("http") and len(urlparse(url).path) > 1 and "google." not in site(url) and url not in seen and site(url) not in {s["site"] for s in found}:
            seen.add(url)
            found.append({"title": " ".join((res.get("title") or site(url)).split())[:120], "url": url, "site": site(url),
                          "snippet": " ".join((res.get("content") or "").split())[:400], "date": (res.get("publishedDate") or "")[:10]})
        if len(found) == 6:
            break

    async def read(src):
        return page_text(await security.fetch_page(http, src["url"]))
    pages = await asyncio.gather(*(read(s) for s in found[:3]))
    notes = []
    for n, src in enumerate(found):
        body = pages[n] if n < len(pages) and pages[n] else src["snippet"]
        if body:
            notes.append(f"[{src['site']}{', ' + src['date'] if src['date'] else ''}] {src['title']}: {body}")
    return [{k: s[k] for k in ("title", "url", "site")} for s in found], "\n".join(notes)


def prune():
    eps = sorted(EPISODES.iterdir(), key=lambda p: p.stat().st_mtime)
    for old in eps[:-KEEP_EPISODES]:
        for f in old.iterdir():
            f.unlink(missing_ok=True)
        old.rmdir()


@router.get("/podcast")
def page():
    return FileResponse(config.ROOT / "web" / "podcast.html", headers={"cache-control": "no-cache"})


@router.get("/podcast/api/health")
async def health():
    llm_ok, tts_ok, search_ok = await asyncio.gather(llm.health(), tts.health(), search.health())
    return {"ok": True, "model": llm.model, "llm": llm_ok, "tts": bool(tts_ok), "search": search_ok}


@router.get("/podcast/api/hosts")
def hosts():
    return [{k: h[k] for k in ("key", "name", "hue", "bio")} for h in HOSTS]


class Script(BaseModel):
    topic: str
    minutes: float = 2
    web: bool = True


@router.post("/podcast/api/script")
async def script(req: Script):
    """Streams NDJSON: {type: episode|title|line|done|error}. Lines arrive as the model writes them."""
    topic = " ".join(req.topic.split())[:600]
    if not topic:
        raise HTTPException(400, f"Give {HOSTS[0]['name']} and {HOSTS[1]['name']} something to talk about.")
    minutes = max(0.5, min(5.0, req.minutes))
    words = int(minutes * WORDS_PER_MIN)
    eid = uuid.uuid4().hex[:12]
    folder = EPISODES / eid
    folder.mkdir()
    meta = {"id": eid, "topic": topic, "minutes": minutes, "title": "", "lines": [], "created": int(time.time()), "model": llm.model, "sources": []}
    LIVE[eid] = meta

    web = req.web and search.enabled

    def messages(notes):
        system = SYSTEM.format(a=A, b=B, a_bio=HOSTS[0]["bio"], b_bio=HOSTS[1]["bio"], a_name=HOSTS[0]["name"], b_name=HOSTS[1]["name"],
                               made_of=made_of(), today=date.today().strftime("%A, %B %-d, %Y"),
                               notes=NOTES.format(notes=notes) if notes else NO_NOTES if web else "")
        return [{"role": "system", "content": system},
                {"role": "user", "content": f"Topic: {topic}\nLength: about {words} spoken words in total."}]

    def emit(obj):
        return json.dumps(obj, ensure_ascii=False) + "\n"

    def take(raw):
        m = LINE.match(raw)
        if not m:
            return None
        key, text = m.group(1).upper(), " ".join(CLEAN.sub("", m.group(2)).split())
        text = re.sub(rf"\b({A}|{B})\b", lambda n: HOST[n.group(1)]["name"], text)
        if len(text) > 1 and text[0] == text[-1] == '"':
            text = text[1:-1].strip()
        if not text:
            return None
        if key == "TITLE":
            meta["title"] = text[:80]
            return {"type": "title", "title": meta["title"]}
        line = {"i": len(meta["lines"]), "speaker": key, "text": text}
        meta["lines"].append(line)
        return {"type": "line", **line}

    async def gen():
        yield emit({"type": "episode", "id": eid})
        buf, t0, first, notes = "", time.perf_counter(), None, ""
        if web:
            yield emit({"type": "research"})
            meta["sources"], notes = await research(topic)
            yield emit({"type": "sources", "sources": meta["sources"], "ms": round((time.perf_counter() - t0) * 1000)})
        hood.event("write", f"Writing a {minutes:g}-minute script" + (f" from {len(meta['sources'])} sources" if meta["sources"] else ""))
        writing = hood.busy("write")
        await writing.__aenter__()
        try:
            async for delta in llm.stream(messages(notes), temperature=0.9, max_tokens=int(words * 2.2) + 200):
                buf += delta
                while "\n" in buf:
                    raw, buf = buf.split("\n", 1)
                    if (out := take(raw)):
                        if first is None:
                            first = round((time.perf_counter() - t0) * 1000)
                        if out["type"] == "line":
                            hood.event("write", f"Line {out['i'] + 1} · {HOST[out['speaker']]['name']}", words=len(out["text"].split()))
                        yield emit(out)
            if (out := take(buf)):
                yield emit(out)
        except httpx.HTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            msg = "The script writer isn't running. Start the chat model, then try again." \
                if isinstance(e, httpx.ConnectError) else f"The script writer stopped ({f'HTTP {status}' if status else type(e).__name__})."
            yield emit({"type": "error", "error": msg})
        finally:
            await writing.__aexit__(None, None, None)
            if meta["lines"]:
                hood.event("write", f"Script done · {len(meta['lines'])} lines in {time.perf_counter() - t0:.1f} s")
            (folder / "episode.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
            LIVE.pop(eid, None)
            prune()
        if meta["lines"]:
            yield emit({"type": "done", "lines": len(meta["lines"]), "first_line_ms": first,
                        "script_ms": round((time.perf_counter() - t0) * 1000)})

    return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"cache-control": "no-cache"})


class Voice(BaseModel):
    episode: str
    i: int


@router.post("/podcast/api/voice")
async def voice(req: Voice):
    """Voice one line of an episode in its host's saved voice; the WAV is kept with the episode."""
    folder = episode_dir(req.episode)
    wav_path = folder / f"{req.i:03d}.wav"
    if not wav_path.exists():
        meta = LIVE.get(folder.name)
        if meta is None and (folder / "episode.json").exists():
            meta = json.loads((folder / "episode.json").read_text())
        line = None
        if meta and 0 <= req.i < len(meta["lines"]):
            line = meta["lines"][req.i]
        if line is None:
            raise HTTPException(404, "That line isn't in the script yet.")
        host = HOST[line["speaker"]]
        t0 = time.perf_counter()
        async with hood.busy("voice"):
            if tts.caps["clone"] and host.get("voice"):
                wav = await tts.synth(line["text"], clone=((config.ROOT / host["voice"]).read_bytes(), host["voice_text"]))
            else:
                wav = await tts.synth(line["text"], voice=host.get("fallback_voice"))
        wav_path.write_bytes(wav)
        gen_ms = round((time.perf_counter() - t0) * 1000)
        with wave.open(io.BytesIO(wav)) as w:
            secs = w.getnframes() / w.getframerate()
        hood.event("voice", f"Line {req.i + 1} · {host['name']} · {secs:.1f} s of audio in {gen_ms / 1000:.1f} s", ms=gen_ms, secs=round(secs, 1))
    else:
        gen_ms = 0
    return Response(wav_path.read_bytes(), media_type="audio/wav", headers={"x-gen-ms": str(gen_ms)})


@router.get("/podcast/api/episodes")
def episodes():
    out = []
    for folder in sorted(EPISODES.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            meta = json.loads((folder / "episode.json").read_text())
        except (FileNotFoundError, ValueError):
            continue
        if meta["lines"]:
            voiced = sum(1 for i in range(len(meta["lines"])) if (folder / f"{i:03d}.wav").exists())
            out.append({k: meta[k] for k in ("id", "title", "topic", "created")} | {"lines": len(meta["lines"]), "voiced": voiced})
    return out[:20]


@router.get("/podcast/api/episodes/{eid}")
def episode(eid: str):
    return json.loads((episode_dir(eid) / "episode.json").read_text())


@router.delete("/podcast/api/episodes/{eid}")
def delete_episode(eid: str):
    folder = episode_dir(eid)
    for f in folder.iterdir():
        f.unlink(missing_ok=True)
    folder.rmdir()
    return {"ok": True}
