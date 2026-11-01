"""Under the hood: live stats from every machine plus what each model is doing right now.

GET /hood/api/state?after=<event id> returns each node's GPU, memory and power (from the stats agent in
engines/stats), how much memory each model holds, the script writer's token rate, which jobs are busy,
and the events since <after>. Polling starts with the first request and stops 30 s after the last one.
"""
import asyncio
import re
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from itertools import count

import httpx
from fastapi import APIRouter

from app.config import CFG
from app.providers import asr, http, llm, tts

METRIC = re.compile(r"^vllm:(generation_tokens_total|num_requests_running)\{[^}]*\}\s+([0-9.e+]+)", re.M)

router = APIRouter()
STATE = {"nodes": {}, "tok_s": 0.0, "llm_running": 0, "gb": {}}
ACT = Counter()
EVENTS = deque(maxlen=200)
SEQ = count(1)
LAST_ASK = 0.0
TASKS = []


def event(kind, text, **extra):
    EVENTS.append({"id": next(SEQ), "t": time.time(), "kind": kind, "text": text, **extra})


@asynccontextmanager
async def busy(kind):
    ACT[kind] += 1
    try:
        yield
    finally:
        ACT[kind] -= 1


def wanted():
    return time.time() - LAST_ASK < 30


async def node_stats():
    while wanted():
        async def one(node):
            try:
                r = await http.get(node["stats"].rstrip("/") + "/stats", timeout=2)
                STATE["nodes"][node["name"]] = {**r.json(), "ts": time.time()}
            except (httpx.HTTPError, ValueError, KeyError):
                pass
        await asyncio.gather(*(one(n) for n in CFG["nodes"] if n.get("stats")))
        await asyncio.sleep(1)


async def model_memory():
    """GPU memory per model: the Ming engines report their own; the chat model is found by process name."""
    process = re.compile(CFG["llm"].get("process", r"VLLM::EngineCore"))
    while wanted():
        for job, engine in (("voice", tts), ("listen", asr)):
            h = await engine.health()
            STATE["gb"][job] = (h or {}).get("gpu_gb")
        node = STATE["nodes"].get(CFG["llm"].get("node"), {})
        STATE["gb"]["write"] = round(sum(a["gb"] for a in node.get("apps", []) if process.search(a["name"])), 1) or None
        await asyncio.sleep(3)


async def writer_metrics():
    prev = None
    while wanted():
        try:
            vals = {k: float(v) for k, v in METRIC.findall((await http.get(llm.metrics_url(), timeout=2)).text)}
            now, toks = time.time(), vals.get("generation_tokens_total", 0)
            if prev and now > prev[0] and toks >= prev[1]:
                STATE["tok_s"] = round((toks - prev[1]) / (now - prev[0]), 1)
            prev = (now, toks)
            STATE["llm_running"] = int(vals.get("num_requests_running", 0))
        except (httpx.HTTPError, ValueError):
            STATE["tok_s"], prev = 0.0, None
        await asyncio.sleep(0.5)


def ensure_running():
    global LAST_ASK
    LAST_ASK = time.time()
    if all(t.done() for t in TASKS):
        TASKS.clear()
        TASKS.extend(asyncio.create_task(job()) for job in (node_stats, model_memory, writer_metrics))


@router.get("/hood/api/state")
async def state(after: int = 0):
    ensure_running()
    now = time.time()
    nodes = [{"name": n["name"], **STATE["nodes"].get(n["name"], {}), "ok": now - STATE["nodes"].get(n["name"], {}).get("ts", 0) < 5}
             for n in CFG["nodes"]]
    return {"t": now, "nodes": nodes, "gb": STATE["gb"], "tok_s": STATE["tok_s"], "llm_running": STATE["llm_running"],
            "act": {k: v for k, v in ACT.items() if v > 0}, "events": [e for e in EVENTS if e["id"] > after][-60:]}
