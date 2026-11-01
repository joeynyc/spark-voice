"""Spark Voice: the voice assistant (/), the voice studio (/studio) and the Spark Talk podcast (/podcast).

Run: uvicorn app.main:app --host 0.0.0.0 --port 7799
"""
import asyncio

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app import assistant, config, hood, podcast, security, studio
from app.config import CFG
from app.providers import asr, llm, search, tts

WEB = config.ROOT / "web"

app = FastAPI(title="Spark Voice", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(security.Guard)
app.include_router(assistant.router)   # /talk, /speech/*
app.include_router(studio.router)      # /studio, /studio/api/*
app.include_router(podcast.router)     # /podcast, /podcast/api/*
app.include_router(hood.router)        # /hood/api/state


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"cache-control": "no-cache"})


@app.get("/icon.png")
def icon():
    return FileResponse(WEB / "icon.png")


@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(WEB / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/api/config")
def page_config():
    """What the pages need to describe this setup: machines, model names, and what the speech model can do."""
    def model(job, role):
        c = CFG[job]
        return {"label": c.get("label", job), "spec": c.get("spec", ""), "node": c.get("node", ""), "role": role}
    return {
        "badge": config.badge(),
        "machine": CFG["machine"],
        "nodes": [n["name"] for n in CFG["nodes"]],
        "models": {"write": model("llm", "Writes and answers"), "voice": model("tts", "Speaks"),
                   "listen": model("asr", "Listens"), "search": model("search", "Web search")},
        "search": search.enabled,
        "tts": {"caps": tts.caps, "voices": tts.builtin, "assistant_voice": assistant.VOICE},
        "hosts": podcast.hosts(),
    }


@app.get("/health")
async def health():
    llm_ok, tts_ok, asr_ok = await asyncio.gather(llm.health(), tts.health(), asr.health())
    return {"gateway": True, "llm": llm_ok, "tts": bool(tts_ok), "asr": bool(asr_ok)}
