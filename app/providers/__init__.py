"""One object per job, built from config.yaml. Swap a model by changing its section there."""
import httpx

from app.config import CFG
from app.providers import asr as _asr
from app.providers import search as _search
from app.providers import tts as _tts
from app.providers.llm import ChatModel

http = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=5))


def _pick(table, job, default):
    name = CFG[job].get("provider", default)
    if name not in table:
        raise SystemExit(f"config.yaml: {job}.provider must be one of {', '.join(table)} (got {name!r})")
    return table[name](CFG[job], http)


llm = ChatModel(CFG["llm"], http)
tts = _pick(_tts.PROVIDERS, "tts", "ming")
asr = _pick(_asr.PROVIDERS, "asr", "ming")
search = _pick(_search.PROVIDERS, "search", "searxng")
