"""Loads config.yaml: which model does each job, where it runs, and the podcast cast.

Values may use ${VAR} or ${VAR:-default}; they're filled from the environment (and .env via compose).
"""
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PATH = Path(os.environ.get("SPARK_VOICE_CONFIG", ROOT / "config.yaml"))
DATA = Path(os.environ.get("SPARK_VOICE_DATA", ROOT / "data"))
VAR = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def expand(value):
    if isinstance(value, str):
        return VAR.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return value


def load():
    cfg = expand(yaml.safe_load(PATH.read_text()) or {})
    for job in ("llm", "tts", "asr", "search"):
        cfg.setdefault(job, {})
    cfg.setdefault("machine", "DGX Spark")
    # A node whose address is left empty (e.g. SPARK2 unset on a one-Spark setup) isn't there.
    cfg["nodes"] = [n for n in cfg.get("nodes") or [] if "://:" not in n.get("stats", "")]
    cfg.setdefault("assistant", {})
    cfg.setdefault("podcast", {})
    return cfg


CFG = load()


def badge():
    """The header badge, e.g. 'On-device · 2 × DGX Spark'."""
    n = len(CFG["nodes"]) or 1
    return f"On-device · {n} × {CFG['machine']}" if n > 1 else f"On-device · {CFG['machine']}"


def where():
    """How the assistant describes where it runs, e.g. 'two local NVIDIA DGX Spark machines'."""
    n = len(CFG["nodes"]) or 1
    words = {1: "one", 2: "two", 3: "three", 4: "four"}
    return f"{words.get(n, n)} local {CFG['machine']} machine{'s' if n > 1 else ''}"
