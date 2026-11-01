"""Stats agent: one per machine. GET /stats -> GPU load, power, temperature, memory, and GPU memory per process.

Runs with the host's PID namespace so nvidia-smi can name other containers' processes.
"""
import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("STATS_PORT", "7796"))
BIND = os.environ.get("STATS_BIND", "127.0.0.1")   # Spark 2 sets its link address so Spark 1 can read it


def smi(query, kind="gpu"):
    out = subprocess.run(["nvidia-smi", f"--query-{kind}={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=5).stdout
    return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def num(x):
    try:
        return float(x)
    except ValueError:
        return 0.0


def stats():
    util, power, temp = (num(x) for x in smi("utilization.gpu,power.draw,temperature.gpu")[0])
    mem = dict(line.split(":")[0:2] for line in open("/proc/meminfo") if line.startswith(("MemTotal", "MemAvailable")))
    total, avail = (int(mem[k].split()[0]) / 1048576 for k in ("MemTotal", "MemAvailable"))
    apps = [{"name": name.rsplit("/", 1)[-1], "gb": round(num(used) / 1024, 1)}
            for name, used in smi("process_name,used_memory", "compute-apps")]
    return {"util": util, "power": round(power, 1), "temp": temp, "mem_total": round(total, 1),
            "mem_used": round(total - avail, 1), "apps": apps}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] not in ("/stats", "/health"):
            self.send_error(404)
            return
        try:
            body, code = json.dumps(stats()).encode(), 200
        except Exception as e:  # no GPU visible, nvidia-smi missing...
            body, code = json.dumps({"error": str(e)}).encode(), 500
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
