# One per machine: GPU and memory stats for the under-the-hood views. nvidia-smi comes from the
# NVIDIA container runtime; run with the host's PID namespace so it can name other processes.
FROM python:3.12-slim
COPY agent.py /agent.py
ENV STATS_PORT=7796 STATS_BIND=127.0.0.1 NVIDIA_DRIVER_CAPABILITIES=utility
USER 65534:65534
CMD ["python", "/agent.py"]
