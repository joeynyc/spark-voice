#!/bin/sh
# First-time setup: checks Docker and the GPU, then writes .env with a fresh search secret.
set -e
cd "$(dirname "$0")/.."
command -v docker >/dev/null || { echo "Docker isn't installed."; exit 1; }
docker info 2>/dev/null | grep -qi nvidia || echo "Note: Docker's NVIDIA runtime wasn't found. DGX OS includes it; elsewhere install the NVIDIA Container Toolkit."
if [ -f .env ]; then
  echo ".env already exists; leaving it alone."
else
  umask 077
  sed "s/^SEARXNG_SECRET=.*/SEARXNG_SECRET=$(openssl rand -hex 32)/" .env.example > .env
  printf 'UID=%s\nGID=%s\n' "$(id -u)" "$(id -g)" >> .env   # the app writes data/ as you
  echo "Wrote .env (readable only by you)."
fi
mkdir -p data
echo "Next: docker compose --profile app --profile models up -d   (first start downloads about 25 GB of models)"
