#!/bin/bash
# Build fbxify worker image (full GPU)
# Context = fbxify/docker (so COPY worker-entrypoint.sh works). Dockerfile clones the repo into the image.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
docker build -f Dockerfile.worker -t fbxify-worker .
