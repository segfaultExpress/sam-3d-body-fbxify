#!/bin/bash
# Build fbxify UI image (slim, no GPU)
# Run from repo root. Dockerfile clones the repo into the image.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
docker build -f Dockerfile.ui -t fbxify-ui .
