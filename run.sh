#!/bin/bash
# Start Docker container (same clone location pattern as build scripts)
# Resolves REPO_DIR: if script is in repo use it, else use ../sam-3d-body-fbxify
# RUN_WITH_LOCAL=1 (from docker-build.config): mount local repo over /fbxify for dev without rebuilding.

REPO_URL="https://github.com/segfaultExpress/sam-3d-body-fbxify"
REPO_NAME="sam-3d-body-fbxify"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load config (RUN_WITH_LOCAL)
if [ -f "$SCRIPT_DIR/fbxify/docker/docker-build.config" ]; then
  while IFS='=' read -r key value; do
    key="${key%%#*}"
    key="${key// /}"
    value="${value%$'\r'}"
    [ -n "$key" ] && export "$key=$value"
  done < "$SCRIPT_DIR/fbxify/docker/docker-build.config"
fi
RUN_WITH_LOCAL="${RUN_WITH_LOCAL:-1}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$SCRIPT_DIR/checkpoints}"
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/cache}"

# Check if docker is available
if ! command -v docker >/dev/null 2>&1; then
  echo "Error: 'docker' is not a recognized command."
  echo "You must run build_docker.bat or build_docker.sh first to build the image."
  exit 1
fi

# Check if fbxify-standalone image exists
IMAGE_NAME="${IMAGE_NAME:-mordommin94/fbxify-standalone:latest}"
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "Error: Docker image '$IMAGE_NAME' does not exist."
  echo "You must run build_docker_standalone.bat or build_docker_standalone.sh first to build the image."
  exit 1
fi

if [ -d "$SCRIPT_DIR/.git" ]; then
  REPO_DIR="$SCRIPT_DIR"
else
  if [ ! -d "$SCRIPT_DIR/$REPO_NAME" ]; then
    echo "Cloning $REPO_URL..."
    git clone "$REPO_URL" "$SCRIPT_DIR/$REPO_NAME"
  fi
  REPO_DIR="$SCRIPT_DIR/$REPO_NAME"
fi

echo "Starting Docker Container..."
echo ""

if [ "$RUN_WITH_LOCAL" = "1" ]; then
  # Mount full repo for local dev
  docker run --rm -it --gpus all --shm-size=8g \
    -e HF_TOKEN=your_hf_token_here \
    -p 7444:7444 \
    -v "$REPO_DIR":/fbxify \
    -v "$REPO_DIR/cache/videt_checkpoint":/root/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692 \
    -v "$REPO_DIR/cache/hf_cache":/root/.cache/huggingface \
    -v "$REPO_DIR/cache/mhr_assets":/fbxify/cache/mhr_assets \
    "$IMAGE_NAME" ./start_server.sh
else
  # Use image code; mount only checkpoints and cache
  docker run --rm -it --gpus all --shm-size=8g \
    -e HF_TOKEN=your_hf_token_here \
    -e CHECKPOINTS_DIR=/fbxify/checkpoints \
    -p 7444:7444 \
    -v "$CHECKPOINTS_DIR":/fbxify/checkpoints:ro \
    -v "$CACHE_DIR/videt_checkpoint":/root/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692 \
    -v "$CACHE_DIR/hf_cache":/root/.cache/huggingface \
    -v "$CACHE_DIR/mhr_assets":/fbxify/cache/mhr_assets \
    "$IMAGE_NAME" ./start_server.sh
fi
