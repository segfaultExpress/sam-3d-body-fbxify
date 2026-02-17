#!/bin/bash
# Start Docker container (same clone location pattern as build scripts)
# Resolves REPO_DIR: if script is in repo use it, else use ../sam-3d-body-fbxify

REPO_URL="https://github.com/segfaultExpress/sam-3d-body-fbxify"
REPO_NAME="sam-3d-body-fbxify"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if docker is available
if ! command -v docker >/dev/null 2>&1; then
  echo "Error: 'docker' is not a recognized command."
  echo "You must run build_docker.bat or build_docker.sh first to build the image."
  exit 1
fi

# Check if sam-3d-3.12 image exists
if ! docker image inspect sam-3d-3.12 >/dev/null 2>&1; then
  echo "Error: Docker image 'sam-3d-3.12' does not exist."
  echo "You must run build_docker.bat or build_docker.sh first to build the image."
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

docker run --rm -it --gpus all --shm-size=8g \
  -e HF_TOKEN=your_hf_token_here \
  -p 7444:7444 \
  -v "$REPO_DIR":/workspace \
  -v "$REPO_DIR/cache/videt_checkpoint":/root/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692 \
  -v "$REPO_DIR/cache/hf_cache":/root/.cache/huggingface \
  -v "$REPO_DIR/cache/mhr_assets":/opt/venv/lib/python3.12/site-packages/assets \
  sam-3d-3.12 ./start_server.sh
