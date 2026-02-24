#!/usr/bin/env bash
# CHECKPOINTS_DIR and CACHE_DIR are configurable. Set them to wherever you mount your volumes.
# PORT is configurable (default 8000). One volume mount can back both dirs.
set -e
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/fbxify/checkpoints}"
CACHE_DIR="${CACHE_DIR:-/fbxify/cache}"
export CHECKPOINTS_DIR
export CACHE_DIR

export HF_HOME="$CACHE_DIR/hf_cache"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR/hf_cache/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR/hf_cache/hub"

export TORCH_HOME="$CACHE_DIR/torch"
mkdir -p "$TORCH_HOME"

mkdir -p "$CHECKPOINTS_DIR"
mkdir -p "$CACHE_DIR/videt_checkpoint" \
         "$CACHE_DIR/hf_cache" \
         "$CACHE_DIR/mhr_assets"

# Symlink cache dirs so first-run downloads persist (same layout as run.bat).
mkdir -p /root/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h
ln -snf "$CACHE_DIR/videt_checkpoint" /root/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692
ln -snf "$CACHE_DIR/hf_cache" /root/.cache/huggingface

# mhr_assets: symlink assets -> CACHE_DIR/mhr_assets so mhr finds lod1.fbx etc.
# API startup downloads to CACHE_DIR/mhr_assets; symlink lets mhr find them.
ASSETS_PATH="/opt/venv/lib/python3.12/site-packages/assets"
if (mountpoint -q "$ASSETS_PATH" 2>/dev/null) || grep -q " $ASSETS_PATH " /proc/mounts 2>/dev/null; then
  : # Already mounted, nothing to do
elif [ -L "$ASSETS_PATH" ]; then
  : # Already a symlink, nothing to do
else
  [ -d "$ASSETS_PATH" ] && cp -a "$ASSETS_PATH/." "$CACHE_DIR/mhr_assets/" 2>/dev/null || true
  rm -rf "$ASSETS_PATH"
  ln -snf "$CACHE_DIR/mhr_assets" "$ASSETS_PATH"
fi

exec uvicorn fbxify.api:app --host 0.0.0.0 --port "${PORT:-8000}"
