#!/usr/bin/env bash
# CHECKPOINTS_DIR and CACHE_DIR are configurable. Set them to wherever you mount your volumes.
# Same layout as worker; run.bat mount -v cache/mhr_assets:assets works as override.
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

# mhr reads site-packages/assets/lod1.fbx (Path(__file__).parent.parent / "assets").
# A stale symlink is replaced. A mount or a directory that already has lod1.fbx is left alone.
link_mhr_runtime_assets() {
  local assets_path="$1"
  local cache_assets="$2"
  mkdir -p "$cache_assets"
  if [ "$(readlink -f "$assets_path" 2>/dev/null || true)" = "$(readlink -f "$cache_assets")" ]; then
    return 0
  fi
  if (mountpoint -q "$assets_path" 2>/dev/null) || grep -q " ${assets_path} " /proc/mounts 2>/dev/null; then
    echo "mhr_assets: ${assets_path} is mounted; leaving it unchanged"
    return 0
  fi
  if [ -L "$assets_path" ]; then
    if [ -f "$assets_path/lod1.fbx" ]; then
      echo "mhr_assets: runtime symlink ${assets_path} already has lod1.fbx"
      return 0
    fi
    echo "mhr_assets: replacing stale symlink ${assets_path} -> ${cache_assets}"
    rm -f "$assets_path"
    ln -snf "$cache_assets" "$assets_path"
    return 0
  fi
  if [ -d "$assets_path" ]; then
    if [ -f "$assets_path/lod1.fbx" ]; then
      echo "mhr_assets: runtime directory ${assets_path} already has lod1.fbx"
      return 0
    fi
    cp -an "$assets_path/." "$cache_assets/" 2>/dev/null || true
    rm -rf "$assets_path"
    ln -snf "$cache_assets" "$assets_path"
    return 0
  fi
  mkdir -p "$(dirname "$assets_path")"
  ln -snf "$cache_assets" "$assets_path"
}
link_mhr_runtime_assets "/opt/venv/lib/python3.12/site-packages/assets" "$CACHE_DIR/mhr_assets"

exec "$@"
