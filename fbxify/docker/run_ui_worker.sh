#!/bin/bash
# Run UI + worker locally for testing.
# Uses repo-root/checkpoints and repo-root/cache by default (override with CHECKPOINTS_DIR, CACHE_DIR).
# Open http://localhost:7444 when ready.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$REPO_ROOT/checkpoints}"
export CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/cache}"

if [ ! -d "$CHECKPOINTS_DIR/sam-3d-body-vith" ]; then
  echo "WARNING: checkpoints not found at $CHECKPOINTS_DIR/sam-3d-body-vith"
  echo "Set CHECKPOINTS_DIR to your checkpoints path, e.g.:"
  echo "  export CHECKPOINTS_DIR=$REPO_ROOT/checkpoints"
  echo ""
fi

docker compose -f fbxify/docker/docker-compose.yml up
