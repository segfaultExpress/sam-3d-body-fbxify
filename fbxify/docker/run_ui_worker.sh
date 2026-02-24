#!/bin/bash
# Run UI + worker locally for testing.
# Uses repo-root/checkpoints and repo-root/cache by default (override with CHECKPOINTS_DIR, CACHE_DIR).
# Open http://localhost:7444 when ready.
# RUN_WITH_LOCAL=1 (from docker-build.config): mount local repo over /fbxify for dev without rebuilding.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Load config (PULL_FROM_LOCAL, BRANCH, RUN_WITH_LOCAL)
if [ -f "$SCRIPT_DIR/docker-build.config" ]; then
  while IFS='=' read -r key value; do
    key="${key%%#*}"
    key="${key// /}"
    value="${value%$'\r'}"
    [ -n "$key" ] && export "$key=$value"
  done < "$SCRIPT_DIR/docker-build.config"
fi

export REPO_ROOT
export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$REPO_ROOT/checkpoints}"
export CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/cache}"
export PULL_FROM_LOCAL="${PULL_FROM_LOCAL:-0}"
export BRANCH="${BRANCH:-main}"
export RUN_WITH_LOCAL="${RUN_WITH_LOCAL:-0}"

if [ ! -d "$CHECKPOINTS_DIR/sam-3d-body-vith" ]; then
  echo "WARNING: checkpoints not found at $CHECKPOINTS_DIR/sam-3d-body-vith"
  echo "Set CHECKPOINTS_DIR to your checkpoints path, e.g.:"
  echo "  export CHECKPOINTS_DIR=$REPO_ROOT/checkpoints"
  echo ""
fi

COMPOSE_FILES="-f fbxify/docker/docker-compose.yml"
[ "$RUN_WITH_LOCAL" = "1" ] && COMPOSE_FILES="$COMPOSE_FILES -f fbxify/docker/docker-compose.local.yml"
docker compose $COMPOSE_FILES up
