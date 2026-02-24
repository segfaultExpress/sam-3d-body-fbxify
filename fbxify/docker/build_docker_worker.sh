#!/bin/bash
# Build fbxify worker image (full GPU)
# Reads USERNAME and VERSION from docker-build.config (same dir as this script).
# Command-line args override: ./build_docker_worker.sh [USERNAME] [VERSION]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# Load config (lines starting with # are skipped)
if [ -f docker-build.config ]; then
  while IFS='=' read -r key value; do
    key="${key%%#*}"
    key="${key// /}"
    value="${value%$'\r'}"
    [ -n "$key" ] && export "$key=$value"
  done < docker-build.config
fi

USERNAME="${1:-$USERNAME}"
VERSION="${2:-$VERSION}"
USERNAME="${USERNAME:-mordommin94}"
VERSION="${VERSION:-0.1.4}"
PULL_FROM_LOCAL="${PULL_FROM_LOCAL:-0}"
BRANCH="${BRANCH:-main}"

cd "$REPO_ROOT"
docker build -f fbxify/docker/Dockerfile.worker \
  --build-arg PULL_FROM_LOCAL="${PULL_FROM_LOCAL}" \
  --build-arg BRANCH="${BRANCH}" \
  -t "${USERNAME}/fbxify-worker:${VERSION}" .
