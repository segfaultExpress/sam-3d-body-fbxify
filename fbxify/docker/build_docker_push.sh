#!/bin/bash
# Build and push fbxify-worker and fbxify-ui to Docker Hub.
# Reads USERNAME and VERSION from docker-build.config (same dir as this script).
# Command-line args override: ./build_docker_push.sh [USERNAME] [VERSION]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
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

echo "Building and pushing ${USERNAME}/fbxify-worker:${VERSION} and ${USERNAME}/fbxify-ui:${VERSION}"
echo

set -e

# Worker
echo "[1/4] Building fbxify-worker..."
docker build -f Dockerfile.worker -t "${USERNAME}/fbxify-worker:${VERSION}" .

echo "[2/4] Pushing fbxify-worker..."
docker push "${USERNAME}/fbxify-worker:${VERSION}"

# UI
echo "[3/4] Building fbxify-ui..."
docker build -f Dockerfile.ui -t "${USERNAME}/fbxify-ui:${VERSION}" .

echo "[4/4] Pushing fbxify-ui..."
docker push "${USERNAME}/fbxify-ui:${VERSION}"

echo
echo "Done. Images pushed: ${USERNAME}/fbxify-worker:${VERSION} and ${USERNAME}/fbxify-ui:${VERSION}"
