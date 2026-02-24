#!/bin/bash
# Build and push fbxify-worker, fbxify-ui, and fbxify-standalone to Docker Hub.
# Context = repo root (same as build_docker_worker/ui.sh).
# Reads USERNAME and VERSION from docker-build.config (same dir as this script).
# Command-line args override: ./build_docker_push.sh [USERNAME] [VERSION]

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

echo "Building and pushing fbxify-worker, fbxify-ui, fbxify-standalone (${VERSION} and :latest)"
echo

set -e

cd "$REPO_ROOT"

# Worker
echo "[1/9] Building fbxify-worker..."
docker build -f fbxify/docker/Dockerfile.worker -t "${USERNAME}/fbxify-worker:${VERSION}" .

echo "[2/9] Pushing fbxify-worker:${VERSION}..."
docker push "${USERNAME}/fbxify-worker:${VERSION}"

echo "[3/9] Tagging and pushing fbxify-worker:latest..."
docker tag "${USERNAME}/fbxify-worker:${VERSION}" "${USERNAME}/fbxify-worker:latest"
docker push "${USERNAME}/fbxify-worker:latest"

# UI
echo "[4/9] Building fbxify-ui..."
docker build -f fbxify/docker/Dockerfile.ui -t "${USERNAME}/fbxify-ui:${VERSION}" .

echo "[5/9] Pushing fbxify-ui:${VERSION}..."
docker push "${USERNAME}/fbxify-ui:${VERSION}"

echo "[6/9] Tagging and pushing fbxify-ui:latest..."
docker tag "${USERNAME}/fbxify-ui:${VERSION}" "${USERNAME}/fbxify-ui:latest"
docker push "${USERNAME}/fbxify-ui:latest"

# Standalone
echo "[7/9] Building fbxify-standalone..."
docker build -f fbxify/docker/Dockerfile -t "${USERNAME}/fbxify-standalone:${VERSION}" .

echo "[8/9] Pushing fbxify-standalone:${VERSION}..."
docker push "${USERNAME}/fbxify-standalone:${VERSION}"

echo "[9/9] Tagging and pushing fbxify-standalone:latest..."
docker tag "${USERNAME}/fbxify-standalone:${VERSION}" "${USERNAME}/fbxify-standalone:latest"
docker push "${USERNAME}/fbxify-standalone:latest"

echo
echo "Done. Images pushed: fbxify-worker, fbxify-ui, fbxify-standalone (${VERSION} and :latest)"
