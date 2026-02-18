#!/bin/bash
# Build fbxify standalone image (full GPU)
# Reads USERNAME and VERSION from docker-build.config (same dir as this script).
# Command-line args override: ./build_docker_standalone.sh [USERNAME] [VERSION]

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

cd "$REPO_ROOT"
docker build -f fbxify/docker/Dockerfile -t "${USERNAME}/sam-3d-body:${VERSION}" .
