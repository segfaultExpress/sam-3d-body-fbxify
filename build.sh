#!/bin/bash
# Build FBXify: tries Docker first, then conda or system Python if Docker unavailable.
# Run from repo root.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if command -v docker >/dev/null 2>&1; then
  echo "Building Docker image..."
  exec "$SCRIPT_DIR/fbxify/docker/build_docker_standalone.sh" "$@"
fi

echo "Docker not found."
read -p "Create conda environment 'fbxify'? (y/n): " CHOICE

if [ "${CHOICE,,}" = "y" ] || [ "${CHOICE,,}" = "yes" ]; then
  if command -v conda >/dev/null 2>&1; then
    echo "Creating conda environment fbxify..."
    if ! conda create -n fbxify python=3.12 -y; then
      echo "Conda create failed. Falling back to system Python."
    else
      echo "Installing requirements..."
      conda run -n fbxify pip install -r "$SCRIPT_DIR/fbxify/requirements-full.txt"
      exit $?
    fi
  else
    echo "Conda not found. Falling back to system Python."
  fi
fi

# Python fallback
PYTHON=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
fi

if [ -z "$PYTHON" ]; then
  echo "Python not found. Please install Python 3.12 or use Docker/Conda."
  exit 1
fi

echo "Installing requirements with system Python..."
$PYTHON -m pip install -r "$SCRIPT_DIR/fbxify/requirements-full.txt"
exit $?
