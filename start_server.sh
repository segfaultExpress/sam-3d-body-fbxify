#!/bin/bash

echo "Starting SAM 3D Body FBX Server..."
echo "Running: fbxify/app.py (port: ${FBXIFY_UI_PORT:-7444})"
echo ""

export FBXIFY_UI_PORT="${FBXIFY_UI_PORT:-7444}"
exec python fbxify/app.py