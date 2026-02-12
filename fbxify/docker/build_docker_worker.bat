@echo off
REM Build fbxify worker image (full GPU)
REM Run from repo root. Dockerfile clones the repo into the image.
cd /d "%~dp0"
docker build -f Dockerfile.worker -t fbxify-worker .
