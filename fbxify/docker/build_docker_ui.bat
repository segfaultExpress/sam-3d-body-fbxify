@echo off
REM Build fbxify UI image (slim, no GPU)
REM Run from repo root. Dockerfile clones the repo into the image.
cd /d "%~dp0"
docker build -f Dockerfile.ui -t fbxify-ui .
