@echo off
REM Build fbxify worker image (full GPU). Context = fbxify/docker (so COPY worker-entrypoint.sh works).
cd /d "%~dp0"
docker build -f Dockerfile.worker -t fbxify-worker .
