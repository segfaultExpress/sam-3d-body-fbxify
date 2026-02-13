@echo off
REM Build fbxify worker image (full GPU). Context = repo root.
cd /d "%~dp0..\.."
docker build -f fbxify/docker/Dockerfile.worker -t fbxify-worker .
