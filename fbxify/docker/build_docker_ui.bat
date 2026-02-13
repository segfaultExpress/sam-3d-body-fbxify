@echo off
REM Build fbxify UI image (slim, no GPU). Context = repo root.
cd /d "%~dp0..\.."
docker build -f fbxify/docker/Dockerfile.ui -t fbxify-ui .
