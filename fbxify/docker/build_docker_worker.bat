@echo off
REM Build fbxify worker image (full GPU). Context = repo root.
REM Reads USERNAME and VERSION from docker-build.config (same dir as this script).
REM Command-line args override: build_docker_worker.bat [USERNAME] [VERSION]

cd /d "%~dp0"
for /f "usebackq eol=# tokens=1,* delims==" %%a in ("docker-build.config") do set "%%a=%%b"
if "%~1" neq "" set USERNAME=%~1
if "%~2" neq "" set VERSION=%~2
if "%USERNAME%"=="" set USERNAME=mordommin94
if "%VERSION%"=="" set VERSION=0.1.4

cd /d "%~dp0..\.."
docker build -f fbxify/docker/Dockerfile.worker -t %USERNAME%/fbxify-worker:%VERSION% .
