@echo off
REM Build and push fbxify-worker, fbxify-ui, and fbxify-standalone to Docker Hub.
REM Context = repo root (same as build_docker_worker/ui.bat).
REM Reads USERNAME and VERSION from docker-build.config (same dir as this script).
REM Command-line args override: build_docker_push.bat [USERNAME] [VERSION]

cd /d "%~dp0"
for /f "usebackq eol=# tokens=1,* delims==" %%a in ("docker-build.config") do set "%%a=%%b"
if "%~1" neq "" set USERNAME=%~1
if "%~2" neq "" set VERSION=%~2
if "%USERNAME%"=="" set USERNAME=mordommin94
if "%VERSION%"=="" set VERSION=0.1.4

echo Building and pushing fbxify-worker, fbxify-ui, fbxify-standalone (%VERSION% and :latest)
echo.

cd /d "%~dp0..\.."

REM Worker
echo [1/9] Building fbxify-worker...
docker build -f fbxify/docker/Dockerfile.worker -t %USERNAME%/fbxify-worker:%VERSION% .
if errorlevel 1 exit /b 1

echo [2/9] Pushing fbxify-worker:%VERSION%...
docker push %USERNAME%/fbxify-worker:%VERSION%
if errorlevel 1 exit /b 1

echo [3/9] Tagging and pushing fbxify-worker:latest...
docker tag %USERNAME%/fbxify-worker:%VERSION% %USERNAME%/fbxify-worker:latest
docker push %USERNAME%/fbxify-worker:latest
if errorlevel 1 exit /b 1

REM UI
echo [4/9] Building fbxify-ui...
docker build -f fbxify/docker/Dockerfile.ui -t %USERNAME%/fbxify-ui:%VERSION% .
if errorlevel 1 exit /b 1

echo [5/9] Pushing fbxify-ui:%VERSION%...
docker push %USERNAME%/fbxify-ui:%VERSION%
if errorlevel 1 exit /b 1

echo [6/9] Tagging and pushing fbxify-ui:latest...
docker tag %USERNAME%/fbxify-ui:%VERSION% %USERNAME%/fbxify-ui:latest
docker push %USERNAME%/fbxify-ui:latest
if errorlevel 1 exit /b 1

REM Standalone
echo [7/9] Building fbxify-standalone...
docker build -f fbxify/docker/Dockerfile -t %USERNAME%/fbxify-standalone:%VERSION% .
if errorlevel 1 exit /b 1

echo [8/9] Pushing fbxify-standalone:%VERSION%...
docker push %USERNAME%/fbxify-standalone:%VERSION%
if errorlevel 1 exit /b 1

echo [9/9] Tagging and pushing fbxify-standalone:latest...
docker tag %USERNAME%/fbxify-standalone:%VERSION% %USERNAME%/fbxify-standalone:latest
docker push %USERNAME%/fbxify-standalone:latest
if errorlevel 1 exit /b 1

echo.
echo Done. Images pushed: fbxify-worker, fbxify-ui, fbxify-standalone (%VERSION% and :latest)
