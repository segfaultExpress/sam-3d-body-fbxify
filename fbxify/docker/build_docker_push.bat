@echo off
REM Build and push fbxify-worker and fbxify-ui to Docker Hub.
REM Reads USERNAME and VERSION from docker-build.config (same dir as this script).
REM Command-line args override: build_docker_push.bat [USERNAME] [VERSION]

cd /d "%~dp0"
for /f "usebackq eol=# tokens=1,* delims==" %%a in ("docker-build.config") do set "%%a=%%b"
if "%~1" neq "" set USERNAME=%~1
if "%~2" neq "" set VERSION=%~2
if "%USERNAME%"=="" set USERNAME=mordommin94
if "%VERSION%"=="" set VERSION=0.1.4

echo Building and pushing %USERNAME%/fbxify-worker:%VERSION% and %USERNAME%/fbxify-ui:%VERSION%
echo.

REM Worker
echo [1/4] Building fbxify-worker...
docker build -f Dockerfile.worker -t %USERNAME%/fbxify-worker:%VERSION% .
if errorlevel 1 exit /b 1

echo [2/4] Pushing fbxify-worker...
docker push %USERNAME%/fbxify-worker:%VERSION%
if errorlevel 1 exit /b 1

REM UI
echo [3/4] Building fbxify-ui...
docker build -f Dockerfile.ui -t %USERNAME%/fbxify-ui:%VERSION% .
if errorlevel 1 exit /b 1

echo [4/4] Pushing fbxify-ui...
docker push %USERNAME%/fbxify-ui:%VERSION%
if errorlevel 1 exit /b 1

echo.
echo Done. Images pushed: %USERNAME%/fbxify-worker:%VERSION% and %USERNAME%/fbxify-ui:%VERSION%
