@echo off
REM Start Docker container (same clone location pattern as build scripts)
REM Resolves REPO_DIR: if script is in repo use it, else use ../sam-3d-body-fbxify
REM RUN_WITH_LOCAL=1 (from docker-build.config): mount local repo over /fbxify for dev without rebuilding.
set REPO_URL=https://github.com/segfaultExpress/sam-3d-body-fbxify
set REPO_NAME=sam-3d-body-fbxify
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Load config (RUN_WITH_LOCAL)
for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%SCRIPT_DIR%fbxify\docker\docker-build.config") do set "%%a=%%b"
if "%RUN_WITH_LOCAL%"=="" set RUN_WITH_LOCAL=1

REM Default checkpoints and cache (used when RUN_WITH_LOCAL=0)
if not defined CHECKPOINTS_DIR set "CHECKPOINTS_DIR=%SCRIPT_DIR%\checkpoints"
if not defined CACHE_DIR set "CACHE_DIR=%SCRIPT_DIR%\cache"

REM Check if docker is available
where docker >nul 2>&1
if %errorlevel% neq 0 (
  echo Error: 'docker' is not a recognized command.
  echo This command assumes you are using docker. If you would rather go without, see fbxify/Install.md for instructions, then run using 'python fbxify/app.py'
  exit /b 1
)

set "IMAGE_NAME=mordommin94/fbxify-standalone:latest"

REM Check if sam-3d-3.12 image exists
docker image inspect %IMAGE_NAME% >nul 2>&1
if %errorlevel% neq 0 (
  echo Error: Docker image '%IMAGE_NAME%' does not exist.
  echo You must run build_docker_standalone.bat or build_docker_standalone.sh first to build the image.
  exit /b 1
)

if exist "%SCRIPT_DIR%\.git" (
  set "REPO_DIR=%SCRIPT_DIR%"
) else (
  if not exist "%SCRIPT_DIR%\%REPO_NAME%" (
    echo Cloning %REPO_URL%...
    git clone %REPO_URL% "%SCRIPT_DIR%\%REPO_NAME%"
  )
  set "REPO_DIR=%SCRIPT_DIR%\%REPO_NAME%"
)

echo "Starting Docker Container..."
echo ""

if "%RUN_WITH_LOCAL%"=="1" (
  REM Mount full repo for local dev
  docker run --rm -it --gpus all --shm-size=8g ^
    -e HF_TOKEN=your_hf_token_here ^
    -e CHECKPOINTS_DIR=/fbxify/checkpoints ^
    -p 7444:7444 ^
    -v "%REPO_DIR%":/fbxify ^
    -v "%CACHE_DIR%\videt_checkpoint":/root/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692 ^
    -v "%CACHE_DIR%\hf_cache":/root/.cache/huggingface ^
    -v "%CACHE_DIR%\mhr_assets":/fbxify/cache/mhr_assets ^
    %IMAGE_NAME% ./start_server.sh
) else (
  REM Use image code; mount only checkpoints and cache
  docker run --rm -it --gpus all --shm-size=8g ^
    -e HF_TOKEN=your_hf_token_here ^
    -e CHECKPOINTS_DIR=/fbxify/checkpoints ^
    -p 7444:7444 ^
    -v "%CHECKPOINTS_DIR%":/fbxify/checkpoints:ro ^
    -v "%CACHE_DIR%\videt_checkpoint":/root/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692 ^
    -v "%CACHE_DIR%\hf_cache":/root/.cache/huggingface ^
    -v "%CACHE_DIR%\mhr_assets":/fbxify/cache/mhr_assets ^
    %IMAGE_NAME% ./start_server.sh
)
