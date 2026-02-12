@echo off
REM Start Docker container (same clone location pattern as build scripts)
REM Resolves REPO_DIR: if script is in repo use it, else use ../sam-3d-body-fbxify
set REPO_URL=https://github.com/segfaultExpress/sam-3d-body-fbxify
set REPO_NAME=sam-3d-body-fbxify
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Check if docker is available
where docker >nul 2>&1
if %errorlevel% neq 0 (
  echo Error: 'docker' is not a recognized command.
  echo This command assumes you are using docker. If you would rather go without, see fbxify/Install.md for instructions, then run using 'python fbxify/app.py'
  exit /b 1
)

REM Check if sam-3d-3.12 image exists
docker image inspect sam-3d-3.12 >nul 2>&1
if %errorlevel% neq 0 (
  echo Error: Docker image 'sam-3d-3.12' does not exist.
  echo You must run build_docker.bat or build_docker.sh first to build the image.
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

docker run --rm -it --gpus all --shm-size=8g ^
  -e HF_TOKEN=your_hf_token_here ^
  -p 7444:7444 ^
  -v "%REPO_DIR%":/workspace ^
  -v "%REPO_DIR%\cache\videt_checkpoint":/root/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692 ^
  -v "%REPO_DIR%\cache\hf_cache":/root/.cache/huggingface ^
  -v "%REPO_DIR%\cache\mhr_assets":/opt/venv/lib/python3.12/site-packages/assets ^
  sam-3d-3.12 ./start_server.sh
