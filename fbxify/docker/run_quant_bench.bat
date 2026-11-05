@echo off
REM Run the quantized-vs-fp32 x normal-vs-fast benchmark matrix in Docker, then the comparison.
REM
REM Usage (from repo root):
REM   set HF_TOKEN=hf_xxx
REM   fbxify\docker\run_quant_bench.bat                      (all 4 configs + compare)
REM   fbxify\docker\run_quant_bench.bat fp32-normal fp32-fast (subset)
REM   set QUANT_MODE=pt2 & fbxify\docker\run_quant_bench.bat  (torch.export instead of TensorRT)
REM Env: IMAGE, QUANT_MODE (trt|pt2), FRAMES, BENCH_DIR, HF_TOKEN, BUILD=1 to rebuild the image.
setlocal EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.."
set "REPO_DIR=%CD%"
popd
if "%IMAGE%"=="" set "IMAGE=mordommin94/fbxify-worker:quantized"
if "%QUANT_MODE%"=="" set "QUANT_MODE=trt"
if "%FRAMES%"=="" set "FRAMES=20"
if "%BENCH_DIR%"=="" set "BENCH_DIR=workspace/quant_bench"
if "%CACHE_DIR%"=="" set "CACHE_DIR=%REPO_DIR%\cache"
if "%CHECKPOINTS_DIR%"=="" set "CHECKPOINTS_DIR=%REPO_DIR%\checkpoints"

docker image inspect %IMAGE% >nul 2>&1
if %errorlevel% neq 0 set BUILD=1
if "%BUILD%"=="1" (
  echo Building %IMAGE%...
  docker build -f "%REPO_DIR%\fbxify\docker\Dockerfile.quantized" -t %IMAGE% "%REPO_DIR%"
  if errorlevel 1 exit /b 1
)

set "CONFIGS=%*"
if "%CONFIGS%"=="" set "CONFIGS=fp32-normal fp32-fast %QUANT_MODE%-normal %QUANT_MODE%-fast"

if not exist "%REPO_DIR%\%BENCH_DIR:/=\%" mkdir "%REPO_DIR%\%BENCH_DIR:/=\%"

for %%c in (%CONFIGS%) do (
  set "cfg=%%c"
  for /f "tokens=1,2 delims=-" %%a in ("%%c") do (
    set "backbone=%%a"
    set "mode=%%b"
  )
  set "extra="
  if "!mode!"=="fast" set "extra=--fast_inference"
  echo === !cfg! ^(backbone=!backbone!, !mode!^) ===
  docker run --rm --gpus all --shm-size=8g ^
    -e HF_TOKEN=%HF_TOKEN% ^
    -e CHECKPOINTS_DIR=/fbxify/checkpoints ^
    -e CACHE_DIR=/fbxify/cache ^
    -v "%REPO_DIR%":/fbxify ^
    -v "%CHECKPOINTS_DIR%":/fbxify/checkpoints ^
    -v "%CACHE_DIR%":/fbxify/cache ^
    -w /fbxify ^
    %IMAGE% python -m fbxify.cli_bench_quantized --backbone !backbone! --frames %FRAMES% --output_dir %BENCH_DIR% --config_name !cfg! !extra!
)

echo === compare ===
docker run --rm --gpus all --shm-size=8g ^
  -e CHECKPOINTS_DIR=/fbxify/checkpoints ^
  -e CACHE_DIR=/fbxify/cache ^
  -v "%REPO_DIR%":/fbxify ^
  -v "%CHECKPOINTS_DIR%":/fbxify/checkpoints ^
  -v "%CACHE_DIR%":/fbxify/cache ^
  -w /fbxify ^
  %IMAGE% python -m fbxify.cli_compare_quantized --bench_dir %BENCH_DIR% --baseline fp32-normal
echo Results: %REPO_DIR%\%BENCH_DIR:/=\%\compare\summary.json
endlocal
