@echo off
REM Run UI + worker locally for testing.
REM Requires: checkpoints at repo-root/checkpoints/sam-3d-body-vith (or set CHECKPOINTS_DIR)
REM Open http://localhost:7444 when ready.

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\.."
cd /d "%REPO_ROOT%"

if not exist "checkpoints\sam-3d-body-vith" (
  echo WARNING: checkpoints not found at %REPO_ROOT%\checkpoints\sam-3d-body-vith
  echo Set CHECKPOINTS_DIR to your checkpoints path, e.g.:
  echo   set CHECKPOINTS_DIR=F:\path\to\checkpoints
  echo.
)

docker compose -f fbxify/docker/docker-compose.yml up
