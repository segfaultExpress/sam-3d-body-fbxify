@echo off
REM Run UI + worker locally for testing.
REM Uses repo-root\checkpoints by default (override with CHECKPOINTS_DIR).
REM If port 7444 is in use, set UI_PORT=7445 (then open http://localhost:7445).

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\.."
cd /d "%REPO_ROOT%"

REM Default checkpoints and cache to repo root (match standalone run.bat layout)
if not defined CHECKPOINTS_DIR set "CHECKPOINTS_DIR=%REPO_ROOT%\checkpoints"
if not defined CACHE_DIR set "CACHE_DIR=%REPO_ROOT%\cache"

if not exist "%CHECKPOINTS_DIR%\sam-3d-body-vith" (
  echo WARNING: checkpoints not found at %CHECKPOINTS_DIR%\sam-3d-body-vith
  echo Set CHECKPOINTS_DIR to your checkpoints path, e.g.:
  echo   set CHECKPOINTS_DIR=F:\sam-3d-body-fbxify\checkpoints
  echo.
)

if defined UI_PORT (echo UI will be at http://localhost:%UI_PORT%) else (echo UI will be at http://localhost:7444)
docker compose -f "%SCRIPT_DIR%docker-compose.yml" up --build
