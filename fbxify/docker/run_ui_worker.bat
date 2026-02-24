@echo off
REM Run UI + worker locally for testing.
REM Uses repo-root\checkpoints by default (override with CHECKPOINTS_DIR).
REM If port 7444 is in use, set UI_PORT=7445 (then open http://localhost:7445).
REM RUN_WITH_LOCAL=1 (from docker-build.config): mount local repo over /fbxify for dev without rebuilding.

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%..\.."
cd /d "%REPO_ROOT%"

REM Load config (PULL_FROM_LOCAL, BRANCH, RUN_WITH_LOCAL)
for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%SCRIPT_DIR%docker-build.config") do set "%%a=%%b"
if "%PULL_FROM_LOCAL%"=="" set PULL_FROM_LOCAL=0
if "%BRANCH%"=="" set BRANCH=main
if "%RUN_WITH_LOCAL%"=="" set RUN_WITH_LOCAL=0

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
REM CACHEBUST=commit forces fresh clone; run_ui_worker sets it to HEAD automatically
for /f "delims=" %%i in ('git rev-parse HEAD 2^>nul') do set "CACHEBUST=%%i"

set "COMPOSE_FILES=-f %SCRIPT_DIR%docker-compose.yml"
if "%RUN_WITH_LOCAL%"=="1" set "COMPOSE_FILES=%COMPOSE_FILES% -f %SCRIPT_DIR%docker-compose.local.yml"
docker compose %COMPOSE_FILES% up
