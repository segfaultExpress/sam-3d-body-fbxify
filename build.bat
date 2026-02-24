@echo off
REM Build FBXify: tries Docker first, then conda or system Python if Docker unavailable.
REM Run from repo root.

cd /d "%~dp0"

where docker >nul 2>&1
if %errorlevel% equ 0 (
  echo Building Docker image...
  call "%~dp0fbxify\docker\build_docker_standalone.bat" %*
  exit /b %errorlevel%
)

echo Docker not found.
set /p CHOICE="Create conda environment 'fbxify'? (y/n): "

if /i "%CHOICE%"=="y" (
  where conda >nul 2>&1
  if %errorlevel% equ 0 (
    echo Creating conda environment fbxify...
    conda create -n fbxify python=3.12 -y
    if %errorlevel% neq 0 (
      echo Conda create failed. Falling back to system Python.
      goto :python_fallback
    )
    echo Installing requirements...
    conda run -n fbxify pip install -r "%~dp0fbxify\requirements-full.txt"
    exit /b %errorlevel%
  ) else (
    echo Conda not found. Falling back to system Python.
    goto :python_fallback
  )
)

:python_fallback
where python >nul 2>&1
if %errorlevel% neq 0 (
  echo Python not found. Please install Python 3.12 or use Docker/Conda.
  exit /b 1
)
echo Installing requirements with system Python...
python -m pip install -r "%~dp0fbxify\requirements-full.txt"
exit /b %errorlevel%
