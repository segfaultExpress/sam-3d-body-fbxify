@echo off
echo Starting SAM 3D Body FBX Server...
if not defined FBXIFY_UI_PORT set FBXIFY_UI_PORT=7444
echo Running: fbxify/app.py (port: %FBXIFY_UI_PORT%)
echo.

python fbxify/app.py