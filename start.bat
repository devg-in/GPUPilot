@echo off
title GPUPilot Launcher
cd /d "%~dp0"

echo ========================================
echo           GPUPILOT STARTING
echo ========================================
echo.
echo Project: %CD%
echo.

echo [1/2] Starting Backend...
start "GPUPilot Backend" cmd /k "cd /d "%~dp0" && python -m uvicorn backend.app.main:app --reload --port 8000"

timeout /t 3 /nobreak >nul

echo [2/2] Checking Frontend dependencies...
if not exist "%~dp0frontend\node_modules" (
    echo Frontend dependencies not found. Running npm install...
    pushd "%~dp0frontend"
    call npm install
    if errorlevel 1 (
        echo ERROR: npm install failed. Check Node.js/npm and internet connectivity.
        popd
        pause
        exit /b 1
    )
    popd
)
start "GPUPilot Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo ========================================
echo Backend  : http://localhost:8000
echo Swagger  : http://localhost:8000/docs
echo Frontend : http://localhost:5173
echo ========================================
echo.
pause