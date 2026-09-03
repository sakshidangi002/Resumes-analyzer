@echo off
title Attendance Management System

echo ============================================
echo   Attendance Management System - Startup
echo ============================================
echo.

cd /d "%~dp0"

:: --- Check if frontend build exists ---
REM Vite writes its build to backend\frontend_build (see frontend\vite.config.ts),
REM which is also the folder app\main.py serves the SPA from -- check there.
if not exist "backend\frontend_build\index.html" (
    echo [!] Frontend build not found. Building now...
    echo.
    call "%~dp0build-frontend.bat" || (
        echo [ERROR] Frontend build failed. The UI will not load.
        pause
        exit /b 1
    )
    echo.
    echo [OK] Frontend built successfully.
    echo.
)

:: NOTE: The old standalone Face_detection service is no longer needed.
:: Face detection now runs in-process in the HRMS backend
:: (app/services/face_service.py, insightface). See archive/Face_detection/.

:: --- Start Backend (serves both API + Frontend) ---
echo [*] Starting server on port 5001...
echo.
echo     App URL  : http://localhost:5001
echo     API Docs : http://localhost:5001/docs
echo.
echo     To access from other machines on the same network,
echo     use this machine's IP address instead of localhost.
echo.
echo ============================================
echo   Press Ctrl+C to stop the server
echo ============================================

cd backend

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: Expose the repo root on PYTHONPATH so the unified server can import the
:: Resume Analyzer (backend/api.py) and serve its UI from frontend/.
set "PYTHONPATH=%cd%;%~dp0..;%PYTHONPATH%"

uvicorn app.main:app --host 0.0.0.0 --port 5001
