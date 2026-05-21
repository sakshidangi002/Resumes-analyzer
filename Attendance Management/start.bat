@echo off
title Attendance Management System

echo ============================================
echo   Attendance Management System - Startup
echo ============================================
echo.

cd /d "%~dp0"

:: --- Check if frontend build exists ---
if not exist "frontend\dist\index.html" (
    echo [!] Frontend build not found. Building now...
    echo.
    cd frontend
    call npm install
    call npm run build
    cd ..
    echo.
    echo [OK] Frontend built successfully.
    echo.
)

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
