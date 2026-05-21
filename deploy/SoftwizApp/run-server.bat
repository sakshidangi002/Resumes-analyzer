@echo off
title Softwiz Unified Server
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found at .venv\
    echo Run the install steps in SERVER-SETUP.txt first.
    pause
    exit /b 1
)

set "PORT=5001"
if not "%~1"=="" set "PORT=%~1"

set "PYTHONPATH=%~dp0hrms\backend;%~dp0;%PYTHONPATH%"
echo.
echo ============================================
echo   Softwiz Unified Server -- port %PORT%
echo ============================================
echo   Portal           : http://localhost:%PORT%/portal.html
echo   Attendance HRMS  : http://localhost:%PORT%/
echo   Resume Analyzer  : http://localhost:%PORT%/resume/      (Admin/HR only)
echo   HRMS API         : http://localhost:%PORT%/api
echo   Resume API       : http://localhost:%PORT%/resume-api   (Admin/HR only)
echo ============================================
echo.
cd hrms\backend
"%~dp0.venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port %PORT%
