@echo off
title Softwiz - First-time setup
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10 and add to PATH.
    pause
    exit /b 1
)

if not exist ".env" (
    echo [ERROR] Missing .env in this folder. Re-package with .\scripts\package-for-server.ps1
    pause
    exit /b 1
)
if not exist "hrms\backend\.env" (
    echo [ERROR] Missing hrms\backend\.env
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
if exist ".venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m pip --version >nul 2>&1
    if errorlevel 1 (
        echo       Broken .venv detected - removing and recreating...
        rmdir /s /q ".venv"
    )
)
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv
        pause
        exit /b 1
    )
)

echo [2/4] Installing Python packages (may take several minutes)...
set "PY=%~dp0.venv\Scripts\python.exe"
"%PY%" -m ensurepip --upgrade
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r "%~dp0requirements-all.txt"
if errorlevel 1 (
    echo [ERROR] pip install failed
    pause
    exit /b 1
)

echo [3/4] Downloading spaCy model (optional)...
"%PY%" -m spacy download en_core_web_sm

echo [4/4] Running database migrations...
set "PYTHONPATH=%~dp0hrms\backend;%~dp0"
cd /d "%~dp0hrms\backend"
"%PY%" -m alembic upgrade head
cd /d "%~dp0"

echo.
echo ============================================
echo   Setup complete.
echo   Next: double-click run-server.bat
echo ============================================
pause
