@echo off
setlocal
echo === Building backend EXE (PyInstaller) ===
cd /d "%~dp0backend" || exit /b 1

if not exist venv (
  echo Creating virtual environment...
  python -m venv venv || exit /b 1
)

call venv\Scripts\activate || exit /b 1

echo Installing backend dependencies...
python -m pip install -r requirements.txt || exit /b 1

echo Installing PyInstaller...
python -m pip install pyinstaller || exit /b 1

echo Packaging backend into single EXE...
pyinstaller --noconfirm --clean --onefile --name hrms-backend -p . app\run.py || exit /b 1

echo Backend build complete: backend\dist\hrms-backend.exe
endlocal

