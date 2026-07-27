@echo off
setlocal
echo === Building frontend ===
cd /d "%~dp0frontend" || exit /b 1

if not exist node_modules (
  echo Installing frontend dependencies...
  npm install || exit /b 1
)

npm run build || exit /b 1
echo.
echo Copying Vite dist to backend\frontend_build\ ...
if exist "%~dp0backend\frontend_build" rmdir /s /q "%~dp0backend\frontend_build"
mkdir "%~dp0backend\frontend_build" || exit /b 1
xcopy "%~dp0frontend\dist" "%~dp0backend\frontend_build" /E /I /Y >nul || exit /b 1
echo Frontend build complete: frontend\dist
echo Integrated copy: backend\frontend_build
endlocal

