@echo off
setlocal
echo === Building frontend ===
cd /d "%~dp0frontend" || exit /b 1

if not exist node_modules (
  echo Installing frontend dependencies...
  npm install || exit /b 1
)

REM vite.config.ts sets build.outDir = ../backend/frontend_build, so Vite writes
REM the build straight into the folder the backend serves. Do NOT wipe/copy that
REM folder afterwards -- doing so deletes the build that was just produced.
npm run build || exit /b 1

if not exist "%~dp0backend\frontend_build\index.html" (
  echo [ERROR] Build finished but backend\frontend_build\index.html is missing.
  echo         Check build.outDir in frontend\vite.config.ts.
  exit /b 1
)

echo.
echo Frontend build complete: backend\frontend_build
endlocal
