@echo off
setlocal
echo === Building frontend + backend ===

call "%~dp0build-frontend.bat" || exit /b 1
call "%~dp0build-backend.bat" || exit /b 1

echo.
echo === Creating release folder ===
set RELEASE_DIR=%~dp0release
if exist "%RELEASE_DIR%" rmdir /s /q "%RELEASE_DIR%"
mkdir "%RELEASE_DIR%\backend" || exit /b 1

copy "%~dp0backend\dist\hrms-backend.exe" "%RELEASE_DIR%\backend\" >nul || exit /b 1
xcopy "%~dp0backend\frontend_build" "%RELEASE_DIR%\backend\frontend_build" /E /I /Y >nul || exit /b 1

echo Release created at: release\
echo - backend\hrms-backend.exe
echo - backend\frontend_build\
endlocal

