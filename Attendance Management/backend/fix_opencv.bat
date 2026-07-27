@echo off
REM ============================================================
REM  fix_opencv.bat  –  Fix OpenCV FFmpeg/RTSP support
REM  Removes conflicting headless/contrib builds, installs the
REM  full opencv-python which ships with FFmpeg on Windows.
REM ============================================================
echo.
echo [1/4] Uninstalling conflicting OpenCV packages...
.\venv\Scripts\pip.exe uninstall -y opencv-python-headless opencv-contrib-python opencv-python 2>NUL
echo Done.

echo.
echo [2/4] Installing opencv-python (full build with FFmpeg + RTSP)...
.\venv\Scripts\pip.exe install "opencv-python>=4.8.0,<4.10.0"
echo Done.

echo.
echo [3/4] Installing scikit-image (image quality checks)...
.\venv\Scripts\pip.exe install "scikit-image>=0.21.0"
echo Done.

echo.
echo [4/4] Verifying FFmpeg support...
.\venv\Scripts\python.exe -c "import cv2; info=cv2.getBuildInformation(); ok='FFmpeg' in info; print('FFmpeg Support:', ok); exit(0 if ok else 1)"

if %ERRORLEVEL%==0 (
    echo.
    echo SUCCESS: OpenCV is correctly configured with FFmpeg/RTSP support.
) else (
    echo.
    echo WARNING: FFmpeg still not detected. You may need to install FFmpeg system-wide:
    echo   https://www.gyan.dev/ffmpeg/builds/ -- add bin/ to system PATH
)
echo.
pause
