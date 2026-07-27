@echo off
REM ============================================================
REM  Start Softwiz (HRMS + Resume Analyzer) on port 5002.
REM  Uses the project's .venv Python so ALL dependencies are
REM  present (insightface, ultralytics, apscheduler, transformers,
REM  chromadb...). Do NOT launch with a bare `python run_app.py`,
REM  which may pick the system Python and break face recognition
REM  and the resume database/email import.
REM ============================================================
cd /d "%~dp0"
".venv\Scripts\python.exe" run_app.py --port 5002
pause
