# Package Softwiz (HRMS + Resume Analyzer) as a single-port server for Windows deployment.
#
# Output layout (mirrors the dev repo so app/main.py finds Resume API + UI):
#   deploy\SoftwizApp\
#     backend\               <- Resume Analyzer API (Python package)
#     frontend\              <- Resume Analyzer UI (static HTML)
#     portal.html            <- Landing page
#     chromadb\              <- vector store data (if present)
#     Resumes-analyzer\      <- (optional)
#     hrms\backend\          <- Attendance/HRMS app + built React SPA
#     requirements-all.txt   <- combined HRMS + Resume deps (single venv)
#     run-server.bat         <- One-click start (port 5001)
#     SERVER-SETUP.txt       <- Step-by-step install for Windows server
#
# Usage from project root:
#   .\scripts\package-for-server.ps1            # build + copy
#   .\scripts\package-for-server.ps1 -Zip       # also create SoftwizApp.zip
#   .\scripts\package-for-server.ps1 -SkipBuild # skip npm run build

param(
    [string]$OutDir = "",
    [switch]$SkipBuild,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not $OutDir) {
    $OutDir = Join-Path $Root "deploy\SoftwizApp"
}

$HrmsBackend    = Join-Path $Root "Attendance Management\backend"
$HrmsFrontend   = Join-Path $Root "Attendance Management\frontend"
$ResumeBackend  = Join-Path $Root "backend"
$ResumeFrontend = Join-Path $Root "frontend"

Write-Host "=== Softwiz unified deployment package ===" -ForegroundColor Cyan
Write-Host "Source: $Root"
Write-Host "Output: $OutDir"
Write-Host ""

# ---------------------------------------------------------------------------
# 1) Build HRMS React app (output goes to Attendance Management\backend\frontend_build)
# ---------------------------------------------------------------------------
if (-not $SkipBuild) {
    Write-Host "[1/5] Building HRMS frontend..." -ForegroundColor Yellow
    Push-Location $HrmsFrontend
    npm run build
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw "npm run build failed" }
    Pop-Location
    Write-Host "      OK -> $HrmsBackend\frontend_build" -ForegroundColor Green
} else {
    Write-Host "[1/5] Skipping npm build (-SkipBuild)" -ForegroundColor Gray
}

# ---------------------------------------------------------------------------
# 2) Clean output folder
# ---------------------------------------------------------------------------
Write-Host "[2/5] Preparing output folder..." -ForegroundColor Yellow
if (Test-Path $OutDir) { Remove-Item $OutDir -Recurse -Force }
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

function Copy-Tree {
    param([string]$Src, [string]$Dst, [string[]]$ExcludeDirs = @())
    if (-not (Test-Path $Src)) {
        Write-Warning "  Skip (missing): $Src"
        return
    }
    New-Item -ItemType Directory -Path $Dst -Force | Out-Null
    # robocopy uses non-zero exit codes for normal success (1 = files copied).
    # Anything < 8 is success; >= 8 means real failure. Reset $LASTEXITCODE so
    # PowerShell's StrictMode/ErrorActionPreference doesn't see it as an error.
    robocopy $Src $Dst /E /NFL /NDL /NJH /NJS /nc /ns /np `
        /XD node_modules __pycache__ .venv venv .git dist .pytest_cache `
        /XF *.pyc *.pyo .DS_Store tsconfig.tsbuildinfo `
        | Out-Null
    $rc = $LASTEXITCODE
    if ($rc -ge 8) { throw "robocopy failed (exit code $rc) while copying $Src" }
    $global:LASTEXITCODE = 0
    if ($ExcludeDirs) {
        foreach ($d in $ExcludeDirs) {
            $p = Join-Path $Dst $d
            if (Test-Path $p) { Remove-Item $p -Recurse -Force -ErrorAction SilentlyContinue }
        }
    }
}

function Remove-Sensitive {
    param([string]$Folder)
    if (-not (Test-Path $Folder)) { return }
    Get-ChildItem $Folder -Recurse -Force -ErrorAction SilentlyContinue `
        | Where-Object { $_.Name -in @(".env") } `
        | Remove-Item -Force -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------------------
# 3) Copy application files (mirror dev layout so app.main can import everything)
# ---------------------------------------------------------------------------
Write-Host "[3/5] Copying files..." -ForegroundColor Yellow

# HRMS backend -> hrms\backend\  (so PROJECT_ROOT = SoftwizApp\)
$hrmsOut = Join-Path $OutDir "hrms\backend"
Copy-Tree -Src $HrmsBackend -Dst $hrmsOut

# Resume Analyzer Python package -> backend\
Copy-Tree -Src $ResumeBackend -Dst (Join-Path $OutDir "backend")

# Resume Analyzer static UI -> frontend\
Copy-Tree -Src $ResumeFrontend -Dst (Join-Path $OutDir "frontend")

# Portal landing page
Copy-Item (Join-Path $Root "portal.html") (Join-Path $OutDir "portal.html") -Force

# Vector store and other Resume Analyzer data folders (optional)
foreach ($folder in @("chromadb", "Resumes-analyzer")) {
    $src = Join-Path $Root $folder
    $dst = Join-Path $OutDir $folder
    if (Test-Path $src) {
        Copy-Tree -Src $src -Dst $dst
    } else {
        New-Item -ItemType Directory -Path $dst -Force | Out-Null
    }
}

# Strip dev .env files (real secrets must not ship to the server).
# .env.example is kept as a template.
Remove-Sensitive -Folder $OutDir

# Combined requirements (single venv; one bcrypt pin for chromadb + HRMS login)
function Get-RequirementLines {
    param(
        [string]$Path,
        [switch]$StopBeforeAttendanceBlock,
        [switch]$StripSharedAuth
    )
    $lines = Get-Content $Path -Encoding UTF8
    $out = New-Object System.Collections.Generic.List[string]
    foreach ($line in $lines) {
        if ($StopBeforeAttendanceBlock -and $line -match '^\s*#\s*Attendance Management') { break }
        if ($StripSharedAuth) {
            if ($line -match '^\s*python-jose') { continue }
            if ($line -match '^\s*passlib') { continue }
            if ($line -match '^\s*bcrypt') { continue }
        }
        $out.Add($line)
    }
    return $out
}

$resumeLines = Get-RequirementLines -Path (Join-Path $Root "requirements.txt") -StopBeforeAttendanceBlock
$hrmsLines   = Get-RequirementLines -Path (Join-Path $HrmsBackend "requirements.txt") -StripSharedAuth

$header = @(
    "# AUTO-GENERATED by scripts\package-for-server.ps1",
    "# Combined requirements for the unified single-port server.",
    "# Install once on the server:",
    "#   python -m venv .venv",
    "#   .\.venv\Scripts\pip install --upgrade pip",
    "#   .\.venv\Scripts\pip install -r requirements-all.txt",
    "",
    "# -------- Shared auth (HRMS login + chromadb) --------",
    "python-jose[cryptography]>=3.3.0",
    "bcrypt>=4.0.1,<5",
    "",
    "# -------- Resume Analyzer (ML, parsing) --------"
)
$footer = @(
    "",
    "# -------- Attendance / HRMS (auth, payroll) --------"
)
$combined = ($header + $resumeLines + $footer + $hrmsLines) -join "`r`n"
Set-Content -Path (Join-Path $OutDir "requirements-all.txt") -Value $combined -Encoding UTF8

# ---------------------------------------------------------------------------
# 4) run-server.bat (one-click start; same port logic as run_app.py)
# ---------------------------------------------------------------------------
Write-Host "[4/5] Writing run-server.bat..." -ForegroundColor Yellow

$runBat = @"
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
"@
Set-Content -Path (Join-Path $OutDir "run-server.bat") -Value $runBat -Encoding ASCII

# ---------------------------------------------------------------------------
# 5) SERVER-SETUP.txt
# ---------------------------------------------------------------------------
Write-Host "[5/5] Writing SERVER-SETUP.txt..." -ForegroundColor Yellow

$setup = @"
SOFTWIZ HRMS + RESUME ANALYZER - SERVER SETUP (Windows, single-port)
=====================================================================

The whole product (HRMS + Resume Analyzer + Portal) now runs as ONE
unified process on ONE port. Open ONE URL in the browser.

  http://<server>:5001/portal.html  ->  Landing portal
  http://<server>:5001/             ->  Attendance HRMS  (everyone)
  http://<server>:5001/resume/      ->  Resume Analyzer  (Admin / HR only)

Place this folder on the server as e.g. C:\SoftwizApp\ .

PREREQUISITES ON SERVER
-----------------------
1. Python 3.10 (same as dev machine), add to PATH
2. PostgreSQL 14+
3. NSSM (https://nssm.cc) -> copy nssm.exe to C:\Windows\System32\
4. Open Windows Firewall: ONE port only (default 5001).

DATABASES (PostgreSQL)
----------------------
Create two databases:
  - attendance_hrms      (HRMS)
  - Resume_analyzer      (Resume Analyzer)

CONFIG FILE 1  ->  C:\SoftwizApp\hrms\backend\.env
--------------------------------------------------
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=YOUR_PASSWORD
POSTGRES_DB=attendance_hrms
SECRET_KEY=change-to-long-random-string
SMTP_HOST=...
SMTP_PORT=587
SMTP_USER=...
SMTP_PASSWORD=...
SMTP_USE_TLS=true
SMTP_FROM_EMAIL=noreply@yourcompany.com
SMTP_FROM_NAME=HRMS

CONFIG FILE 2  ->  C:\SoftwizApp\.env    (Resume Analyzer DB)
-------------------------------------------------------------
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/Resume_analyzer

PYTHON SETUP (run ONCE on the server, in PowerShell)
----------------------------------------------------
cd C:\SoftwizApp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-all.txt

# Optional: pre-download the spaCy NER model used by the Resume Analyzer
python -m spacy download en_core_web_sm

# HRMS database migrations
`$env:PYTHONPATH = "C:\SoftwizApp\hrms\backend"
cd C:\SoftwizApp\hrms\backend
python -m alembic upgrade head

QUICK START (manual, no service)
--------------------------------
Double-click  C:\SoftwizApp\run-server.bat
Then open    http://localhost:5001/portal.html

PRODUCTION  -  Install as a Windows Service (NSSM)
--------------------------------------------------
ONE service runs the whole product:

  nssm install SoftwizApp ^
    "C:\SoftwizApp\.venv\Scripts\python.exe" ^
    "-m" "uvicorn" "app.main:app" "--host" "0.0.0.0" "--port" "5001"

  nssm set SoftwizApp AppDirectory       "C:\SoftwizApp\hrms\backend"
  nssm set SoftwizApp AppEnvironmentExtra ^
    "PYTHONPATH=C:\SoftwizApp\hrms\backend;C:\SoftwizApp"
  nssm set SoftwizApp Start SERVICE_AUTO_START

  nssm start SoftwizApp

FIREWALL
--------
netsh advfirewall firewall add rule name="SoftwizApp" ^
    protocol=TCP dir=in localport=5001 action=allow

VERIFY
------
From the server itself:    http://localhost:5001/health   -> {"status":"ok"}
From any LAN machine:      http://<SERVER-IP>:5001/portal.html

ACCESS CONTROL
--------------
The Resume Analyzer (/resume/ and /resume-api/) is locked to users whose
JWT has the Admin or HR role. Employees and Managers cannot reach it
even if they type the URL directly (server returns 302 redirect or 403).

UPDATING LATER
--------------
1. On the dev laptop:   .\scripts\package-for-server.ps1
2. Upload deploy\SoftwizApp to the server, overwriting the old folder
   (keep the existing .venv and .env files).
3. On the server:
     nssm restart SoftwizApp
   If the HRMS database schema changed:
     cd C:\SoftwizApp\hrms\backend
     `$env:PYTHONPATH = "C:\SoftwizApp\hrms\backend"
     ..\..\.venv\Scripts\python -m alembic upgrade head

TROUBLESHOOTING
---------------
* "ModuleNotFoundError: No module named 'jose'" or similar
  -> The venv is missing deps. Re-run:
       cd C:\SoftwizApp
       .\.venv\Scripts\pip install -r requirements-all.txt

* "Port 5001 is already in use"
  -> Stop the old uvicorn process or run on another port:
       run-server.bat 5002

* Browser shows old "8501" / "8001" / blank Resume page
  -> Hard refresh (Ctrl+Shift+R). The compiled bundle was updated.
"@

Set-Content -Path (Join-Path $OutDir "SERVER-SETUP.txt") -Value $setup -Encoding UTF8

# ---------------------------------------------------------------------------
# Optional ZIP
# ---------------------------------------------------------------------------
if ($Zip) {
    $zipPath = "$OutDir.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Write-Host ""
    Write-Host "Creating ZIP (this can take a minute for large bundles)..." -ForegroundColor Yellow
    Compress-Archive -Path $OutDir -DestinationPath $zipPath -Force
    $sizeMb = "{0:N1}" -f ((Get-Item $zipPath).Length / 1MB)
    Write-Host "ZIP created: $zipPath ($sizeMb MB)" -ForegroundColor Green
}

Write-Host ""
Write-Host "DONE. Upload this folder to your server:" -ForegroundColor Green
Write-Host "  $OutDir"
Write-Host ""
Write-Host "Then read SERVER-SETUP.txt on the server for install steps." -ForegroundColor Cyan
