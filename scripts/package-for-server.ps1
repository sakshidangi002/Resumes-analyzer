# Package Softwiz (HRMS + Resume Analyzer) as a single-port server for Windows deployment.
#
# Output layout (mirrors the dev repo so app/main.py finds Resume API + UI):
#   deploy\SoftwizApp\
#     backend\               <- Resume Analyzer API (Python package)
#     frontend\              <- Resume Analyzer UI (static HTML)
#     chromadb\              <- vector store data (if present)
#     Resumes-analyzer\      <- (optional)
#     hrms\backend\          <- Attendance/HRMS app + built React SPA
#     requirements-all.txt   <- combined HRMS + Resume deps (single venv)
#     .env                   <- Resume Analyzer config (remote DB)
#     hrms\backend\.env      <- HRMS config (remote DB)
#     setup-server.bat       <- One-time venv + pip + alembic on target PC
#     run-server.bat         <- One-click start (port 5001)
#     SERVER-SETUP.txt       <- Step-by-step install for Windows server
#
# Usage from project root:
#   .\scripts\package-for-server.ps1            # build + copy + .env
#   .\scripts\package-for-server.ps1 -Zip       # also create SoftwizApp.zip
#   .\scripts\package-for-server.ps1 -SkipBuild # skip npm run build
#   .\scripts\package-for-server.ps1 -NoEnv     # omit .env (templates only)

param(
    [string]$OutDir = "",
    [switch]$SkipBuild,
    [switch]$Zip,
    [switch]$NoEnv
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

function Copy-DeployEnvFile {
    param(
        [string]$SourcePath,
        [string]$DestPath,
        [hashtable]$Replacements = @{}
    )
    $lines = if (Test-Path $SourcePath) {
        Get-Content $SourcePath -Encoding UTF8
    } else {
        @()
    }
    if ($lines.Count -eq 0) { return $false }
    $out = foreach ($line in $lines) {
        $updated = $line
        foreach ($key in $Replacements.Keys) {
            if ($line -match "^\s*$([regex]::Escape($key))\s*=") {
                $updated = "$key=$($Replacements[$key])"
            }
        }
        $updated
    }
    $dir = Split-Path -Parent $DestPath
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    Write-Utf8NoBom -Path $DestPath -Content (($out -join "`r`n") + "`r`n")
    return $true
}

function Write-Utf8NoBom {
    param([string]$Path, [string]$Content)
    $dir = Split-Path -Parent $Path
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($Path, $Content, $utf8)
}

function Write-DeployEnvTemplates {
    param([string]$OutDir)
    $hrmsEnv = @"
POSTGRES_HOST=YOUR_DB_SERVER_IP
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=YOUR_PASSWORD
POSTGRES_DB=Attendance_system
SECRET_KEY=change-to-long-random-string
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USE_TLS=true
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM_EMAIL=noreply@yourcompany.com
SMTP_FROM_NAME=Softwiz HRMS
HR_NOTIFICATION_EMAIL=
"@
    $rootEnv = @"
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@YOUR_DB_SERVER_IP:5432/Attendance_system
BASE_URL=http://127.0.0.1:5001
CHAT_MODEL=Qwen/Qwen2.5-1.5B-Instruct
IMAP_HOST=imap.gmail.com
IMAP_PORT=993
IMAP_SSL=1
IMAP_MAILBOX=INBOX
IMAP_USER=
IMAP_PASSWORD=
INDEED_STORAGE_STATE=storage_state.json
INDEED_HEADLESS=1
"@
    Write-Utf8NoBom -Path (Join-Path $OutDir "hrms\backend\.env") -Content ($hrmsEnv.Trim() + "`r`n")
    Write-Utf8NoBom -Path (Join-Path $OutDir ".env") -Content ($rootEnv.Trim() + "`r`n")
}

function Install-DeployEnvFiles {
    param([string]$OutDir)
    $hrmsSrc = Join-Path $HrmsBackend ".env"
    $hrmsDst = Join-Path $OutDir "hrms\backend\.env"
    $rootSrc = Join-Path $Root ".env"
    $rootDst = Join-Path $OutDir ".env"

    $hrmsOk = Copy-DeployEnvFile -SourcePath $hrmsSrc -DestPath $hrmsDst
    $rootOk = Copy-DeployEnvFile -SourcePath $rootSrc -DestPath $rootDst -Replacements @{
        "BASE_URL" = "http://127.0.0.1:5001"
        "INDEED_STORAGE_STATE" = "storage_state.json"
        "INDEED_HEADLESS" = "1"
    }

    if (-not $hrmsOk -or -not $rootOk) {
        Write-Warning "  Missing dev .env; writing templates (edit DB host/password on server)."
        Write-DeployEnvTemplates -OutDir $OutDir
    } else {
        Write-Host "      OK -> .env + hrms\backend\.env (UTF-8, no BOM)" -ForegroundColor Green
    }

    $storageSrc = Join-Path $Root "storage_state.json"
    if (Test-Path $storageSrc) {
        Copy-Item $storageSrc (Join-Path $OutDir "storage_state.json") -Force
    }
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

# Remove any .env copied by robocopy; we add deploy-ready .env in the next step.
Remove-Sensitive -Folder $OutDir

if ($NoEnv) {
    Write-Host "      Skipping .env (-NoEnv); copy .env.example manually on server." -ForegroundColor Gray
} else {
    Write-Host "      Adding .env files for server (remote DB)..." -ForegroundColor Yellow
    Install-DeployEnvFiles -OutDir $OutDir
}

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
# 4) setup-server.bat + run-server.bat
# ---------------------------------------------------------------------------
Write-Host "[4/5] Writing setup-server.bat and run-server.bat..." -ForegroundColor Yellow

$setupBat = @"
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
"@
Set-Content -Path (Join-Path $OutDir "setup-server.bat") -Value $setupBat -Encoding ASCII

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
echo   Login / HRMS     : http://localhost:%PORT%/
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
SOFTWIZ HRMS + RESUME ANALYZER - COPY, SETUP, RUN (Windows)
============================================================

This folder is ready to copy to any Windows PC. React is already built
(no npm on server). PostgreSQL can be on this PC or a remote server.

  http://<server>:5001/             ->  Attendance HRMS (login page)
  http://<server>:5001/resume/      ->  Resume Analyzer (Admin / HR only)

WHAT IS INCLUDED
----------------
  .env                 -> Resume Analyzer (DATABASE_URL, IMAP, etc.)
  hrms\backend\.env    -> HRMS (POSTGRES_*, SMTP, SECRET_KEY)
  setup-server.bat     -> One-time: venv + pip + DB migrations
  run-server.bat       -> Start the app every day

PREREQUISITES ON TARGET PC
--------------------------
1. Python 3.10+ in PATH (no Node.js needed)
2. PostgreSQL reachable (local or remote IP in .env)
3. Firewall: allow TCP port 5001 (or your custom port)

STEP 1 - COPY FOLDER
--------------------
Copy this entire SoftwizApp folder to e.g. C:\SoftwizApp\

STEP 2 - EDIT .env (only if DB host/password differ)
----------------------------------------------------
  C:\SoftwizApp\.env
  C:\SoftwizApp\hrms\backend\.env

Both files should use the SAME database server IP and password.
POSTGRES_HOST in hrms\backend\.env must match the host in DATABASE_URL.

STEP 3 - ONE-TIME SETUP (double-click)
--------------------------------------
  setup-server.bat

Or in PowerShell:
  cd C:\SoftwizApp
  .\setup-server.bat

STEP 4 - START APP (every time)
-------------------------------
  run-server.bat

Then open:  http://localhost:5001/

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
From any LAN machine:      http://<SERVER-IP>:5001/

ACCESS CONTROL
--------------
The Resume Analyzer (/resume/ and /resume-api/) is locked to users whose
JWT has the Admin or HR role. Employees and Managers cannot reach it
even if they type the URL directly (server returns 302 redirect or 403).

UPDATING LATER
--------------
1. On dev laptop:   .\scripts\package-for-server.ps1 -Zip
2. Copy to server; keep existing .venv if deps unchanged.
   Overwrite app files; merge .env if you edited secrets on server.
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
