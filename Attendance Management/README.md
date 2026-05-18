# Attendance Management System – Deployment Guide

## Architecture

```
┌────────────────────────────────────────────┐
│         FastAPI (port 5001)                │
│                                            │
│  /api/*   →  REST API (Python backend)     │
│  /*       →  React SPA (frontend/dist/)    │
└────────────────────────────────────────────┘
```

**One process. One port. No separate frontend server needed.**

---

## Quick Start (Any Windows Machine)

### Step 1 – Prerequisites (install once)

- [Python 3.11+](https://python.org/downloads) — check "Add to PATH" during install
- [Node.js 18+](https://nodejs.org) — for building the frontend
- [PostgreSQL](https://www.postgresql.org/download/windows/) — database

### Step 2 – Configure the Database

Edit `backend\.env` and fill in your PostgreSQL credentials:

```
DATABASE_URL=postgresql://username:password@localhost:5432/attendance_db
SECRET_KEY=your-secret-key-here
```

### Step 3 – Install Backend Dependencies

```bat
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 4 – Run Database Migrations

```bat
cd backend
venv\Scripts\activate
alembic upgrade head
```

### Step 5 – Build the Frontend (first time only)

**Recommended (copies the build into the backend for a single server):** run from project root:

```bat
.\build-frontend.bat
```

This creates `frontend\dist\` and copies it to **`backend\frontend_build\`** (served by FastAPI).

Or manually:

```bat
cd frontend
npm install
npm run build
xcopy /E /I /Y dist ..\backend\frontend_build
```

### Step 6 – Start the Application

Double-click **`start.bat`** in the root folder.

Or manually:
```bat
cd backend
venv\Scripts\activate
set PORT=5001
python -m app.run
```

Then open: **http://localhost:5001**

---

## Accessing from Other Machines

1. Find this machine's IP address: run `ipconfig` in Command Prompt
2. On another PC (same network), open: `http://<this-machine-ip>:5001`
3. Make sure Windows Firewall allows port **5001**

To allow port 5001 through Windows Firewall:
```
netsh advfirewall firewall add rule name="HRMS App" protocol=TCP dir=in localport=5001 action=allow
```

---

## Development Mode (with hot reload)

Run two terminals:

**Terminal 1 – Backend:**
```bat
cd backend
venv\Scripts\activate
set PORT=5001
set RELOAD=true
python -m app.run
```

**Terminal 2 – Frontend (dev server with hot reload):**
```bat
cd frontend
npm run dev
```
Then open: **http://localhost:5173** (proxies API to port 5001)

---

## File Structure

```
Attendance Management/
├── start.bat              ← Double-click to launch
├── frontend/
│   ├── dist/              ← Vite build output
│   ├── src/               ← React source code
│   └── package.json
└── backend/
    ├── frontend_build/    ← Copy of dist/ (served by FastAPI in production)
    ├── app/
    │   ├── main.py        ← FastAPI entry point (serves frontend + API)
    │   └── ...
    ├── requirements.txt
    └── .env               ← Database credentials (DO NOT commit)
```
