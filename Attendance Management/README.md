# Attendance Management System â€“ Deployment Guide

## Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚         FastAPI (port 5001)                â”‚
â”‚                                            â”‚
â”‚  /api/*   â†’  REST API (Python backend)     â”‚
â”‚  /*       â†’  React SPA (frontend/dist/)    â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

**One process. One port. No separate frontend server needed.**

---

## Quick Start (Any Windows Machine)

### Step 1 â€“ Prerequisites (install once)

- [Python 3.11+](https://python.org/downloads) â€” check "Add to PATH" during install
- [Node.js 18+](https://nodejs.org) â€” for building the frontend
- [PostgreSQL](https://www.postgresql.org/download/windows/) â€” database

### Step 2 â€“ Configure the Database

Edit `backend\.env` and fill in your PostgreSQL credentials:

```
DATABASE_URL=postgresql://username:password@localhost:5432/attendance_db
SECRET_KEY=your-secret-key-here
```

### Step 3 â€“ Install Backend Dependencies

```bat
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 4 â€“ Run Database Migrations

```bat
cd backend
venv\Scripts\activate
alembic upgrade head
```

### Step 5 â€“ Build the Frontend (first time only)

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

### Step 6 â€“ Start the Application

Double-click **`start.bat`** in the root folder. It starts the HRMS server on port `5001` and the face-detection service on port `8000`.

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

**Terminal 1 â€“ Backend:**
```bat
cd backend
venv\Scripts\activate
set PORT=5001
set RELOAD=true
python -m app.run
```

**Terminal 2 â€“ Frontend (dev server with hot reload):**
```bat
cd frontend
npm run dev
```
Then open: **http://localhost:5173** (proxies HRMS API to port `5001` and face-registration calls to `8000`)

---

## File Structure

```
Attendance Management/
â”œâ”€â”€ start.bat              â† Double-click to launch
â”œâ”€â”€ frontend/
â”‚   â”œâ”€â”€ dist/              â† Vite build output
â”‚   â”œâ”€â”€ src/               â† React source code
â”‚   â””â”€â”€ package.json
â””â”€â”€ backend/
    â”œâ”€â”€ frontend_build/    â† Copy of dist/ (served by FastAPI in production)
    â”œâ”€â”€ app/
    â”‚   â”œâ”€â”€ main.py        â† FastAPI entry point (serves frontend + API)
    â”‚   â””â”€â”€ ...
    â”œâ”€â”€ requirements.txt
    â””â”€â”€ .env               â† Database credentials (DO NOT commit)
```
