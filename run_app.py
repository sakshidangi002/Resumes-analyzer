"""
Single-server launcher for Softwiz (HRMS + Resume Analyzer).

The Attendance HRMS backend (Attendance Management/backend/app/main.py) now
also mounts the Resume Analyzer UI and API, so the entire product runs on
ONE process and ONE URL:

    http://<host>:<port>/              -> Attendance HRMS login / dashboard
    http://<host>:<port>/api/...       -> Attendance HRMS API
    http://<host>:<port>/resume/       -> Resume Analyzer UI
    http://<host>:<port>/resume-api/   -> Resume Analyzer API

Usage (PowerShell):
    python run_app.py                  # serves on 0.0.0.0:5001
    python run_app.py --port 8080      # custom port
    python run_app.py --host 127.0.0.1 # bind to loopback only
"""

import argparse
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


def get_lan_ip() -> str:
    """Return the machine's LAN IP address (best-effort, not 127.0.0.1)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def is_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the unified Softwiz server")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5001")),
                        help="HTTP port (default: 5001)")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"),
                        help="Bind host (default: 0.0.0.0 for LAN access)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open the browser automatically")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    hrms_backend = repo_root / "Attendance Management" / "backend"

    if not hrms_backend.exists():
        sys.exit(f"[error] HRMS backend folder not found: {hrms_backend}")

    lan_ip = get_lan_ip()
    port = args.port

    if is_port_open("127.0.0.1", port):
        sys.exit(f"[error] Port {port} is already in use. Stop the other process first.")

    print(f"\n{'=' * 60}")
    print(f"  Softwiz Unified Server")
    print(f"{'=' * 60}")
    print(f"  Login / HRMS     : http://{lan_ip}:{port}/")
    print(f"  Resume Analyzer  : http://{lan_ip}:{port}/resume/")
    print(f"  HRMS API         : http://{lan_ip}:{port}/api")
    print(f"  Resume API       : http://{lan_ip}:{port}/resume-api")
    print(f"{'=' * 60}\n")

    env = os.environ.copy()
    # Make sure the HRMS package (`app.main`) and the Resume Analyzer
    # package (`backend.api`) are both importable.
    existing_pp = env.get("PYTHONPATH", "")
    extra_pp = os.pathsep.join([str(hrms_backend), str(repo_root)])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (extra_pp, existing_pp) if p)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", args.host,
        "--port", str(port),
    ]

    if not args.no_browser:
        threading.Timer(
            2.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")
        ).start()

    proc = subprocess.Popen(cmd, cwd=hrms_backend, env=env)

    try:
        print("Server is running. Press Ctrl+C to stop.")
        while True:
            ret = proc.poll()
            if ret is not None:
                sys.exit(ret)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
