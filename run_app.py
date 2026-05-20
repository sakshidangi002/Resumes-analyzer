import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
import http.server
import socketserver
from pathlib import Path
import argparse


def get_lan_ip() -> str:
    """Return the machine's LAN IP address (not 127.0.0.1)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def is_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    """Best-effort check for an already-running service on the target port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def main() -> None:
    """
    Start FastAPI backend, static frontend server, and Attendance backend.
    All servers bind to 0.0.0.0 so other machines on the network can connect.
    API_URL is set to the machine's LAN IP so browser API calls work remotely.
    """
    parser = argparse.ArgumentParser(description="Start the Resume Analyzer app")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8001")), help="Resume API backend port")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Bind host (default: 0.0.0.0 for LAN access)")
    parser.add_argument("--attendance-port", type=int, default=5001, help="Attendance backend port")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    bind_host = args.host
    port = str(args.port)
    attendance_port = str(args.attendance_port)

    # Detect LAN IP so browser-side API calls reach the correct address
    lan_ip = get_lan_ip()
    network_base = f"http://{lan_ip}:{port}"

    child_env = os.environ.copy()
    # Override API_URL / BASE_URL with network IP so browser requests work from other devices
    child_env["API_URL"]  = os.getenv("API_URL")  or network_base
    child_env["BASE_URL"] = os.getenv("BASE_URL") or network_base

    print(f"\n{'='*55}")
    print(f"  Attendance & HRMS Console — Network Access URLs")
    print(f"{'='*55}")
    print(f"  Attendance & HRMS: http://{lan_ip}:{attendance_port}/")
    print(f"  Resume Analyzer:   http://{lan_ip}:8501/")
    print(f"  Resume API:        http://{lan_ip}:{port}/")
    print(f"  Attendance API:    http://{lan_ip}:{attendance_port}/")
    print(f"{'='*55}\n")

    # --- Resume API Backend ---
    backend_cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.api:app",
        "--host", bind_host,
        "--port", port,
    ]

    # --- Attendance Backend ---
    attendance_backend_cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", bind_host,
        "--port", attendance_port,
    ]
    attendance_dir = root / "Attendance Management" / "backend"

    backend_proc = None
    if is_port_open("127.0.0.1", int(port)):
        print(f"[resume-api] Port {port} already in use, reusing existing backend.")
    else:
        try:
            backend_proc = subprocess.Popen(backend_cmd, cwd=root, env=child_env)
        except OSError as exc:
            print(f"[resume-api] Could not start on port {port}: {exc}. Reusing existing server if available.")

    attendance_proc = None
    if attendance_dir.exists():
        attendance_env = child_env.copy()
        attendance_env["PYTHONPATH"] = str(attendance_dir)
        if is_port_open("127.0.0.1", int(attendance_port)):
            print(f"[attendance-api] Port {attendance_port} already in use, reusing existing backend.")
        else:
            try:
                attendance_proc = subprocess.Popen(attendance_backend_cmd, cwd=attendance_dir, env=attendance_env)
            except OSError as exc:
                print(f"[attendance-api] Could not start on port {attendance_port}: {exc}. Reusing existing server if available.")

    # --- Portal static file server (bind to all interfaces) ---
    PORTAL_PORT = 8000

    def serve_portal():
        class SilentHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass  # suppress per-request log spam

        if is_port_open("127.0.0.1", PORTAL_PORT):
            print(f"[portal] Port {PORTAL_PORT} already in use, reusing existing server.")
            return
        try:
            with socketserver.TCPServer(("", PORTAL_PORT), SilentHandler) as httpd:
                httpd.serve_forever()
        except OSError as exc:
            print(f"[portal] Could not bind to port {PORTAL_PORT}: {exc}. Reusing existing server if available.")

    threading.Thread(target=serve_portal, daemon=True).start()

    # --- Resume Analyzer frontend static server ---
    ANALYZER_PORT = 8501

    def serve_analyzer():
        class AnalyzerHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(root / "frontend"), **kwargs)
            def log_message(self, fmt, *args):
                pass  # suppress per-request log spam

        if is_port_open("127.0.0.1", ANALYZER_PORT):
            print(f"[resume-analyzer] Port {ANALYZER_PORT} already in use, reusing existing server.")
            return
        try:
            with socketserver.TCPServer(("", ANALYZER_PORT), AnalyzerHandler) as httpd:
                httpd.serve_forever()
        except OSError as exc:
            print(f"[resume-analyzer] Could not bind to port {ANALYZER_PORT}: {exc}. Reusing existing server if available.")

    threading.Thread(target=serve_analyzer, daemon=True).start()

    # Open browser on local machine after short delay
    threading.Timer(2.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORTAL_PORT}/portal.html")).start()

    try:
        print("Both backends are running. Press Ctrl+C to terminate.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTerminating servers...")
    finally:
        if backend_proc:
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=10)
            except Exception:
                backend_proc.kill()

        if attendance_proc:
            attendance_proc.terminate()
            try:
                attendance_proc.wait(timeout=10)
            except Exception:
                attendance_proc.kill()


if __name__ == "__main__":
    main()
