import os
import subprocess
import sys
from pathlib import Path
import argparse


def main() -> None:
    """
    Start FastAPI backend, Streamlit frontend, and Attendance backend together.
    - Backend: uvicorn backend.api:app --host HOST --port PORT
      Use HOST=0.0.0.0 to allow access from other machines by IP; set BASE_URL/API_URL
      to the URL clients use (e.g. http://192.168.x.x:8501) so resume links work.
    - Frontend: streamlit run frontend/app.py
    - Attendance: uvicorn app.main:app --host HOST --port ATTENDANCE_PORT
    """
    parser = argparse.ArgumentParser(description="Start the Resume Analyzer app")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8501")), help="Backend port")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"), help="Backend host")
    parser.add_argument("--attendance-port", type=int, default=5001, help="Attendance backend port")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    host = args.host
    port = str(args.port)
    attendance_port = str(args.attendance_port)

    child_env = os.environ.copy()
    default_app_url = f"http://127.0.0.1:{port}"
    child_env["API_URL"] = os.getenv("API_URL") or default_app_url
    child_env["BASE_URL"] = os.getenv("BASE_URL") or default_app_url

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.api:app",
        "--host",
        host,
        "--port",
        port,
    ]
    frontend_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(root / "frontend" / "app.py"),
        "--server.headless",
        "true"
    ]
    
    attendance_backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        attendance_port,
    ]
    attendance_dir = root / "Attendance Management" / "backend"

    backend_proc = subprocess.Popen(backend_cmd, cwd=root, env=child_env)
    
    attendance_proc = None
    if attendance_dir.exists():
        attendance_env = child_env.copy()
        attendance_env["PYTHONPATH"] = str(attendance_dir)
        attendance_proc = subprocess.Popen(attendance_backend_cmd, cwd=attendance_dir, env=attendance_env)
        
    # Open the attendance system in the browser after a short delay
    import threading
    import webbrowser
    import http.server
    import socketserver
    
    PORTAL_PORT = 8000
    
    def serve_portal():
        Handler = http.server.SimpleHTTPRequestHandler
        # Start server in the root directory to serve portal.html
        with socketserver.TCPServer(("", PORTAL_PORT), Handler) as httpd:
            httpd.serve_forever()
            
    portal_thread = threading.Thread(target=serve_portal, daemon=True)
    portal_thread.start()

    def open_browser():
        webbrowser.open(f"http://127.0.0.1:{PORTAL_PORT}/portal.html")
    
    threading.Timer(2.0, open_browser).start()

    try:
        subprocess.run(frontend_cmd, cwd=root, check=False, env=child_env)
    finally:
        # Shut down backend when Streamlit exits
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
