import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """
    Start FastAPI backend and Streamlit frontend together.
    - Backend: uvicorn backend.api:app --host HOST --port 8001
      Use HOST=0.0.0.0 to allow access from other machines by IP; set BASE_URL in .env to
      the URL clients use (e.g. http://192.168.x.x:8001) so resume links work.
    - Frontend: streamlit run frontend/app.py
    """
    root = Path(__file__).resolve().parent
    host = os.getenv("HOST", "127.0.0.1")

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.api:app",
        "--host",
        host,
        "--port",
        "8001",
    ]
    frontend_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(root / "frontend" / "app.py"),
    ]

    backend_proc = subprocess.Popen(backend_cmd, cwd=root)
    try:
        subprocess.run(frontend_cmd, cwd=root, check=False)
    finally:
        # Shut down backend when Streamlit exits
        backend_proc.terminate()
        try: 
            backend_proc.wait(timeout=10)
        except Exception:
            backend_proc.kill()


if __name__ == "__main__":
    main()

