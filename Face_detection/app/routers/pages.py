from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

router = APIRouter(tags=["pages"])


def _render_page(filename: str) -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / filename).read_text(encoding="utf-8"))


@router.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return _render_page("dashboard.html")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_page() -> HTMLResponse:
    return _render_page("dashboard.html")


@router.get("/employees", response_class=HTMLResponse)
def employees_page() -> HTMLResponse:
    return _render_page("employees.html")


@router.get("/webcam", response_class=HTMLResponse)
def webcam_page() -> HTMLResponse:
    return _render_page("webcam.html")

@router.get("/face-detection", response_class=HTMLResponse)
def face_detection_page() -> HTMLResponse:
    return _render_page("face-detection.html")


@router.get("/cameras", response_class=HTMLResponse)
def cameras_page() -> HTMLResponse:
    return _render_page("cameras.html")


@router.get("/attendance", response_class=HTMLResponse)
def attendance_page() -> HTMLResponse:
    return _render_page("attendance.html")
