"""Click-to-name: teach a room camera who an unknown body is.

Why this exists
---------------
Body Re-ID can only match an unknown body against a GALLERY of known bodies, and
that gallery is normally bootstrapped by a face match ("this body belongs to the
face we just recognised"). On a ceiling camera ArcFace finds ZERO faces
(measured on the live Dev-room feed: 69x325 / 107x256 px bodies, no face even
after the 3x zoom), so nothing is ever enrolled and Re-ID has nothing to compare
against.

This endpoint supplies the missing bootstrap manually: an operator clicks an
"Unknown #3" box, picks the employee, and that person's CURRENT body embedding
is enrolled exactly as a face match would have done. Re-ID then carries the name
on that track — and onto the other room camera — for the rest of the day.

Clothing changes daily, so the enrolment is day-scoped like every other body
embedding: it must be repeated each day. That is the honest cost of identifying
people a camera cannot see the face of.

LABELLING ONLY. Nothing here can mark attendance — monitor cameras are blocked
from that at two independent layers, and this writes no attendance record.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import Employee, User
from app.api.deps import require_roles
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class TrackOut(BaseModel):
    track_id: int
    box: list[int]          # x1, y1, x2, y2 in frame pixels
    frame_w: int
    frame_h: int
    employee_id: int | None = None
    label: str              # current on-screen label
    named: bool


class NameTrackIn(BaseModel):
    camera_id: int
    track_id: int
    employee_id: int


def _worker(camera_id: int):
    from app.services.camera_service import camera_manager

    w = camera_manager._workers.get(int(camera_id))
    if w is None:
        raise HTTPException(
            status_code=404,
            detail="That camera is not currently running. Start it in Camera Manager.",
        )
    return w


@router.get("/tracks/{camera_id}", response_model=list[TrackOut])
def live_tracks(
    camera_id: int,
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """The people this camera can see right now, so the UI can offer them."""
    w = _worker(camera_id)
    frame = w.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No picture from this camera yet.")
    h, width = frame.shape[:2]
    out: list[TrackOut] = []
    with w._frame_lock:
        tracks = list(getattr(w, "_latest_tracks", []) or [])
    for t in tracks:
        box = [int(v) for v in (getattr(t, "box", None) or [0, 0, 0, 0])]
        emp = getattr(t, "employee_id", None)
        name = getattr(t, "employee_name", None)
        named = bool(getattr(t, "matched", False) and emp)
        out.append(
            TrackOut(
                track_id=int(getattr(t, "track_id", 0)),
                box=box, frame_w=int(width), frame_h=int(h),
                employee_id=int(emp) if emp else None,
                label=(name if named else f"Unknown #{getattr(t, 'track_id', '?')}"),
                named=named,
            )
        )
    return out


@router.post("/name-track")
def name_track(
    data: NameTrackIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Name a live track and teach Re-ID what that person looks like today."""
    import numpy as np  # noqa: F401  (used indirectly by reid_service)

    emp = db.query(Employee).filter(Employee.id == data.employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    w = _worker(data.camera_id)
    frame = w.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No picture from this camera yet.")

    with w._frame_lock:
        tracks = list(getattr(w, "_latest_tracks", []) or [])
    track = next((t for t in tracks if int(getattr(t, "track_id", -1)) == int(data.track_id)), None)
    if track is None:
        raise HTTPException(
            status_code=404,
            detail="That person has moved out of view. Refresh and pick them again.",
        )

    from app.services import reid_service
    from app.services.identity_manager import identity_manager

    if not reid_service.is_available():
        raise HTTPException(status_code=503, detail="Body Re-ID model is not available.")

    embs = reid_service.extract_body_embeddings(frame, [track.box])
    emb = embs[0] if embs else None
    if emb is None:
        raise HTTPException(status_code=422, detail="Could not read that person's appearance.")

    # Same call the face path makes — this is a legitimate gallery entry, just
    # vouched for by a human instead of by ArcFace.
    identity_manager.enroll(
        employee_id=int(emp.id),
        camera_id=str(w.camera_id),
        embedding=emb,
        centroid=track.centroid() if hasattr(track, "centroid") else None,
        score=1.0,                       # human-confirmed
        name=emp.full_name,
        code=emp.employee_code,
    )
    # Label the box immediately, rather than waiting for the next match cycle.
    try:
        track.bind_identity(int(emp.id), emp.full_name, emp.employee_code, True, 1.0)
    except Exception:
        logger.warning("track.bind_identity failed", exc_info=True)

    return {
        "message": f"{emp.full_name} identified. Re-ID will keep this name today.",
        "employee_id": emp.id,
        "employee_name": emp.full_name,
        "camera_id": w.camera_id,
        "track_id": data.track_id,
    }


class PauseIn(BaseModel):
    camera_id: int
    paused: bool


@router.post("/pause-analysis")
def pause_analysis(
    data: PauseIn,
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Pause/resume a camera's AI analysis WITHOUT stopping its video.

    Person detection costs ~2.3 s of CPU per frame here, and several cameras
    analysing at once starve a 4-core machine — scheduled jobs were seen running
    8 minutes late and the app looked like it had "shut down" when it was simply
    unresponsive. Pausing a room camera frees that CPU immediately; the live
    picture keeps working because the frame-grab thread is cheap and untouched.

    Attendance cameras (IN/OUT) can be paused too, but doing so stops attendance
    being marked from them, so the UI should warn before allowing it.
    """
    w = _worker(data.camera_id)
    w.analysis_paused = bool(data.paused)
    return {
        "camera_id": w.camera_id,
        "paused": w.analysis_paused,
        "message": ("Analysis paused — video keeps running, CPU freed."
                    if w.analysis_paused else "Analysis resumed."),
    }


@router.get("/analysis-status")
def analysis_status(
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Which cameras are analysing right now (for the UI toggles)."""
    from app.services.camera_service import camera_manager

    out = []
    for cid, w in list(camera_manager._workers.items()):
        out.append({
            "camera_id": int(cid),
            "name": getattr(w, "name", str(cid)),
            "camera_purpose": getattr(w, "camera_purpose", ""),
            "paused": bool(getattr(w, "analysis_paused", False)),
            "analysis_interval": float(getattr(w, "analysis_interval", 0.0)),
        })
    return sorted(out, key=lambda x: x["camera_id"])
