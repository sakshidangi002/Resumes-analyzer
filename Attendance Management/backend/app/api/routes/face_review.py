"""Admin surface for the unknown-face review queue and per-camera profiles.

Two capabilities that previously had no home:

  * **Unknown-face review.** Faces the cameras saw, judged usable, but could not
    name. Clustering them turns "recognition is about 75%" into "these four
    people are never recognised at the check-in camera", which is the only
    actionable form of that information when the cameras cannot be moved.
    Assigning a cluster enrols those embeddings for that employee, tagged with
    the camera they came from — viewpoint-matched enrolment from live traffic.

  * **Camera recognition profiles.** Editing the thresholds, quality limits and
    evidence requirements that were previously process-wide environment
    variables. Changes take effect within seconds, without restarting the
    server and therefore without dropping every camera's stream.

Both are Admin/HR only. The unknown-face endpoints return biometric data
(face crops), so the crop download goes through the same containment check and
role gate as attendance snapshots.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_roles
from app.models.user import User
from app.services import camera_profile, unknown_faces
from app.services import unknown_attendance

logger = logging.getLogger(__name__)

router = APIRouter()


class ResolveUnknownAttendanceRequest(BaseModel):
    employee_id: Optional[int] = Field(default=None, ge=1)


@router.get("/unknown-attendance")
def list_unknown_attendance(
    status: str = Query(default="PENDING", pattern="^(PENDING|ASSIGNED|IGNORED)$"),
    limit: int = Query(default=100, ge=1, le=500),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    return {"events": unknown_attendance.list_events(status=status, limit=limit)}


@router.post("/unknown-attendance/{event_id}/resolve")
def resolve_unknown_attendance(
    event_id: int,
    payload: ResolveUnknownAttendanceRequest,
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Assign or dismiss the anonymous event; payroll is never changed silently."""
    try:
        return unknown_attendance.resolve(
            event_id, employee_id=payload.employee_id,
            reviewed_by=getattr(current_user, "id", None),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/unknown-attendance/{event_id}/crop")
def get_unknown_attendance_crop(
    event_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    from app.models.unknown_attendance_event import UnknownAttendanceEvent
    row = db.query(UnknownAttendanceEvent).filter(UnknownAttendanceEvent.id == event_id).first()
    if row is None or not row.crop_path:
        raise HTTPException(status_code=404, detail="No crop stored for this event")
    path = unknown_faces.resolve_crop(row.crop_path)
    if path is None:
        raise HTTPException(status_code=404, detail="Crop file is missing")
    return FileResponse(str(path), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Unknown faces
# ---------------------------------------------------------------------------
@router.post("/unknown-faces/recluster")
def recluster_unknown_faces(
    min_cos: float = Query(
        default=unknown_faces.CLUSTER_MIN_COS,
        ge=0.0, le=1.0,
        description="Cosine similarity required to merge two sightings into one cluster",
    ),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Regroup pending unknown faces by appearance.

    Cheap enough to run on demand (a few thousand 512-d vectors is milliseconds),
    so it is exposed as an action rather than a scheduled job — a reviewer can
    recluster after a busy morning and immediately see the new groups.
    """
    return unknown_faces.recluster(min_cos=min_cos)


@router.get("/unknown-faces/clusters")
def list_unknown_clusters(
    limit: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Pending clusters, largest first.

    Largest first is deliberate: a cluster of 40 sightings is somebody the
    cameras are systematically failing to recognise and is worth an HR minute; a
    cluster of 1 is probably a visitor.
    """
    return {"clusters": unknown_faces.list_clusters(limit=limit)}


@router.get("/unknown-faces/{face_id}/crop")
def get_unknown_face_crop(
    face_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """The stored face crop for one sighting, so a reviewer can identify it."""
    from app.models.unknown_face import UnknownFace

    row = db.query(UnknownFace).filter(UnknownFace.id == face_id).first()
    if row is None or not row.crop_path:
        raise HTTPException(status_code=404, detail="No crop stored for this face")

    # Containment is enforced in resolve_crop: crop_path round-trips through the
    # database and must never be able to escape the data directory.
    path = unknown_faces.resolve_crop(row.crop_path)
    if path is None:
        raise HTTPException(status_code=404, detail="Crop file is missing")
    return FileResponse(str(path), media_type="image/jpeg")


class AssignClusterRequest(BaseModel):
    employee_id: int = Field(..., description="Employee this cluster belongs to")
    max_enroll: int = Field(
        default=5, ge=1, le=10,
        description="How many of the cluster's best faces to add to the gallery",
    )


@router.post("/unknown-faces/clusters/{cluster_id}/assign")
def assign_unknown_cluster(
    cluster_id: int,
    payload: AssignClusterRequest,
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Attribute a cluster to an employee and enrol its best faces.

    Only the highest-quality members are enrolled. Adding every sighting would
    inflate that employee's gallery depth with near-duplicate vectors, which the
    matcher's enrolment-bias correction then has to undo — and a deep gallery of
    mediocre vectors is the condition that produced this system's known
    mislabelling.
    """
    try:
        result = unknown_faces.assign_cluster(
            cluster_id=cluster_id,
            employee_id=payload.employee_id,
            reviewed_by=getattr(current_user, "id", None),
            max_enroll=payload.max_enroll,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    logger.info(
        "UNKNOWN-FACE cluster=%s assigned to employee=%s by user=%s",
        cluster_id, payload.employee_id, getattr(current_user, "id", None),
    )
    return result


@router.post("/unknown-faces/clusters/{cluster_id}/ignore")
def ignore_unknown_cluster(
    cluster_id: int,
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Mark a cluster as 'not an employee' so it leaves the review queue."""
    count = unknown_faces.ignore_cluster(
        cluster_id=cluster_id, reviewed_by=getattr(current_user, "id", None)
    )
    return {"ignored": count}


# ---------------------------------------------------------------------------
# Camera recognition profiles
# ---------------------------------------------------------------------------
class CameraProfileUpdate(BaseModel):
    """Every field is optional; NULL means 'inherit the purpose default'.

    Sending null for a field explicitly clears an override and returns that
    setting to the default for the camera's role, which is how an operator
    backs out a change they cannot justify with measurements.
    """

    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    match_margin: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_face_px: Optional[int] = Field(default=None, ge=8, le=500)
    min_det_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_yaw_deg: Optional[float] = Field(default=None, ge=0.0, le=90.0)
    max_pitch_deg: Optional[float] = Field(default=None, ge=0.0, le=90.0)
    max_landmark_asym: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_blur_var: Optional[float] = Field(default=None, ge=0.0)
    min_observations: Optional[int] = Field(default=None, ge=1, le=30)
    min_quality: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_consensus: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    analysis_interval: Optional[float] = Field(default=None, ge=0.02, le=10.0)
    face_crop_scale: Optional[int] = Field(default=None, ge=1, le=6)
    attendance_cooldown: Optional[float] = Field(default=None, ge=0.0, le=3600.0)


_PROFILE_FIELDS = tuple(CameraProfileUpdate.model_fields.keys())


@router.get("/cameras/{camera_id}/profile")
def get_camera_profile(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Stored overrides plus the RESOLVED values actually in force.

    Both are returned because they answer different questions. `overrides` is
    what an operator typed; `effective` is what the camera is running, after
    defaults and safety floors. Showing only one of them is how a camera ends up
    running a value nobody believes it has.
    """
    from app.models.camera import CameraConfig

    row = db.query(CameraConfig).filter(CameraConfig.id == camera_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    resolved = camera_profile.get_profile(camera_id, row.camera_purpose or "IN")
    return {
        "camera_id": camera_id,
        "purpose": resolved.purpose,
        "marks_attendance": resolved.marks_attendance,
        "overrides": {name: getattr(row, name, None) for name in _PROFILE_FIELDS},
        "effective": {
            "threshold": resolved.threshold,
            "match_margin": resolved.margin,
            "min_face_px": resolved.limits.min_face_px,
            "min_det_score": resolved.limits.min_det_score,
            "max_yaw_deg": resolved.limits.max_yaw_deg,
            "max_pitch_deg": resolved.limits.max_pitch_deg,
            "max_landmark_asym": resolved.limits.max_landmark_asym,
            "min_blur_var": resolved.limits.min_blur_var,
            "min_observations": resolved.min_observations,
            "min_quality": resolved.min_quality,
            "min_consensus": resolved.min_consensus,
            "analysis_interval": resolved.analysis_interval,
            "face_crop_scale": resolved.face_crop_scale,
            "attendance_cooldown": resolved.attendance_cooldown,
        },
        "summary": resolved.describe(),
    }


@router.put("/cameras/{camera_id}/profile")
def update_camera_profile(
    camera_id: int,
    payload: CameraProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin"])),
):
    """Update a camera's recognition settings. Applies without a restart.

    Admin-only, not HR: these values decide when a payroll row gets written, and
    loosening them on an IN/OUT camera is the fastest way to manufacture false
    attendance. The safety floors in camera_profile still apply on top of
    whatever is stored, so a bad value degrades to the floor rather than to
    "match anybody".
    """
    from app.models.camera import CameraConfig

    row = db.query(CameraConfig).filter(CameraConfig.id == camera_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    # exclude_unset so an omitted field keeps its stored value, while an
    # explicit null clears the override back to the purpose default.
    changes = payload.model_dump(exclude_unset=True)
    for name, value in changes.items():
        setattr(row, name, value)
    db.commit()

    # Drop the cached profile so the change lands on the next analysis tick
    # instead of after the TTL.
    camera_profile.invalidate(camera_id)
    resolved = camera_profile.get_profile(camera_id, row.camera_purpose or "IN")

    logger.info(
        "CAMERA-PROFILE camera=%s updated by user=%s fields=%s -> %s",
        camera_id, getattr(current_user, "id", None),
        sorted(changes), resolved.describe(),
    )
    return {"camera_id": camera_id, "applied": changes, "summary": resolved.describe()}
