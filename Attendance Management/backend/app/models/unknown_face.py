"""Faces the system saw but could not name — kept instead of discarded.

Every unmatched face used to end at ``state = "unknown"`` and vanish with the
frame. That threw away the only feedback signal this deployment has.

On a site where the cameras cannot be moved, global threshold tuning quickly
hits a ceiling: the remaining errors are not spread evenly across employees, they
are concentrated in a handful of people whose enrolled photos do not resemble
what their camera actually sees. Without a record of unmatched faces there is no
way to discover WHO those people are — you can only observe that "accuracy is
about 75%" and turn knobs blindly.

With this table the loop closes:

    unmatched face -> stored with its crop, embedding and quality
                   -> clustered by appearance (same stranger, many sightings)
                   -> HR assigns a cluster to an employee
                   -> those embeddings are enrolled, FROM THAT CAMERA'S VIEWPOINT
                   -> that employee starts matching

which is also the cheapest possible path to viewpoint-matched enrolment.

Retention: rows carry a crop on disk and a biometric embedding, so they are
subject to the same handling as enrolled faces. ``purge_older_than`` in
services/unknown_faces.py is the intended scheduled cleanup.
"""
from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Index, Integer, String,
)

from app.core.datetime_utils import get_ist_now
from app.core.encrypted_types import EncryptedBinary
from app.db.base_class import Base


# Lifecycle
STATUS_PENDING = "pending"     # awaiting review
STATUS_ASSIGNED = "assigned"   # attributed to an employee and enrolled
STATUS_IGNORED = "ignored"     # visitor / not a person of interest


class UnknownFace(Base):
    __tablename__ = "unknown_faces"

    id = Column(Integer, primary_key=True, index=True)

    camera_id = Column(String(50), nullable=False, index=True)
    captured_at = Column(DateTime, default=get_ist_now, nullable=False, index=True)

    embedding = Column(EncryptedBinary, nullable=False)   # encrypted float32[512]
    crop_path = Column(String(300), nullable=True)        # saved face crop, for review

    # --- Quality of the observation ----------------------------------------
    # Only faces that PASSED the camera's quality gate are stored: an unusable
    # face is not evidence of a recognition failure, it is evidence of a bad
    # frame, and filling the queue with those makes it useless to review.
    quality_score = Column(Float, nullable=True)
    face_px = Column(Float, nullable=True)
    yaw = Column(Float, nullable=True)
    pitch = Column(Float, nullable=True)
    blur_var = Column(Float, nullable=True)

    # --- Why it was not matched --------------------------------------------
    # The near-miss. "Scored 0.41 against Sakshi, needed 0.45" is a completely
    # different finding from "scored 0.08 against everyone" — the first is a
    # threshold or enrolment problem, the second is a stranger.
    best_score = Column(Float, nullable=True)
    best_margin = Column(Float, nullable=True)
    best_employee_id = Column(
        Integer, ForeignKey("employees.id", ondelete="SET NULL"), nullable=True,
    )

    # --- Clustering / review ------------------------------------------------
    cluster_id = Column(Integer, nullable=True, index=True)
    status = Column(String(16), nullable=False, default=STATUS_PENDING, index=True)
    assigned_employee_id = Column(
        Integer, ForeignKey("employees.id", ondelete="SET NULL"), nullable=True,
    )
    reviewed_by = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
    )
    reviewed_at = Column(DateTime, nullable=True)


Index("ix_unknown_faces_status_camera", UnknownFace.status, UnknownFace.camera_id)
Index("ix_unknown_faces_cluster_status", UnknownFace.cluster_id, UnknownFace.status)
