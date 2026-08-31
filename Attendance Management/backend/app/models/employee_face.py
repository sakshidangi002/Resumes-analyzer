"""One row per enrolled face embedding, with the provenance needed to trust it.

Before this table an employee's entire enrolment was a single opaque column,
``employees.embedding``, holding an (N, 512) float32 stack. That representation
works for matching — the matcher scores against the best row — but it discards
everything needed to REASON about the gallery:

* **Which model produced it.** ``EMBEDDING_MODEL_VERSION`` existed as a constant
  in face_service and was referenced nowhere. Swapping the recognition model, or
  switching FACE_DETECTOR between insightface and yolo, silently mixes vectors
  from different spaces into one comparison. Nothing could detect that, and the
  symptom — everyone's scores quietly drop — looks exactly like a camera problem.
* **Whether it was landmark-aligned.** The YOLO detector path falls back to a
  raw resized crop when the weight file has no keypoints. Those vectors are not
  comparable with SCRFD-aligned ones and were being enrolled indistinguishably.
* **Which camera it came from.** The single highest-leverage fix available on a
  deployment whose cameras cannot be moved is enrolling people from the viewpoint
  that will later have to recognise them. That is only actionable if you can ask
  "does this employee have an embedding from the check-in camera?".
* **How good the source photo was.** A gallery entry enrolled from a marginal
  frame drags a whole employee's matching down and there was no way to find it.

``employees.embedding`` is kept as the hot path — it is what the in-memory
matcher cache loads, and rebuilding it from these rows on change keeps every
existing query working unchanged. This table is the source of truth behind it.
"""
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String,
)
from sqlalchemy.orm import relationship

from app.core.datetime_utils import get_ist_now
from app.core.encrypted_types import EncryptedBinary
from app.db.base_class import Base


class EmployeeFaceEmbedding(Base):
    __tablename__ = "employee_face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(
        Integer, ForeignKey("employees.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    embedding = Column(EncryptedBinary, nullable=False)     # encrypted float32[512]

    # --- Provenance ---------------------------------------------------------
    model_version = Column(String(64), nullable=False)      # e.g. insightface_buffalo_l_v1
    detector = Column(String(32), nullable=True)            # insightface | yolo
    aligned = Column(Boolean, nullable=False, default=True) # landmark-aligned crop?
    source = Column(String(16), nullable=False, default="upload")   # upload | cctv
    camera_id = Column(String(50), nullable=True, index=True)

    # --- Quality of the frame this came from --------------------------------
    # Stored so a badly-enrolled employee can be found and re-enrolled without
    # re-processing the original photos (which may no longer exist).
    quality_score = Column(Float, nullable=True)
    face_px = Column(Float, nullable=True)
    yaw = Column(Float, nullable=True)
    pitch = Column(Float, nullable=True)
    blur_var = Column(Float, nullable=True)

    # Soft-delete. Retiring an embedding must not destroy the audit trail of
    # what the system believed when it wrote an attendance row last month.
    active = Column(Boolean, nullable=False, default=True, index=True)

    created_at = Column(DateTime, default=get_ist_now, nullable=False)

    employee = relationship("Employee", backref="face_embeddings")


Index(
    "ix_employee_face_embeddings_emp_active",
    EmployeeFaceEmbedding.employee_id,
    EmployeeFaceEmbedding.active,
)
