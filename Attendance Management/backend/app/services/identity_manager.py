"""Global Identity Manager — keeps an employee's identity on their body track
across cameras and across the day, without needing their face every frame.

Two independent signals, tried in order:

1. BODY Re-ID (OSNet appearance embedding)
   * Gallery is keyed by (employee, camera). A ceiling-mounted seated view and a
     standing check-in view look nothing alike to a ReID model, so matching is
     done against embeddings captured BY THE SAME CAMERA whenever possible.
   * Enrolment is OPPORTUNISTIC: every time ArcFace positively recognises a face
     on a camera, that person's body crop from that camera is added to the
     gallery. Employees therefore teach every camera what they look like from
     that camera's angle, simply by being recognised there once.
   * A cross-camera fallback exists (match against the employee's embeddings from
     OTHER cameras) but demands a HIGHER similarity, because the viewpoint gap
     makes those comparisons less trustworthy.

2. SEAT ANCHOR (spatial memory)
   * In a fixed-desk room the strongest signal is simply "who sits there". When a
     face identifies someone at a location, that location is remembered. An
     unknown track appearing at the same spot inherits the identity.
   * Requires no zone configuration — anchors are learned from face matches.

Both signals are DAY-SCOPED (clothes and seats change daily) and are only ever
used to LABEL people. They must never mark attendance — only ArcFace on an
IN/OUT camera may do that (enforced in camera_service / recognition).
"""
from __future__ import annotations

import logging
import threading
from datetime import date
from typing import Optional

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# At most this many embeddings per (employee, camera) — newest kept. Like the face
# gallery, they are NOT averaged: different poses stay separate vectors.
_MAX_PER_SLOT = 8


def _blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _unblob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


class GlobalIdentityManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day: Optional[date] = None
        # (employee_id, camera_id) -> list[np.ndarray]
        self._gallery: dict[tuple[int, str], list[np.ndarray]] = {}
        # camera_id -> list[dict(centroid=(x,y), employee_id=int)]
        self._seats: dict[str, list[dict]] = {}
        # employee_id -> (full_name, employee_code) so a Re-ID bind can label the
        # box without a DB round-trip on every frame.
        self._names: dict[int, tuple] = {}
        self._loaded = False

    def label(self, employee_id: int) -> tuple:
        """(name, code) for an employee we have seen today."""
        return self._names.get(int(employee_id), ("Person", None))

    # ---------------- day handling ----------------
    def _roll(self, today: date) -> None:
        """Clothes and seating change daily — start each day from a clean slate."""
        if self._day != today:
            self._day = today
            self._gallery.clear()
            self._seats.clear()
            self._names.clear()
            self._loaded = False
            logger.info("IdentityManager: new day %s — gallery reset", today)

    def _ensure_loaded(self, today: date) -> None:
        """Load today's embeddings from the DB once (survives a restart)."""
        self._roll(today)
        if self._loaded:
            return
        self._loaded = True
        try:
            from app.db.session import SessionLocal
            from app.models import BodyEmbedding, Employee

            with SessionLocal() as db:
                rows = (
                    db.query(BodyEmbedding, Employee)
                    .join(Employee, Employee.id == BodyEmbedding.employee_id)
                    .filter(BodyEmbedding.day == today)
                    .all()
                )
            for r, emp in rows:
                key = (int(r.employee_id), str(r.camera_id))
                self._gallery.setdefault(key, []).append(_unblob(r.embedding))
                self._names[int(r.employee_id)] = (emp.full_name, emp.employee_code)
            if rows:
                logger.info(
                    "IdentityManager: restored %d body embeddings for %s", len(rows), today
                )
        except Exception:
            logger.exception("IdentityManager: could not restore today's embeddings")

    # ---------------- enrolment ----------------
    def enroll(
        self,
        employee_id: int,
        camera_id: str,
        embedding: np.ndarray,
        centroid: Optional[tuple] = None,
        score: float = 0.0,
        name: Optional[str] = None,
        code: Optional[str] = None,
    ) -> None:
        """Teach the gallery: 'on THIS camera, employee X currently looks like this'.
        Called only after a positive FACE match, so the label is trustworthy."""
        if embedding is None:
            return
        today = date.today()
        with self._lock:
            self._ensure_loaded(today)
            if name:
                self._names[int(employee_id)] = (name, code)
            key = (int(employee_id), str(camera_id))
            slot = self._gallery.setdefault(key, [])

            # Skip near-duplicates — keep the gallery diverse rather than 8 copies
            # of the same pose.
            if any(float(np.dot(embedding, e)) > 0.97 for e in slot):
                persist = False
            else:
                slot.append(np.asarray(embedding, dtype=np.float32))
                if len(slot) > _MAX_PER_SLOT:
                    slot.pop(0)
                persist = True

            if centroid is not None:
                self._remember_seat(str(camera_id), centroid, int(employee_id))

        if persist:
            self._persist(employee_id, camera_id, embedding, today, score)

    def _persist(self, employee_id, camera_id, embedding, day, score) -> None:
        try:
            from app.db.session import SessionLocal
            from app.models import BodyEmbedding

            with SessionLocal() as db:
                db.add(
                    BodyEmbedding(
                        employee_id=int(employee_id),
                        camera_id=str(camera_id),
                        day=day,
                        embedding=_blob(embedding),
                        score=int(float(score) * 100),
                    )
                )
                db.commit()
        except Exception:
            logger.exception("IdentityManager: failed to persist body embedding")

    # ---------------- seat anchors ----------------
    def _remember_seat(self, camera_id: str, centroid: tuple, employee_id: int) -> None:
        radius = get_settings().seat_anchor_radius_px
        seats = self._seats.setdefault(camera_id, [])
        for s in seats:
            if _dist(s["centroid"], centroid) <= radius:
                s["centroid"] = centroid          # drift with the person
                s["employee_id"] = employee_id    # newest face match wins the seat
                return
        seats.append({"centroid": centroid, "employee_id": employee_id})

    def seat_match(self, camera_id: str, centroid: tuple, taken: set) -> Optional[int]:
        """Who normally sits here? Used when Re-ID is not confident enough."""
        s = get_settings()
        if not s.seat_anchor_enabled or centroid is None:
            return None
        with self._lock:
            self._ensure_loaded(date.today())
            best, best_d = None, None
            for seat in self._seats.get(str(camera_id), []):
                emp = seat["employee_id"]
                if emp in taken:
                    continue
                d = _dist(seat["centroid"], centroid)
                if d <= s.seat_anchor_radius_px and (best_d is None or d < best_d):
                    best, best_d = emp, d
            return best

    # ---------------- matching ----------------
    def match(self, camera_id: str, embedding: np.ndarray, taken: set) -> tuple:
        """Return (employee_id, score, source) or (None, 0.0, None).

        `taken` = employee ids already bound to another live track on this camera;
        one person cannot be in two places at once, so they are excluded.
        """
        if embedding is None:
            return (None, 0.0, None)
        s = get_settings()
        today = date.today()
        with self._lock:
            self._ensure_loaded(today)

            same_cam: dict[int, float] = {}
            other_cam: dict[int, float] = {}
            for (emp, cam), vecs in self._gallery.items():
                if emp in taken or not vecs:
                    continue
                best = max(float(np.dot(embedding, v)) for v in vecs)
                bucket = same_cam if cam == str(camera_id) else other_cam
                if best > bucket.get(emp, -1.0):
                    bucket[emp] = best

        # Prefer this camera's own viewpoint; fall back to other cameras with a
        # stricter threshold (the viewpoint gap makes those less reliable).
        for bucket, thresh, source in (
            (same_cam, s.reid_threshold, "reid"),
            (other_cam, s.reid_cross_camera_threshold, "reid-xcam"),
        ):
            if not bucket:
                continue
            ranked = sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)
            emp, score = ranked[0]
            runner = ranked[1][1] if len(ranked) > 1 else -1.0
            margin = score - runner if runner >= 0 else score
            if score >= thresh and margin >= s.reid_min_margin:
                return (emp, score, source)

        return (None, 0.0, None)


def _dist(a, b) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


# Process-wide singleton (cameras run in threads within one process).
identity_manager = GlobalIdentityManager()
