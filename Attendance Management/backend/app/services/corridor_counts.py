"""How many people passed through the corridor, and how many could be named.

WHY THIS IS SEPARATE FROM ATTENDANCE
------------------------------------
"Did somebody pass through?" and "who was it?" are different questions with
different failure modes, and answering them together is what made the corridor
count wrong.

A person is COUNTED on the strength of detection and tracking alone. Recognition
then decides only whether that person gets a name. A face that was never visible
downgrades an event from Employee to Unknown; it never removes the person.

So the totals here are deliberately assembled from two tables:

  * `attendance_events`          - transits where AdaFace named an employee, and
                                   which therefore also touch payroll.
  * `unknown_attendance_events`  - transits nobody could name. These carry a
                                   camera, a direction and a track, and no
                                   employee_id. An unknown person is NEVER
                                   speculatively attached to an employee.

in_count = employees_in + unknown_in, and likewise for out. If those two ever
disagree with the sum, the bug is in this module and not in the pipeline.

WHAT THIS IS NOT
----------------
Not a headcount of the building. It counts CONFIRMED TRANSITS observed by the
doorway cameras. At the measured doorway capture rate a person who crosses
entirely between two analysis passes is never sampled and cannot appear here, so
treat the totals as a reliable lower bound rather than an exact figure.
"""
from __future__ import annotations

import logging
from datetime import date as _date
from typing import Optional

logger = logging.getLogger(__name__)

# Directions a doorway camera can assert. MONITOR cameras are excluded from
# every query here: they watch desks, never a threshold, and letting them
# contribute would count somebody sitting down as somebody arriving.
_IN = "IN"
_OUT = "OUT"


def _doorway_cameras(db) -> dict[str, str]:
    """{camera_id: purpose} for cameras that define a corridor direction."""
    from app.models.camera import CameraConfig

    rows = (
        db.query(CameraConfig.id, CameraConfig.camera_purpose)
        .filter(CameraConfig.camera_purpose.in_([_IN, _OUT]))
        .all()
    )
    return {str(cid): str(purpose).upper() for cid, purpose in rows}


def corridor_summary(day: Optional[_date] = None) -> dict:
    """Counts for one day: transits in/out, and how many carried an identity."""
    from app.db.session import SessionLocal
    from app.models.attendance import AttendanceEvent
    from app.models.unknown_attendance_event import UnknownAttendanceEvent
    from app.core.datetime_utils import get_ist_now

    day = day or get_ist_now().date()

    with SessionLocal() as db:
        doorways = _doorway_cameras(db)
        if not doorways:
            return {
                "date": day.isoformat(), "in_count": 0, "out_count": 0,
                "employees_recognized": 0, "unknown_people": 0,
                "note": "no camera is configured with an IN or OUT purpose",
            }

        emp_in = emp_out = 0
        rows = (
            db.query(AttendanceEvent.camera_id, AttendanceEvent.event_type)
            .filter(
                AttendanceEvent.attendance_date == day,
                AttendanceEvent.camera_id.in_(list(doorways.keys())),
            )
            .all()
        )
        for camera_id, _event_type in rows:
            # Direction comes from the CAMERA's role, not the event name. The
            # event vocabulary in this table is mixed (IN/OUT alongside
            # CHECK_IN/BREAK_OUT/...), and the camera is the reliable signal
            # for which side of the threshold the person was on.
            if doorways.get(str(camera_id)) == _IN:
                emp_in += 1
            else:
                emp_out += 1

        unk_in = (
            db.query(UnknownAttendanceEvent)
            .filter(
                UnknownAttendanceEvent.attendance_date == day,
                UnknownAttendanceEvent.event_type == "UNKNOWN_IN",
            )
            .count()
        )
        unk_out = (
            db.query(UnknownAttendanceEvent)
            .filter(
                UnknownAttendanceEvent.attendance_date == day,
                UnknownAttendanceEvent.event_type == "UNKNOWN_OUT",
            )
            .count()
        )

    return {
        "date": day.isoformat(),
        "in_count": emp_in + unk_in,
        "out_count": emp_out + unk_out,
        "employees_recognized": emp_in + emp_out,
        "unknown_people": unk_in + unk_out,
        "breakdown": {
            "employees_in": emp_in, "employees_out": emp_out,
            "unknown_in": unk_in, "unknown_out": unk_out,
        },
    }


def corridor_events(day: Optional[_date] = None, limit: int = 200) -> list[dict]:
    """Individual transits for one day, newest first.

    Employee and unknown events are merged into one timeline because that is the
    question being asked - who went through this door today - and keeping them in
    separate lists pushes the join onto every caller.

    `identity` is None for an unknown transit. It is never filled in with a
    best-guess employee: a speculative name on a payroll-adjacent record is the
    failure this pipeline has already suffered once.
    """
    from app.db.session import SessionLocal
    from app.models.attendance import AttendanceEvent
    from app.models.employee import Employee
    from app.models.unknown_attendance_event import UnknownAttendanceEvent
    from app.core.datetime_utils import get_ist_now

    day = day or get_ist_now().date()
    out: list[dict] = []

    with SessionLocal() as db:
        doorways = _doorway_cameras(db)

        if doorways:
            rows = (
                db.query(AttendanceEvent, Employee)
                .outerjoin(Employee, Employee.id == AttendanceEvent.employee_id)
                .filter(
                    AttendanceEvent.attendance_date == day,
                    AttendanceEvent.camera_id.in_(list(doorways.keys())),
                )
                .order_by(AttendanceEvent.event_time.desc())
                .limit(limit)
                .all()
            )
            for ev, emp in rows:
                out.append({
                    "direction": doorways.get(str(ev.camera_id), _IN),
                    "identity": emp.full_name if emp else None,
                    "employee_id": ev.employee_id,
                    "type": "employee",
                    "camera_id": ev.camera_id,
                    "track_id": ev.track_id,
                    "timestamp": ev.event_time.isoformat() if ev.event_time else None,
                })

        unknowns = (
            db.query(UnknownAttendanceEvent)
            .filter(UnknownAttendanceEvent.attendance_date == day)
            .order_by(UnknownAttendanceEvent.event_time.desc())
            .limit(limit)
            .all()
        )
        for ev in unknowns:
            out.append({
                "direction": _IN if ev.event_type.endswith("_IN") else _OUT,
                "identity": None,
                "employee_id": None,
                "type": "unknown",
                "camera_id": ev.camera_id,
                "track_id": ev.track_id,
                "timestamp": ev.event_time.isoformat() if ev.event_time else None,
            })

    out.sort(key=lambda e: e["timestamp"] or "", reverse=True)
    return out[:limit]
