"""Recognition evidence on attendance events, and the snapshot store.

An attendance event recorded only `camera_id` — no score, no margin, no track
id, no image. So a disputed record could not be adjudicated, and every
recognition threshold in the CCTV pipeline was justified by anecdote (the code
comments cite one mis-labelled person) rather than by the distribution of real
matches.

These tests cover the parts that can go wrong silently: the columns existing,
the write path accepting evidence, and the snapshot store refusing to be walked
out of its own directory.
"""
from datetime import datetime

import numpy as np
import pytest
from app.models.attendance import AttendanceEvent
from app.services import attendance_snapshot

EVIDENCE_COLUMNS = ("match_score", "match_margin", "track_id", "snapshot_path")


@pytest.mark.parametrize("column", EVIDENCE_COLUMNS)
def test_event_model_has_evidence_column(column):
    assert column in AttendanceEvent.__table__.columns


@pytest.mark.parametrize("column", EVIDENCE_COLUMNS)
def test_evidence_columns_are_nullable(column):
    """Rows written before this existed — and every manual or AUTO_CLOSE event —
    genuinely have no evidence. NULL must mean "not applicable", never 0.0."""
    assert AttendanceEvent.__table__.columns[column].nullable is True


def test_add_attendance_event_accepts_evidence():
    """The write path must thread evidence all the way to the row.

    Signature-level check: the plumbing runs
    camera_service -> mark_cctv_attendance -> _mark_attendance ->
    record_face_attendance -> add_attendance_event, and a dropped kwarg
    anywhere along it silently discards the evidence with no error.
    """
    import inspect

    from app.services.attendance_event_service import (
        add_attendance_event,
        record_face_attendance,
    )
    from app.services.recognition import _mark_attendance, mark_cctv_attendance

    for fn in (add_attendance_event, record_face_attendance,
               _mark_attendance, mark_cctv_attendance):
        assert "evidence" in inspect.signature(fn).parameters, (
            f"{fn.__name__} drops the evidence kwarg — the chain is broken"
        )


def test_mark_cctv_attendance_accepts_an_explicit_event_time():
    """The event must be stamped at frame-capture time, not write time.

    The write is queued on a background executor behind a pipeline that already
    lags several hundred ms; stamping it on arrival drifts the record away from
    the snapshot stored with it.
    """
    import inspect

    from app.services.recognition import mark_cctv_attendance

    assert "event_time" in inspect.signature(mark_cctv_attendance).parameters


# ---------------------------------------------------------------------------
# Snapshot store
# ---------------------------------------------------------------------------
def test_snapshot_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(attendance_snapshot, "SNAPSHOT_ROOT", tmp_path)
    monkeypatch.setattr(attendance_snapshot, "ENABLED", True)

    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    relative = attendance_snapshot.save_face_snapshot(
        frame, [100, 60, 180, 160], employee_id=7, camera_id="53",
        # Naive on purpose: the attendance pipeline works in naive IST
        # throughout, matching TIMESTAMP WITHOUT TIME ZONE columns.
        when=datetime(2026, 7, 28, 9, 15, 30),  # noqa: DTZ001
    )
    assert relative is not None
    assert (tmp_path / relative).is_file()
    assert relative.startswith("2026-07-28/")


def test_snapshot_never_raises_on_bad_input(tmp_path, monkeypatch):
    """Losing evidence must never cost an attendance record.

    A missing image is a gap in the audit trail; a lost event is a missing
    day's pay. So every failure path returns None instead of propagating.
    """
    monkeypatch.setattr(attendance_snapshot, "SNAPSHOT_ROOT", tmp_path)
    monkeypatch.setattr(attendance_snapshot, "ENABLED", True)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    assert attendance_snapshot.save_face_snapshot(None, [0, 0, 5, 5], 1, "c") is None
    assert attendance_snapshot.save_face_snapshot(frame, None, 1, "c") is None
    assert attendance_snapshot.save_face_snapshot(frame, [0, 0], 1, "c") is None
    # Zero-area box -> empty crop, not an exception.
    assert attendance_snapshot.save_face_snapshot(frame, [5, 5, 5, 5], 1, "c") is None


def test_disabled_capture_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(attendance_snapshot, "SNAPSHOT_ROOT", tmp_path)
    monkeypatch.setattr(attendance_snapshot, "ENABLED", False)
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    assert attendance_snapshot.save_face_snapshot(frame, [10, 10, 50, 50], 1, "c") is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "camera_id",
    ["../../etc", "a/b", "c:\\windows", "..", "cam 53"],
)
def test_camera_id_cannot_escape_the_snapshot_directory(tmp_path, monkeypatch, camera_id):
    """camera_id is operator-supplied and lands in the filename."""
    monkeypatch.setattr(attendance_snapshot, "SNAPSHOT_ROOT", tmp_path)
    monkeypatch.setattr(attendance_snapshot, "ENABLED", True)
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)

    relative = attendance_snapshot.save_face_snapshot(
        frame, [10, 10, 60, 60], employee_id=1, camera_id=camera_id,
        when=datetime(2026, 1, 1, 0, 0, 0),  # noqa: DTZ001 — naive IST, see above
    )
    assert relative is not None
    written = (tmp_path / relative).resolve()
    assert written.is_relative_to(tmp_path.resolve()), f"escaped: {written}"


@pytest.mark.parametrize("attack", ["../../../etc/passwd", "/etc/passwd", "..\\..\\secret"])
def test_resolve_snapshot_rejects_traversal(tmp_path, monkeypatch, attack):
    """snapshot_path comes back out of the DB and will feed a download route."""
    monkeypatch.setattr(attendance_snapshot, "SNAPSHOT_ROOT", tmp_path)
    assert attendance_snapshot.resolve_snapshot(attack) is None


def test_prune_removes_old_days_and_keeps_recent(tmp_path, monkeypatch):
    from datetime import date, timedelta

    monkeypatch.setattr(attendance_snapshot, "SNAPSHOT_ROOT", tmp_path)
    # prune_snapshots compares against the local date, so the test must too.
    today = date.today()  # noqa: DTZ011
    old = tmp_path / (today - timedelta(days=200)).isoformat()
    recent = tmp_path / today.isoformat()
    unrelated = tmp_path / "not-a-date"
    for d in (old, recent, unrelated):
        d.mkdir(parents=True)
        (d / "x.jpg").write_bytes(b"x")

    removed = attendance_snapshot.prune_snapshots(retention_days=90)

    assert removed == 1
    assert not old.exists()
    assert recent.exists()
    # A folder that is not date-named is left alone rather than guessed at.
    assert unrelated.exists()
