"""Deduplicate attendance_records and add a unique (employee_id, date) constraint.

Fixes the concurrency bug where two camera worker threads could each insert a
daily attendance row for the same employee (no DB uniqueness guaranteed it),
producing duplicate rows, split events, and flip-flopping work-hours/status.

Steps (idempotent — safe to re-run):
  1. Re-point events of duplicate rows to the KEPT row (lowest id per group).
  2. Delete the now-orphaned duplicate rows (no events are lost).
  3. Add UNIQUE(employee_id, date) so duplicates can never recur.
  4. Add a composite index on attendance_events for the hot per-day/cooldown query.

IMPORTANT: back up the database before running. After running, the kept rows'
summary (hours/status) is recomputed lazily by the app on the next event/read.

DO NOT RUN THIS ALONE. It must be deployed together with its companion code
(reverted for now, to be re-applied when C3/C4 is approved):
  * attendance_service.get_or_create_attendance — SAVEPOINT + IntegrityError retry
  * attendance_event_service.add_attendance_event — per-employee advisory lock
Without them, once this UNIQUE constraint exists a concurrent insert raises an
uncaught IntegrityError and that attendance write fails. This migration is a
DRAFT and is intentionally NOT wired into any deploy until C3/C4 is approved.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "020_attendance_unique_and_dedupe"
down_revision = "019_add_camera_line_crossing"
branch_labels = None
depends_on = None

_UQ_NAME = "uq_attendance_emp_date"
_IX_NAME = "ix_attendance_events_emp_date_time"


def _has_unique(inspector, table: str, name: str) -> bool:
    try:
        return any(uc.get("name") == name for uc in inspector.get_unique_constraints(table))
    except Exception:
        return False


def _has_index(inspector, table: str, name: str) -> bool:
    try:
        return any(ix.get("name") == name for ix in inspector.get_indexes(table))
    except Exception:
        return False


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())
    if "attendance_records" not in tables:
        return

    # 1) Re-point events of duplicate records to the kept (lowest-id) record.
    if "attendance_events" in tables:
        bind.execute(sa.text(
            """
            WITH ranked AS (
                SELECT id,
                       MIN(id) OVER (PARTITION BY employee_id, date) AS keep_id
                FROM attendance_records
            )
            UPDATE attendance_events e
            SET attendance_record_id = r.keep_id
            FROM ranked r
            WHERE e.attendance_record_id = r.id
              AND r.id <> r.keep_id
            """
        ))

    # 2) Delete the orphaned duplicate records (keep the lowest id per group).
    bind.execute(sa.text(
        """
        DELETE FROM attendance_records a
        USING (
            SELECT id,
                   MIN(id) OVER (PARTITION BY employee_id, date) AS keep_id
            FROM attendance_records
        ) r
        WHERE a.id = r.id AND r.id <> r.keep_id
        """
    ))

    # 3) Unique constraint so duplicates can never recur.
    if not _has_unique(inspector, "attendance_records", _UQ_NAME):
        op.create_unique_constraint(
            _UQ_NAME, "attendance_records", ["employee_id", "date"]
        )

    # 4) Composite index for the per-day event lookup + cooldown read (hot path).
    if "attendance_events" in tables and not _has_index(
        inspector, "attendance_events", _IX_NAME
    ):
        op.create_index(
            _IX_NAME,
            "attendance_events",
            ["employee_id", "attendance_date", "event_time"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if _has_index(inspector, "attendance_events", _IX_NAME):
        op.drop_index(_IX_NAME, table_name="attendance_events")
    if _has_unique(inspector, "attendance_records", _UQ_NAME):
        op.drop_constraint(_UQ_NAME, "attendance_records", type_="unique")
    # Deduplication is not reversed (the removed duplicate rows are not restored).
