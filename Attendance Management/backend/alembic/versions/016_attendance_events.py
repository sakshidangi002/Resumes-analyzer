"""Add attendance_events table and total_break_hours on attendance_records"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


def _table_columns(inspector, table_name: str) -> set[str]:
    if table_name not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    event_cols = _table_columns(inspector, "attendance_events")
    if event_cols and "event_time" not in event_cols:
        # Legacy face-detection schema (timestamp/confidence) — replace with HRMS events.
        op.drop_table("attendance_events")
        event_cols = set()

    if not event_cols:
        op.create_table(
            "attendance_events",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("employee_id", sa.Integer(), nullable=False),
            sa.Column("attendance_record_id", sa.Integer(), nullable=True),
            sa.Column("event_time", sa.DateTime(), nullable=False),
            sa.Column("event_type", sa.String(length=10), nullable=False),
            sa.Column("source", sa.String(length=20), nullable=False, server_default="AUTO"),
            sa.Column("camera_id", sa.String(length=50), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["attendance_record_id"], ["attendance_records.id"]),
            sa.ForeignKeyConstraint(["employee_id"], ["employees.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_attendance_events_id"), "attendance_events", ["id"], unique=False)
        op.create_index(
            op.f("ix_attendance_events_employee_id"), "attendance_events", ["employee_id"], unique=False
        )
        op.create_index(
            op.f("ix_attendance_events_attendance_record_id"),
            "attendance_events",
            ["attendance_record_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_attendance_events_event_time"), "attendance_events", ["event_time"], unique=False
        )

    record_cols = _table_columns(inspector, "attendance_records")
    if "total_break_hours" not in record_cols:
        op.add_column(
            "attendance_records",
            sa.Column("total_break_hours", sa.Numeric(precision=5, scale=2), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    record_cols = _table_columns(inspector, "attendance_records")
    if "total_break_hours" in record_cols:
        op.drop_column("attendance_records", "total_break_hours")

    if "attendance_events" in inspector.get_table_names():
        op.drop_index(op.f("ix_attendance_events_event_time"), table_name="attendance_events")
        op.drop_index(op.f("ix_attendance_events_attendance_record_id"), table_name="attendance_events")
        op.drop_index(op.f("ix_attendance_events_employee_id"), table_name="attendance_events")
        op.drop_index(op.f("ix_attendance_events_id"), table_name="attendance_events")
        op.drop_table("attendance_events")
