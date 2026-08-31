"""anonymous IN/OUT events for review"""
from alembic import op
import sqlalchemy as sa

revision = "037"
down_revision = "036"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "unknown_attendance_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("camera_id", sa.String(length=50), nullable=False),
        sa.Column("event_time", sa.DateTime(), nullable=False),
        sa.Column("attendance_date", sa.Date(), nullable=False),
        sa.Column("event_type", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="PENDING"),
        sa.Column("unknown_face_id", sa.Integer(), sa.ForeignKey("unknown_faces.id", ondelete="SET NULL"), nullable=True),
        sa.Column("track_id", sa.Integer(), nullable=True),
        sa.Column("crop_path", sa.String(length=300), nullable=True),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("match_score", sa.Float(), nullable=True),
        sa.Column("match_margin", sa.Float(), nullable=True),
        sa.Column("assigned_employee_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=True),
        sa.Column("reviewed_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_unknown_attendance_events_camera_id", "unknown_attendance_events", ["camera_id"])
    op.create_index("ix_unknown_attendance_events_event_time", "unknown_attendance_events", ["event_time"])
    op.create_index("ix_unknown_attendance_events_attendance_date", "unknown_attendance_events", ["attendance_date"])
    op.create_index("ix_unknown_attendance_events_status", "unknown_attendance_events", ["status"])
    op.create_index("ix_unknown_attendance_review", "unknown_attendance_events", ["status", "event_time"])
    op.create_index("ix_unknown_attendance_camera_time", "unknown_attendance_events", ["camera_id", "event_time"])
    op.create_index("ix_unknown_attendance_events_unknown_face_id", "unknown_attendance_events", ["unknown_face_id"])
    op.create_index("ix_unknown_attendance_events_assigned_employee_id", "unknown_attendance_events", ["assigned_employee_id"])


def downgrade():
    op.drop_table("unknown_attendance_events")
