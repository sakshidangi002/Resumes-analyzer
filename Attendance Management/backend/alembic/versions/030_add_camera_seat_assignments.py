"""Add camera_seat_assignments (manual desk -> employee mapping for MONITOR cameras)

Revision ID: 030_add_camera_seat_assignments
Revises: 029_add_interview_questions
Create Date: 2026-07-20

Purely additive: one new table. Nothing existing is touched, and attendance is
unaffected — seat matches only LABEL a live box on a monitor camera.
"""
from alembic import op
import sqlalchemy as sa


revision = '030_add_camera_seat_assignments'
down_revision = '029_add_interview_questions'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "camera_seat_assignments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("camera_id", sa.String(length=50), nullable=False),
        sa.Column("employee_id", sa.Integer(), nullable=False),
        sa.Column("x1", sa.Float(), nullable=False),
        sa.Column("y1", sa.Float(), nullable=False),
        sa.Column("x2", sa.Float(), nullable=False),
        sa.Column("y2", sa.Float(), nullable=False),
        sa.Column("label", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["employee_id"], ["employees.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("camera_id", "employee_id", name="uq_seat_camera_employee"),
    )
    op.create_index("ix_camera_seat_camera", "camera_seat_assignments", ["camera_id"])
    op.create_index("ix_camera_seat_employee", "camera_seat_assignments", ["employee_id"])


def downgrade():
    op.drop_index("ix_camera_seat_employee", table_name="camera_seat_assignments")
    op.drop_index("ix_camera_seat_camera", table_name="camera_seat_assignments")
    op.drop_table("camera_seat_assignments")
