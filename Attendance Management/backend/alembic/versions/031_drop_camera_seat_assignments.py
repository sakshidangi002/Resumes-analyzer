"""Drop camera_seat_assignments (desk mapping feature removed)

Revision ID: 031_drop_camera_seat_assignments
Revises: 030_add_camera_seat_assignments
Create Date: 2026-07-20

The manual desk->employee mapping was removed at the user's request. Its table
is dropped so no orphan schema (with a live FK to employees) is left behind.
The downgrade recreates it, so the feature can be restored from git history.
"""
from alembic import op
import sqlalchemy as sa


revision = '031_drop_camera_seat_assignments'
down_revision = '030_add_camera_seat_assignments'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_table("camera_seat_assignments")


def downgrade():
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
