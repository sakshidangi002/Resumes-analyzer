"""Let HR enter break time by hand and have it deducted from working hours

Revision ID: 028_manual_break_time
Revises: 027_manual_attendance_times
Create Date: 2026-07-14

Break time was only ever derived from camera OUT->IN pairs. When the camera misses
an employee, it sees no break either, so an HR-entered day counted the full span
as worked. break_manual marks a break that HR typed in, which the recalculation
then uses instead of the camera's value.
"""
from alembic import op
import sqlalchemy as sa


revision = '028_manual_break_time'
down_revision = '027_manual_attendance_times'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "attendance_records",
        sa.Column("break_manual", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )


def downgrade():
    op.drop_column("attendance_records", "break_manual")
