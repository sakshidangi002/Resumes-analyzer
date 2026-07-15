"""Pin HR-entered attendance times so camera detections cannot overwrite them

Revision ID: 027_manual_attendance_times
Revises: 026_unique_attendance_per_day
Create Date: 2026-07-14

An attendance record's sign-in/sign-out were rebuilt from camera events on every
recalculation, which silently wiped times HR had typed in by hand (needed exactly
when the camera missed the employee). Two per-side flags mark a time as manual so
the recalculation leaves it alone.

Existing rows that HR had already corrected (source='ADMIN') are backfilled as
manual so their times are protected from the next detection.
"""
from alembic import op
import sqlalchemy as sa


revision = '027_manual_attendance_times'
down_revision = '026_unique_attendance_per_day'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "attendance_records",
        sa.Column("sign_in_manual", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column(
        "attendance_records",
        sa.Column("sign_out_manual", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.execute("""
        UPDATE attendance_records
        SET sign_in_manual  = (sign_in_time IS NOT NULL),
            sign_out_manual = (sign_out_time IS NOT NULL)
        WHERE source = 'ADMIN'
    """)


def downgrade():
    op.drop_column("attendance_records", "sign_out_manual")
    op.drop_column("attendance_records", "sign_in_manual")
