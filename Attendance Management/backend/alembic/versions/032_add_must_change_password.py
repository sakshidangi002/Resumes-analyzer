"""Require a password change after temporary-password resets.

Revision ID: 032
Revises: 031_drop_camera_seat_assignments
"""

from alembic import op
import sqlalchemy as sa


revision = "032"
down_revision = "031_drop_camera_seat_assignments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "must_change_password",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "must_change_password")
