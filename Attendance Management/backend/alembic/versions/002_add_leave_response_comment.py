"""Add leave response_comment.

Revision ID: 002_add_leave_response_comment
Revises: 001
Create Date: 2026-03-09
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "002_add_leave_response_comment"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("leave_requests", sa.Column("response_comment", sa.String(length=500), nullable=True))


def downgrade() -> None:
    op.drop_column("leave_requests", "response_comment")

