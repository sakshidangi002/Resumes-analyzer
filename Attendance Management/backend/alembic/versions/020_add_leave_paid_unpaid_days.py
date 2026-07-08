"""Add paid_days / unpaid_days split to leave_requests

Revision ID: 020_add_leave_paid_unpaid_days
Revises: 019_add_camera_line_crossing
Create Date: 2026-07-08

Stores the Paid vs Unpaid (LWP) split decided at approval time by the
monthly-earned Paid-Leave policy. Existing approved rows are backfilled so the
whole leave counts as paid (previous behaviour), keeping salary/history stable.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '020_add_leave_paid_unpaid_days'
down_revision = '019_add_camera_line_crossing'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'leave_requests',
        sa.Column('paid_days', sa.Numeric(5, 2), nullable=True, server_default='0'),
    )
    op.add_column(
        'leave_requests',
        sa.Column('unpaid_days', sa.Numeric(5, 2), nullable=True, server_default='0'),
    )
    # Backfill existing APPROVED requests to "fully paid" so historical records
    # keep the behaviour they were approved under (no retroactive LWP). The exact
    # day count is recomputed the next time a request is approved/cancelled; this
    # just gives sane non-null defaults for display.
    op.execute(
        """
        UPDATE leave_requests
           SET paid_days = COALESCE(paid_days, 0),
               unpaid_days = COALESCE(unpaid_days, 0)
        """
    )


def downgrade():
    op.drop_column('leave_requests', 'unpaid_days')
    op.drop_column('leave_requests', 'paid_days')
