"""Add free-text staff_type to employees

Revision ID: 018_add_staff_type
Revises: 017_add_camera_tracking_config
Create Date: 2026-07-02

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '018_add_staff_type'
down_revision = '017_add_camera_tracking_config'
branch_labels = None
depends_on = None


def upgrade():
    # Free-text category so non-employee staff (housekeeping, security, driver…)
    # can be registered for face attendance. Existing rows default to "Employee".
    op.add_column(
        'employees',
        sa.Column('staff_type', sa.String(length=50), nullable=True, server_default='Employee'),
    )
    op.execute("UPDATE employees SET staff_type = 'Employee' WHERE staff_type IS NULL")


def downgrade():
    op.drop_column('employees', 'staff_type')
