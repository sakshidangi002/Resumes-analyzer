"""Add branch_name to employee_bank_details."""

from alembic import op
import sqlalchemy as sa


revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("employee_bank_details", sa.Column("branch_name", sa.String(length=100), nullable=True))


def downgrade() -> None:
    op.drop_column("employee_bank_details", "branch_name")

