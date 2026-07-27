"""Add marital_status to employees."""

from alembic import op
import sqlalchemy as sa


revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "employees",
        sa.Column("marital_status", sa.String(length=20), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("employees", "marital_status")
