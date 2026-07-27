"""Add date_of_marriage to employees."""

from alembic import op
import sqlalchemy as sa


revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "employees",
        sa.Column("date_of_marriage", sa.Date(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("employees", "date_of_marriage")
