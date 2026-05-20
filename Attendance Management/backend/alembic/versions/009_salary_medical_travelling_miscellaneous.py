"""Add medical, travelling, miscellaneous to salary_structures."""

from alembic import op
import sqlalchemy as sa


revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "salary_structures",
        sa.Column("medical", sa.Numeric(precision=12, scale=2), nullable=False, server_default="0"),
    )
    op.add_column(
        "salary_structures",
        sa.Column("travelling", sa.Numeric(precision=12, scale=2), nullable=False, server_default="0"),
    )
    op.add_column(
        "salary_structures",
        sa.Column("miscellaneous", sa.Numeric(precision=12, scale=2), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("salary_structures", "miscellaneous")
    op.drop_column("salary_structures", "travelling")
    op.drop_column("salary_structures", "medical")
