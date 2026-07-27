"""Add employees.date_of_leaving (optional last working day)."""
from alembic import op
import sqlalchemy as sa


revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("employees", sa.Column("date_of_leaving", sa.Date(), nullable=True))


def downgrade() -> None:
    op.drop_column("employees", "date_of_leaving")
