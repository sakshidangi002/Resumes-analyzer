"""Add employees important document fields (PAN/Aadhar/Passport/DL)."""
from alembic import op
import sqlalchemy as sa


revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("employees", sa.Column("pan_number", sa.String(length=50), nullable=True))
    op.add_column("employees", sa.Column("aadhar_number", sa.String(length=50), nullable=True))
    op.add_column("employees", sa.Column("passport_number", sa.String(length=50), nullable=True))
    op.add_column("employees", sa.Column("passport_expiry_date", sa.Date(), nullable=True))
    op.add_column("employees", sa.Column("driving_license_number", sa.String(length=50), nullable=True))
    op.add_column("employees", sa.Column("driving_license_expiry_date", sa.Date(), nullable=True))


def downgrade() -> None:
    op.drop_column("employees", "driving_license_expiry_date")
    op.drop_column("employees", "driving_license_number")
    op.drop_column("employees", "passport_expiry_date")
    op.drop_column("employees", "passport_number")
    op.drop_column("employees", "aadhar_number")
    op.drop_column("employees", "pan_number")

