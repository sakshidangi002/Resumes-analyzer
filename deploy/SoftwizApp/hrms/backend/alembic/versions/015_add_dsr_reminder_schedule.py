"""DSR reminder schedule on company_configs (enabled / time / weekdays)"""
from alembic import op
import sqlalchemy as sa


revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "company_configs",
        sa.Column(
            "dsr_reminder_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )
    op.add_column(
        "company_configs",
        sa.Column(
            "dsr_reminder_time",
            sa.String(length=5),
            nullable=False,
            server_default="17:00",
        ),
    )
    op.add_column(
        "company_configs",
        sa.Column(
            "dsr_reminder_weekdays",
            sa.String(length=64),
            nullable=False,
            server_default="mon,tue,wed,thu,fri",
        ),
    )


def downgrade() -> None:
    op.drop_column("company_configs", "dsr_reminder_weekdays")
    op.drop_column("company_configs", "dsr_reminder_time")
    op.drop_column("company_configs", "dsr_reminder_enabled")
