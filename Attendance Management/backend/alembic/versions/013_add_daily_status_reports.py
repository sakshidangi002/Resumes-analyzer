"""daily_status_reports table"""
from alembic import op
import sqlalchemy as sa


revision = "013"
down_revision = "012_merge"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "daily_status_reports",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "employee_id",
            sa.Integer,
            sa.ForeignKey("employees.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("report_date", sa.Date, nullable=False),
        sa.Column("project_work", sa.String(255), nullable=True),
        sa.Column("work_location", sa.String(50), nullable=True, server_default="Office"),
        sa.Column("total_hours", sa.Numeric(5, 2), nullable=True),
        sa.Column("work_done", sa.Text, nullable=False),
        sa.Column("plan_for_tomorrow", sa.Text, nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="DRAFT"),
        sa.Column("submitted_at", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("employee_id", "report_date", name="uq_dsr_employee_date"),
    )
    op.create_index(
        "ix_daily_status_reports_employee_id",
        "daily_status_reports",
        ["employee_id"],
    )
    op.create_index(
        "ix_daily_status_reports_report_date",
        "daily_status_reports",
        ["report_date"],
    )
    op.create_index(
        "ix_daily_status_reports_status",
        "daily_status_reports",
        ["status"],
    )


def downgrade() -> None:
    op.drop_index("ix_daily_status_reports_status", table_name="daily_status_reports")
    op.drop_index("ix_daily_status_reports_report_date", table_name="daily_status_reports")
    op.drop_index("ix_daily_status_reports_employee_id", table_name="daily_status_reports")
    op.drop_table("daily_status_reports")
