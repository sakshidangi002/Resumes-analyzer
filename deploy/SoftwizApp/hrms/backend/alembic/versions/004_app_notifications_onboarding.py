"""app_notifications and onboarding_tasks"""
from alembic import op
import sqlalchemy as sa


revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "app_notifications",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("body", sa.Text, nullable=True),
        sa.Column("kind", sa.String(50), nullable=False, server_default="GENERAL"),
        sa.Column("link_path", sa.String(255), nullable=True),
        sa.Column("read_at", sa.DateTime, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_app_notifications_user_id", "app_notifications", ["user_id"])
    op.create_index("ix_app_notifications_created_at", "app_notifications", ["created_at"])

    op.create_table(
        "onboarding_tasks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("employee_id", sa.Integer, sa.ForeignKey("employees.id"), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("is_completed", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("completed_at", sa.DateTime, nullable=True),
        sa.Column("sort_order", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_onboarding_tasks_employee_id", "onboarding_tasks", ["employee_id"])


def downgrade() -> None:
    op.drop_index("ix_onboarding_tasks_employee_id", table_name="onboarding_tasks")
    op.drop_table("onboarding_tasks")
    op.drop_index("ix_app_notifications_created_at", table_name="app_notifications")
    op.drop_index("ix_app_notifications_user_id", table_name="app_notifications")
    op.drop_table("app_notifications")
