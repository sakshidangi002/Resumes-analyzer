"""add letter_replies table"""
from alembic import op
import sqlalchemy as sa


revision = "003"
down_revision = "002_add_leave_response_comment"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "letter_replies",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("letter_instance_id", sa.Integer, sa.ForeignKey("letter_instances.id"), nullable=False),
        sa.Column("author_employee_id", sa.Integer, sa.ForeignKey("employees.id"), nullable=True),
        sa.Column("author_user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("letter_replies")

