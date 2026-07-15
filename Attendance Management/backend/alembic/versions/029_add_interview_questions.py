"""Add interview_questions (HR interview-question PDF library)

Revision ID: 029_add_interview_questions
Revises: 028_manual_break_time
Create Date: 2026-07-14

Purely additive: one new table holding uploaded interview-question PDF metadata,
grouped by hiring position. Titles are unique per position. No existing table is
touched.
"""
from alembic import op
import sqlalchemy as sa


revision = '029_add_interview_questions'
down_revision = '028_manual_break_time'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "interview_questions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("position", sa.String(length=100), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("pdf_path", sa.String(length=500), nullable=False),
        sa.Column("pdf_name", sa.String(length=255), nullable=False),
        sa.Column("uploaded_by_user_id", sa.Integer(), nullable=True),
        sa.Column("uploaded_by_name", sa.String(length=150), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["uploaded_by_user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("position", "title", name="uq_interview_position_title"),
    )
    op.create_index("ix_interview_questions_id", "interview_questions", ["id"])
    op.create_index("ix_interview_questions_position", "interview_questions", ["position"])


def downgrade():
    op.drop_index("ix_interview_questions_position", table_name="interview_questions")
    op.drop_index("ix_interview_questions_id", table_name="interview_questions")
    op.drop_table("interview_questions")
