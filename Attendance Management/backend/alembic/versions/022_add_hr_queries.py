"""Add employee ↔ HR query system (hr_queries, hr_query_replies)

Revision ID: 022_add_hr_queries
Revises: 021_add_position_salary_history
Create Date: 2026-07-10

Purely additive: two new tables for the lightweight HR query/inbox feature. No
existing table is touched.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '022_add_hr_queries'
down_revision = '021_add_position_salary_history'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'hr_queries',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('employee_id', sa.Integer(), nullable=False),
        sa.Column('subject', sa.String(length=200), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='OPEN'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['employee_id'], ['employees.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_hr_queries_employee_id'), 'hr_queries', ['employee_id'], unique=False)
    op.create_index(op.f('ix_hr_queries_status'), 'hr_queries', ['status'], unique=False)
    op.create_index(op.f('ix_hr_queries_created_at'), 'hr_queries', ['created_at'], unique=False)

    op.create_table(
        'hr_query_replies',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('query_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('author_name', sa.String(length=150), nullable=True),
        sa.Column('author_role', sa.String(length=20), nullable=True),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['query_id'], ['hr_queries.id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_hr_query_replies_query_id'), 'hr_query_replies', ['query_id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_hr_query_replies_query_id'), table_name='hr_query_replies')
    op.drop_table('hr_query_replies')
    op.drop_index(op.f('ix_hr_queries_created_at'), table_name='hr_queries')
    op.drop_index(op.f('ix_hr_queries_status'), table_name='hr_queries')
    op.drop_index(op.f('ix_hr_queries_employee_id'), table_name='hr_queries')
    op.drop_table('hr_queries')
