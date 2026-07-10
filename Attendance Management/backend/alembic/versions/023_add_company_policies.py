"""Add company policies (versioned)

Revision ID: 023_add_company_policies
Revises: 022_add_hr_queries
Create Date: 2026-07-10

Purely additive: one new table for versioned company policies. No existing table
is touched.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '023_add_company_policies'
down_revision = '022_add_hr_queries'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'company_policies',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=150), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=True),
        sa.Column('category', sa.String(length=80), nullable=True),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('effective_date', sa.Date(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('attachment_path', sa.String(length=500), nullable=True),
        sa.Column('attachment_name', sa.String(length=255), nullable=True),
        sa.Column('published_by_user_id', sa.Integer(), nullable=True),
        sa.Column('published_by_name', sa.String(length=150), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['published_by_user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_company_policies_name'), 'company_policies', ['name'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_company_policies_name'), table_name='company_policies')
    op.drop_table('company_policies')
