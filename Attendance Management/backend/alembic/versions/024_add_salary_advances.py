"""Add salary advances

Revision ID: 024_add_salary_advances
Revises: 023_add_company_policies
Create Date: 2026-07-10

Purely additive: one new table for salary advances recovered on the next payroll
run. No existing table is modified.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '024_add_salary_advances'
down_revision = '023_add_company_policies'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'salary_advances',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('employee_id', sa.Integer(), nullable=False),
        sa.Column('amount', sa.Numeric(12, 2), nullable=False),
        sa.Column('date_taken', sa.Date(), nullable=False),
        sa.Column('reason', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='PENDING'),
        sa.Column('deducted_period_id', sa.Integer(), nullable=True),
        sa.Column('deducted_at', sa.DateTime(), nullable=True),
        sa.Column('created_by_user_id', sa.Integer(), nullable=True),
        sa.Column('created_by_name', sa.String(length=150), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['employee_id'], ['employees.id']),
        sa.ForeignKeyConstraint(['deducted_period_id'], ['payroll_periods.id']),
        sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_salary_advances_employee_id'), 'salary_advances', ['employee_id'], unique=False)
    op.create_index(op.f('ix_salary_advances_status'), 'salary_advances', ['status'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_salary_advances_status'), table_name='salary_advances')
    op.drop_index(op.f('ix_salary_advances_employee_id'), table_name='salary_advances')
    op.drop_table('salary_advances')
