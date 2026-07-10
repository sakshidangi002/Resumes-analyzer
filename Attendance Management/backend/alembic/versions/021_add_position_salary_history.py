"""Add employee position & salary increment history

Revision ID: 021_add_position_salary_history
Revises: 020_add_leave_paid_unpaid_days
Create Date: 2026-07-10

Append-only history of promotions / salary increments. Purely additive: no
existing table is modified, so all existing employee, payroll and attendance
records keep working unchanged.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '021_add_position_salary_history'
down_revision = '020_add_leave_paid_unpaid_days'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'employee_position_salary_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('employee_id', sa.Integer(), nullable=False),
        sa.Column('designation_id', sa.Integer(), nullable=True),
        sa.Column('department_id', sa.Integer(), nullable=True),
        sa.Column('position_title', sa.String(length=150), nullable=True),
        sa.Column('department_name', sa.String(length=150), nullable=True),
        sa.Column('salary', sa.Numeric(12, 2), nullable=True),
        sa.Column('effective_date', sa.Date(), nullable=False),
        sa.Column('reason', sa.String(length=255), nullable=True),
        sa.Column('change_type', sa.String(length=30), nullable=False, server_default='UPDATE'),
        sa.Column('updated_by_user_id', sa.Integer(), nullable=True),
        sa.Column('updated_by_name', sa.String(length=150), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['employee_id'], ['employees.id']),
        sa.ForeignKeyConstraint(['designation_id'], ['designations.id']),
        sa.ForeignKeyConstraint(['department_id'], ['departments.id']),
        sa.ForeignKeyConstraint(['updated_by_user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_employee_position_salary_history_employee_id'),
        'employee_position_salary_history', ['employee_id'], unique=False,
    )
    op.create_index(
        op.f('ix_employee_position_salary_history_effective_date'),
        'employee_position_salary_history', ['effective_date'], unique=False,
    )


def downgrade():
    op.drop_index(
        op.f('ix_employee_position_salary_history_effective_date'),
        table_name='employee_position_salary_history',
    )
    op.drop_index(
        op.f('ix_employee_position_salary_history_employee_id'),
        table_name='employee_position_salary_history',
    )
    op.drop_table('employee_position_salary_history')
