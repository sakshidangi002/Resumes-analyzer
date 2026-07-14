"""Add body_embeddings (person Re-ID gallery)

Revision ID: 025_add_body_embeddings
Revises: 024_add_salary_advances
Create Date: 2026-07-13

Purely additive: one new table holding day-scoped body/appearance embeddings used
to keep an employee's identity on their body track when their face is not
visible. No existing table is touched, and nothing here affects attendance.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '025_add_body_embeddings'
down_revision = '024_add_salary_advances'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'body_embeddings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('employee_id', sa.Integer(), nullable=False),
        sa.Column('camera_id', sa.String(length=50), nullable=False),
        sa.Column('day', sa.Date(), nullable=False),
        sa.Column('embedding', sa.LargeBinary(), nullable=False),
        sa.Column('score', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['employee_id'], ['employees.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_body_embeddings_employee_id'), 'body_embeddings', ['employee_id'])
    op.create_index(op.f('ix_body_embeddings_camera_id'), 'body_embeddings', ['camera_id'])
    op.create_index(op.f('ix_body_embeddings_day'), 'body_embeddings', ['day'])
    op.create_index('ix_body_embeddings_day_camera', 'body_embeddings', ['day', 'camera_id'])


def downgrade():
    op.drop_index('ix_body_embeddings_day_camera', table_name='body_embeddings')
    op.drop_index(op.f('ix_body_embeddings_day'), table_name='body_embeddings')
    op.drop_index(op.f('ix_body_embeddings_camera_id'), table_name='body_embeddings')
    op.drop_index(op.f('ix_body_embeddings_employee_id'), table_name='body_embeddings')
    op.drop_table('body_embeddings')
