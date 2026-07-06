"""Add doorway line-crossing config to cameras

Revision ID: 019_add_camera_line_crossing
Revises: 018_add_staff_type
Create Date: 2026-07-02

"""
from alembic import op
import sqlalchemy as sa


revision = '019_add_camera_line_crossing'
down_revision = '018_add_staff_type'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('cameras', sa.Column('crossing_enabled', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('cameras', sa.Column('line_orientation', sa.String(length=10), nullable=False, server_default='horizontal'))
    op.add_column('cameras', sa.Column('line_position', sa.Float(), nullable=False, server_default='0.5'))
    op.add_column('cameras', sa.Column('entry_direction', sa.String(length=10), nullable=False, server_default='down'))


def downgrade():
    op.drop_column('cameras', 'entry_direction')
    op.drop_column('cameras', 'line_position')
    op.drop_column('cameras', 'line_orientation')
    op.drop_column('cameras', 'crossing_enabled')
