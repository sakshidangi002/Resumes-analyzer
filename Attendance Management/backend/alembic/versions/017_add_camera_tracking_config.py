"""Add camera tracking configuration fields

Revision ID: 017_add_camera_tracking_config
Revises: 016
Create Date: 2026-07-01

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '017_add_camera_tracking_config'
down_revision = '016'
branch_labels = None
depends_on = None


def upgrade():
    # Add new tracking configuration fields to cameras table
    op.add_column('cameras', sa.Column('frame_skip', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('cameras', sa.Column('tracking_max_distance', sa.Float(), nullable=False, server_default='100.0'))
    op.add_column('cameras', sa.Column('tracking_cooldown', sa.Float(), nullable=False, server_default='3.0'))


def downgrade():
    # Remove the new fields
    op.drop_column('cameras', 'tracking_cooldown')
    op.drop_column('cameras', 'tracking_max_distance')
    op.drop_column('cameras', 'frame_skip')
