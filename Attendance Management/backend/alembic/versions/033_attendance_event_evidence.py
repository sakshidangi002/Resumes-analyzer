"""Record the recognition evidence behind each camera attendance event.

Revision ID: 033
Revises: 032

An attendance event stored only `camera_id` — no match score, no margin, no
track id, no image. Two consequences:

  * A disputed record could not be adjudicated. "The camera says you left at
    14:05" with nothing to inspect is not an answer.
  * Every recognition threshold in the CCTV pipeline was set from anecdote
    (the code comments cite a single mis-labelled person), because there was
    no data on how true and false matches actually separate.

All columns are nullable: rows written before this migration have no evidence
and must not have any invented for them. A NULL score means "recorded before
evidence capture existed", which is different from "scored zero".
"""

from alembic import op
import sqlalchemy as sa


revision = "033"
down_revision = "032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "attendance_events",
        sa.Column("match_score", sa.Float(), nullable=True),
    )
    op.add_column(
        "attendance_events",
        sa.Column("match_margin", sa.Float(), nullable=True),
    )
    op.add_column(
        "attendance_events",
        sa.Column("track_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "attendance_events",
        sa.Column("snapshot_path", sa.String(length=300), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("attendance_events", "snapshot_path")
    op.drop_column("attendance_events", "track_id")
    op.drop_column("attendance_events", "match_margin")
    op.drop_column("attendance_events", "match_score")
