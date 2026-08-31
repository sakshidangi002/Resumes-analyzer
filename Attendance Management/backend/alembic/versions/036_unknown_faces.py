"""Queue of faces the cameras saw but could not name.

Revision ID: 036
Revises: 035

Unmatched faces were discarded, which left the deployment with no way to find
out WHICH employees the cameras keep failing on. Since the cameras cannot be
repositioned, global threshold tuning saturates quickly and the remaining errors
are concentrated in a few people whose enrolled photos do not resemble what
their camera sees. This table is what makes those people findable, and its
review flow doubles as the cheapest route to viewpoint-matched enrolment.

Rows hold a biometric embedding and a face crop and must be retention-managed —
see services/unknown_faces.purge_older_than.
"""

from alembic import op
import sqlalchemy as sa


revision = "036"
down_revision = "035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "unknown_faces",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("camera_id", sa.String(length=50), nullable=False),
        sa.Column("captured_at", sa.DateTime(), nullable=False),
        sa.Column("embedding", sa.LargeBinary(), nullable=False),
        sa.Column("crop_path", sa.String(length=300), nullable=True),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("face_px", sa.Float(), nullable=True),
        sa.Column("yaw", sa.Float(), nullable=True),
        sa.Column("pitch", sa.Float(), nullable=True),
        sa.Column("blur_var", sa.Float(), nullable=True),
        sa.Column("best_score", sa.Float(), nullable=True),
        sa.Column("best_margin", sa.Float(), nullable=True),
        sa.Column(
            "best_employee_id", sa.Integer(),
            sa.ForeignKey("employees.id", ondelete="SET NULL"), nullable=True,
        ),
        sa.Column("cluster_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column(
            "assigned_employee_id", sa.Integer(),
            sa.ForeignKey("employees.id", ondelete="SET NULL"), nullable=True,
        ),
        sa.Column(
            "reviewed_by", sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
        ),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_unknown_faces_camera_id", "unknown_faces", ["camera_id"])
    op.create_index("ix_unknown_faces_captured_at", "unknown_faces", ["captured_at"])
    op.create_index("ix_unknown_faces_cluster_id", "unknown_faces", ["cluster_id"])
    op.create_index("ix_unknown_faces_status", "unknown_faces", ["status"])
    op.create_index(
        "ix_unknown_faces_status_camera", "unknown_faces", ["status", "camera_id"]
    )
    op.create_index(
        "ix_unknown_faces_cluster_status", "unknown_faces", ["cluster_id", "status"]
    )


def downgrade() -> None:
    op.drop_index("ix_unknown_faces_cluster_status", "unknown_faces")
    op.drop_index("ix_unknown_faces_status_camera", "unknown_faces")
    op.drop_index("ix_unknown_faces_status", "unknown_faces")
    op.drop_index("ix_unknown_faces_cluster_id", "unknown_faces")
    op.drop_index("ix_unknown_faces_captured_at", "unknown_faces")
    op.drop_index("ix_unknown_faces_camera_id", "unknown_faces")
    op.drop_table("unknown_faces")
