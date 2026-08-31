"""Per-embedding provenance for the enrolled face gallery.

Revision ID: 035
Revises: 034

`employees.embedding` holds an (N, 512) stack with no record of where any row
came from. That prevents three things the deployment needs:

  * detecting that vectors from different models/detectors have been mixed into
    one comparison (EMBEDDING_MODEL_VERSION existed but was never persisted, and
    the YOLO detector path can emit UNALIGNED embeddings that are not comparable
    with SCRFD-aligned ones);
  * asking "which employees have an embedding captured from the check-in
    camera?", which is the prerequisite for viewpoint-matched enrolment — the
    highest-leverage accuracy fix available when the cameras cannot be moved;
  * finding the individual bad gallery entry that is dragging one employee's
    matching down.

`employees.embedding` is retained and kept in sync as the hot path, so nothing
that reads it needs to change. Existing enrolments are NOT back-filled: their
provenance is genuinely unknown, and inventing `model_version` for them would
defeat the point of recording it. They keep working; re-enrolment populates the
table naturally.
"""

from alembic import op
import sqlalchemy as sa


revision = "035"
down_revision = "034"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "employee_face_embeddings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "employee_id", sa.Integer(),
            sa.ForeignKey("employees.id", ondelete="CASCADE"), nullable=False,
        ),
        sa.Column("embedding", sa.LargeBinary(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("detector", sa.String(length=32), nullable=True),
        sa.Column("aligned", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("source", sa.String(length=16), nullable=False, server_default="upload"),
        sa.Column("camera_id", sa.String(length=50), nullable=True),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("face_px", sa.Float(), nullable=True),
        sa.Column("yaw", sa.Float(), nullable=True),
        sa.Column("pitch", sa.Float(), nullable=True),
        sa.Column("blur_var", sa.Float(), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_employee_face_embeddings_employee_id",
        "employee_face_embeddings", ["employee_id"],
    )
    op.create_index(
        "ix_employee_face_embeddings_camera_id",
        "employee_face_embeddings", ["camera_id"],
    )
    op.create_index(
        "ix_employee_face_embeddings_active",
        "employee_face_embeddings", ["active"],
    )
    op.create_index(
        "ix_employee_face_embeddings_emp_active",
        "employee_face_embeddings", ["employee_id", "active"],
    )


def downgrade() -> None:
    op.drop_index("ix_employee_face_embeddings_emp_active", "employee_face_embeddings")
    op.drop_index("ix_employee_face_embeddings_active", "employee_face_embeddings")
    op.drop_index("ix_employee_face_embeddings_camera_id", "employee_face_embeddings")
    op.drop_index("ix_employee_face_embeddings_employee_id", "employee_face_embeddings")
    op.drop_table("employee_face_embeddings")
