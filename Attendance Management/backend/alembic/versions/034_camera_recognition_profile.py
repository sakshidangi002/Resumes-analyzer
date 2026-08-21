"""Per-camera recognition profile.

Revision ID: 034
Revises: 033

Every recognition tunable in the CCTV pipeline was a process-wide constant read
from an environment variable (CCTV_MIN_FACE_PX, CCTV_CONFIRM_FRAMES,
CCTV_IDENT_CONFIRM, CCTV_ANALYSIS_INTERVAL, MATCH margin from settings, ...).
One value therefore had to serve four cameras with irreconcilable requirements:

  * the check-in and check-out cameras, whose matches become payroll rows and
    where a false accept is unacceptable;
  * two ceiling-mounted room cameras that see seated people from above, can
    never mark attendance, and where a missed person defeats the camera's only
    purpose.

The single global compromise is visible in the code history: CCTV_MIN_FACE_PX
walked 45 -> 24 -> 16 because each tightening blinded the room cameras, and the
loosening it forced applied equally to the attendance cameras.

Every column is NULLABLE and NULL means "inherit the default for this camera's
purpose" (services/camera_profile.py). Existing rows therefore keep working with
no data migration, and an operator overrides only what they have actually
measured with scripts/calibrate_thresholds.py.
"""

from alembic import op
import sqlalchemy as sa


revision = "034"
down_revision = "033"
branch_labels = None
depends_on = None


_COLUMNS = (
    ("match_margin", sa.Float()),
    ("min_face_px", sa.Integer()),
    ("min_det_score", sa.Float()),
    ("max_yaw_deg", sa.Float()),
    ("max_pitch_deg", sa.Float()),
    ("max_landmark_asym", sa.Float()),
    ("min_blur_var", sa.Float()),
    ("min_observations", sa.Integer()),
    ("min_quality", sa.Float()),
    ("min_consensus", sa.Float()),
    ("analysis_interval", sa.Float()),
    ("face_crop_scale", sa.Integer()),
    ("attendance_cooldown", sa.Float()),
)


def upgrade() -> None:
    for name, coltype in _COLUMNS:
        op.add_column("cameras", sa.Column(name, coltype, nullable=True))


def downgrade() -> None:
    for name, _coltype in reversed(_COLUMNS):
        op.drop_column("cameras", name)
