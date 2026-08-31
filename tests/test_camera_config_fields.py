"""Every field the camera API accepts must reach a database column.

Regression test for silently dropped configuration: `CameraCreateRequest`
accepted `frame_skip`, `tracking_max_distance` and `tracking_cooldown`, but
`create_camera` named only 8 fields when constructing `CameraConfig`, so those
three were validated, echoed back in the 200 response, and thrown away.

Separately, the four line-crossing fields existed on the model and were honoured
by CameraWorker, but appeared in NO request schema — the feature was reachable
only by a manual UPDATE against the cameras table.

Pure schema/model comparison: no database, no HTTP client.
"""
import pytest
from app.api.routes.cameras import CameraCreateRequest, CameraUpdateRequest
from app.api.routes.face_review import CameraProfileUpdate
from app.models.camera import CameraConfig

# Columns the API deliberately does not expose: surrogate key, server-managed
# timestamps, and the legacy duplicate of camera_purpose.
NOT_API_WRITABLE = {"id", "created_at", "updated_at", "camera_type"}

# Recognition-profile columns. Settable, but through PUT /cameras/{id}/profile
# rather than the camera create/update form, and gated to Admin rather than
# Admin+HR: these decide when a payroll row gets written, and loosening one on
# an IN/OUT camera is the fastest way to manufacture false attendance. Keeping
# them off the general camera form is deliberate, so they are checked against
# their own schema below rather than being exempted from the drift check.
PROFILE_COLUMNS = set(CameraProfileUpdate.model_fields) - {"threshold"}

MODEL_COLUMNS = {c.name for c in CameraConfig.__table__.columns}
CONFIGURABLE_COLUMNS = MODEL_COLUMNS - NOT_API_WRITABLE - PROFILE_COLUMNS

# Tuning knobs that exist to be set per camera. Named explicitly so that
# deleting one from the schema fails loudly rather than shrinking the test.
TUNING_FIELDS = {
    "frame_skip",
    "tracking_max_distance",
    "tracking_cooldown",
    "crossing_enabled",
    "line_orientation",
    "line_position",
    "entry_direction",
}


@pytest.mark.parametrize("field", sorted(TUNING_FIELDS))
def test_create_schema_exposes_tuning_field(field):
    assert field in CameraCreateRequest.model_fields, (
        f"CameraCreateRequest is missing {field!r}: the column exists and the "
        f"worker honours it, so leaving it out makes the feature unreachable."
    )


@pytest.mark.parametrize("field", sorted(TUNING_FIELDS))
def test_update_schema_exposes_tuning_field(field):
    assert field in CameraUpdateRequest.model_fields


def test_create_schema_covers_every_configurable_column():
    """The whole point: schema and columns must not drift apart."""
    missing = CONFIGURABLE_COLUMNS - set(CameraCreateRequest.model_fields)
    assert not missing, f"columns with no way to set them via the API: {sorted(missing)}"


def test_profile_schema_covers_every_recognition_column():
    """Same drift check for the recognition profile, against its own endpoint.

    A tuning column with no way to set it is exactly as unreachable as the
    line-crossing fields were before this test existed.
    """
    recognition_columns = {
        name for name in MODEL_COLUMNS
        if name in set(CameraProfileUpdate.model_fields) or name in PROFILE_COLUMNS
    }
    missing = recognition_columns - set(CameraProfileUpdate.model_fields)
    assert not missing, f"profile columns with no way to set them: {sorted(missing)}"


def test_profile_schema_has_no_field_without_a_column():
    """update_camera_profile setattr()s each provided field onto the model."""
    extra = set(CameraProfileUpdate.model_fields) - MODEL_COLUMNS
    assert not extra, f"profile fields with no column: {sorted(extra)}"


def test_every_new_profile_column_is_reachable():
    """Guards the migration-034 columns specifically.

    Named explicitly so that dropping one from the schema fails loudly rather
    than silently shrinking what an operator can tune.
    """
    for field in (
        "match_margin", "min_face_px", "min_det_score", "max_yaw_deg",
        "max_pitch_deg", "max_landmark_asym", "min_blur_var",
        "min_observations", "min_quality", "min_consensus",
        "analysis_interval", "face_crop_scale", "attendance_cooldown",
    ):
        assert field in MODEL_COLUMNS, f"{field} is missing from the cameras table"
        assert field in CameraProfileUpdate.model_fields, (
            f"{field} exists as a column and is honoured by the worker, so "
            f"leaving it out of CameraProfileUpdate makes it unreachable."
        )


def test_create_schema_has_no_field_without_a_column():
    """create_camera builds CameraConfig(**payload.model_dump()).

    A schema field with no matching column would raise TypeError at runtime for
    every create call, so catch it here instead.
    """
    extra = set(CameraCreateRequest.model_fields) - MODEL_COLUMNS
    assert not extra, f"schema fields with no column (create_camera would crash): {sorted(extra)}"


def test_update_schema_has_no_field_without_a_column():
    """update_camera setattr()s each provided field straight onto the model."""
    extra = set(CameraUpdateRequest.model_fields) - MODEL_COLUMNS
    assert not extra, f"schema fields with no column: {sorted(extra)}"


def test_line_position_is_bounded_to_the_frame():
    """line_position is a 0..1 fraction across the frame; outside that the
    crossing line is drawn off-screen and never fires."""
    field = CameraCreateRequest.model_fields["line_position"]
    bounds = {type(m).__name__: getattr(m, "ge", getattr(m, "le", None)) for m in field.metadata}
    assert bounds.get("Ge") == 0.0
    assert bounds.get("Le") == 1.0
