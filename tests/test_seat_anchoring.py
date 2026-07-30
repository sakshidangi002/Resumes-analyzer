"""Identify a person by BODY position when their face is not visible.

The room cameras see people from above and behind — often only hair. Body
detection still works, so a person is always boxed; the question is whether a
name can be attached. Two signals were bundled together and both switched off:

  * OSNet body appearance (Re-ID) — genuinely unreliable here. Measured: a true
    match scored 0.71-0.82 and a FALSE one 0.77. Correctly disabled.
  * Seat position — where the person was when a face DID confirm them. Strong
    in a fixed-desk room, and it survives losing the track entirely.

Disabling the first should never have disabled the second.
"""
import pytest
from app.services.person_tracker import PersonTrack


def _track(**kw):
    return PersonTrack(track_id=kw.pop("track_id", 1), box=kw.pop("box", (0, 0, 10, 10)), **kw)


def test_face_identity_records_its_source():
    t = _track()
    t.bind_identity(7, "Adarsh", "E7", True, 0.78, source="face")
    assert (t.employee_id, t.identity_source, t.matched) == (7, "face", True)


def test_seat_identity_records_its_source():
    t = _track()
    t.bind_identity(7, "Adarsh", "E7", True, 0.0, source="seat")
    assert t.identity_source == "seat"


def test_seat_guess_never_overwrites_a_face_match():
    """Someone standing at a colleague's desk must not be renamed to them.

    A face was actually seen; a positional guess is weaker evidence and must
    lose.
    """
    t = _track()
    t.bind_identity(7, "Adarsh", "E7", True, 0.78, source="face")
    t.bind_identity(9, "Saloni", "E9", True, 0.0, source="seat")
    assert t.employee_id == 7
    assert t.employee_name == "Adarsh"
    assert t.identity_source == "face"


def test_face_match_upgrades_a_seat_guess():
    """The reverse direction MUST work: once the face is finally seen, it wins."""
    t = _track()
    t.bind_identity(9, "Saloni", "E9", True, 0.0, source="seat")
    t.bind_identity(7, "Adarsh", "E7", True, 0.74, source="face")
    assert t.employee_id == 7
    assert t.identity_source == "face"


def test_failed_read_never_erases_an_established_identity():
    t = _track()
    t.bind_identity(7, "Adarsh", "E7", True, 0.78, source="face")
    t.bind_identity(None, None, None, False, 0.0)
    assert t.employee_id == 7


def test_display_info_exposes_the_source_for_the_overlay():
    """The overlay colours seat-derived names differently, so it needs this."""
    t = _track()
    t.bind_identity(7, "Adarsh", "E7", True, 0.0, source="seat")
    assert t.get_display_info()["identity_source"] == "seat"


def test_enroll_accepts_a_seat_without_a_body_embedding():
    """Seat learning must not require the disabled Re-ID gallery.

    enroll() used to `return` immediately when embedding was None, so with
    Re-ID off nothing was ever learned — including seats.
    """
    import inspect

    from app.services.identity_manager import GlobalIdentityManager

    src = inspect.getsource(GlobalIdentityManager.enroll)
    assert "if embedding is None:\n            return" not in src, (
        "enroll() still bails out without an embedding — seat-only learning "
        "is impossible and seat anchoring can never work with Re-ID disabled"
    )
    assert "_remember_seat" in src


def test_seat_anchoring_is_enabled_by_default():
    from app.core.config import get_settings

    assert get_settings().seat_anchor_enabled is True


def test_seat_anchoring_is_independent_of_reid():
    """The whole point: one flag must not gate the other."""
    import inspect

    camera_service = pytest.importorskip("app.services.camera_service")
    src = inspect.getsource(camera_service._RecognitionThread._anchor_identities)
    assert "_SEAT_ANCHOR" in src
    assert "seat_match" in src
