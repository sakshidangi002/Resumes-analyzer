"""Anti-proxy control on upload-based attendance (S-04, partial).

/recognize-frame records attendance for whoever is recognised in an uploaded
JPEG. Without a check on WHO may be marked, any logged-in employee could POST a
photo of a colleague and mark them present — a photo attack that needs no camera
at all. There is no liveness model here, so a still cannot be told from a live
capture; what can be enforced is that an upload only marks the uploader.
"""
import pytest

try:
    from app.api.routes import recognition
except ImportError as exc:  # pragma: no cover - depends on the environment
    pytest.skip(
        f"requires the vision stack pulled in by app.api.routes ({exc}); "
        "install 'requirements.txt' to run",
        allow_module_level=True,
    )


class _Role:
    def __init__(self, name):
        self.name = name


class _User:
    def __init__(self, employee_id, roles=(), username="tester", user_id=1):
        self.employee_id = employee_id
        self.roles = [_Role(r) for r in roles]
        self.username = username
        self.id = user_id


class _FakeDB:
    """log_audit commits; capture calls instead of touching a database."""
    def __init__(self):
        self.added = []

    def add(self, entry):
        self.added.append(entry)

    def commit(self):
        pass


def _result(employee_id):
    return {
        "faces": [{"matched": True, "employee_id": employee_id}],
        "attendance": {"employee_id": employee_id, "event_type": "IN"},
    }


@pytest.fixture(autouse=True)
def _no_kiosk_roles(monkeypatch):
    monkeypatch.setattr(recognition, "KIOSK_ATTENDANCE_ROLES", set())


def test_self_check_in_is_allowed():
    user = _User(employee_id=7)
    out = recognition._enforce_upload_attendance_policy(_result(7), user, _FakeDB())
    assert out["attendance"] is not None
    assert out["attendance"]["employee_id"] == 7


def test_marking_someone_else_is_refused_and_audited():
    db = _FakeDB()
    attacker = _User(employee_id=7, username="mallory")
    out = recognition._enforce_upload_attendance_policy(_result(99), attacker, db)

    assert out["attendance"] is None, "proxy attendance was recorded"
    assert "attendance_blocked_reason" in out
    assert db.added, "blocked attempt was not audited"


def test_recognition_result_survives_a_block():
    """Only the attendance side effect is refused; the caller still sees the
    match, so legitimate identification tooling keeps working."""
    out = recognition._enforce_upload_attendance_policy(
        _result(99), _User(employee_id=7), _FakeDB()
    )
    assert out["faces"][0]["employee_id"] == 99


def test_kiosk_roles_may_mark_others_when_configured(monkeypatch):
    """A shared reception tablet is a legitimate workflow — but opt-in only."""
    monkeypatch.setattr(recognition, "KIOSK_ATTENDANCE_ROLES", {"HR"})
    db = _FakeDB()
    hr = _User(employee_id=7, roles=("HR",), username="reception")
    out = recognition._enforce_upload_attendance_policy(_result(99), hr, db)

    assert out["attendance"] is not None
    assert db.added, "kiosk marking should still be audited"


def test_kiosk_exemption_does_not_apply_to_other_roles(monkeypatch):
    monkeypatch.setattr(recognition, "KIOSK_ATTENDANCE_ROLES", {"HR"})
    employee = _User(employee_id=7, roles=("Employee",))
    out = recognition._enforce_upload_attendance_policy(_result(99), employee, _FakeDB())
    assert out["attendance"] is None


def test_no_attendance_in_result_is_passed_through_untouched():
    payload = {"faces": [], "attendance": None}
    assert recognition._enforce_upload_attendance_policy(
        payload, _User(employee_id=7), _FakeDB()
    ) is payload


def test_user_with_no_linked_employee_cannot_mark_anyone():
    """An account not linked to an employee record has no 'self' to check in."""
    out = recognition._enforce_upload_attendance_policy(
        _result(99), _User(employee_id=None), _FakeDB()
    )
    assert out["attendance"] is None
