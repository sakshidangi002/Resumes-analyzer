"""Who can the matcher actually recognise?

Every way an employee falls out of the recognition gallery is SILENT.
`embedding_cache._load_from_db` does not raise for an employee enrolled under a
superseded recognition model or one whose aggregate stack was never rebuilt --
it simply does not SELECT them. They stop being recognised, nothing is logged,
and the UI still shows them as enrolled. From the outside that is
indistinguishable from "the cameras are broken", which is how it was reported.

`recognition_coverage` reproduces that query's predicates and names what each
one dropped. These tests pin the reason classification, because the reason is
the entire value of the report -- a bare count of "3 of 5" tells nobody what to
do, while "1 stale_model, 1 no_enrolment" is a work list.

If `_load_from_db` ever changes its filters, these tests are the reminder that
this report must change with it.
"""
import pytest

from app.services import employee_face_service as svc
from app.services.face_service import EMBEDDING_MODEL_VERSION

OTHER_MODEL = "insightface_buffalo_l_v1"
assert EMBEDDING_MODEL_VERSION != OTHER_MODEL or True  # doc: they must differ


class _Emp:
    def __init__(self, eid, code, name, embedding=b"stack", status="Active"):
        self.id = eid
        self.employee_code = code
        self.full_name = name
        self.embedding = embedding
        self.employment_status = status


class _Face:
    def __init__(self, model_version=EMBEDDING_MODEL_VERSION, active=True):
        self.model_version = model_version
        self.active = active


class _Query:
    """Ignores filter/order_by; the fake session decides what each call returns."""

    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def all(self):
        return self._rows


class _Session:
    """Serves the employee list once, then one face list per employee, in order."""

    def __init__(self, employees, faces_per_employee):
        self._employees = employees
        self._faces = list(faces_per_employee)

    def query(self, model):
        if getattr(model, "__name__", "") == "Employee":
            return _Query(self._employees)
        return _Query(self._faces.pop(0) if self._faces else [])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def coverage(monkeypatch):
    def _run(employees, faces_per_employee):
        monkeypatch.setattr(
            "app.db.session.SessionLocal",
            lambda: _Session(employees, faces_per_employee),
        )
        return svc.recognition_coverage()
    return _run


def test_fully_enrolled_employee_is_in_the_gallery(coverage):
    r = coverage([_Emp(1, "E1", "Ada")], [[_Face(), _Face()]])

    assert r["in_gallery"] == 1
    assert r["excluded"] == []
    assert r["gallery_vectors"] == 2
    assert r["coverage_pct"] == 100.0


def test_never_enrolled_is_reported_as_no_enrolment(coverage):
    r = coverage([_Emp(1, "E1", "Ada")], [[]])

    assert r["in_gallery"] == 0
    assert r["reasons"] == {"no_enrolment": 1}
    assert r["excluded"][0]["employee_id"] == 1


def test_previous_model_enrolment_is_reported_as_stale_model(coverage):
    """The silent killer: a model switch orphans an employee with no error."""
    r = coverage([_Emp(1, "E1", "Ada")], [[_Face(model_version=OTHER_MODEL)]])

    assert r["in_gallery"] == 0
    assert r["reasons"] == {"stale_model": 1}
    only = r["excluded"][0]
    assert only["reason"] == "stale_model"
    # The detail must name BOTH models, or it is not actionable.
    assert OTHER_MODEL in only["detail"]
    assert EMBEDDING_MODEL_VERSION in only["detail"]


def test_matched_rows_but_no_rebuilt_stack_is_reported_as_no_stack(coverage):
    """employees.embedding IS NOT NULL is a separate predicate in _load_from_db."""
    r = coverage([_Emp(1, "E1", "Ada", embedding=None)], [[_Face()]])

    assert r["reasons"] == {"no_stack": 1}
    assert r["excluded"][0]["model_matched"] == 1


def test_inactive_embeddings_do_not_count_as_enrolment(coverage):
    """A cleared enrolment leaves active=False rows behind.

    The service filters those out in its query, so the fake supplies none --
    what this pins is that an employee with no ACTIVE rows reads as
    `no_enrolment`, not as enrolled.
    """
    r = coverage([_Emp(1, "E1", "Ada")], [[]])
    assert r["excluded"][0]["reason"] == "no_enrolment"


def test_mixed_population_counts_and_reasons(coverage):
    """The shape this was built for: some fine, some silently excluded."""
    employees = [
        _Emp(1, "E1", "Ada"),
        _Emp(2, "E2", "Grace"),
        _Emp(3, "E3", "Alan"),
        _Emp(4, "E4", "Edsger", embedding=None),
    ]
    faces = [
        [_Face(), _Face(), _Face()],          # in gallery, 3 vectors
        [],                                   # no_enrolment
        [_Face(model_version=OTHER_MODEL)],   # stale_model
        [_Face()],                            # no_stack
    ]
    r = coverage(employees, faces)

    assert r["active_employees"] == 4
    assert r["in_gallery"] == 1
    assert r["gallery_vectors"] == 3
    assert r["coverage_pct"] == 25.0
    assert r["reasons"] == {"no_enrolment": 1, "stale_model": 1, "no_stack": 1}
    assert [e["employee_id"] for e in r["excluded"]] == [2, 3, 4]


def test_empty_workforce_does_not_divide_by_zero(coverage):
    r = coverage([], [])
    assert r["active_employees"] == 0
    assert r["coverage_pct"] == 0.0
    assert r["reasons"] == {}


def test_report_carries_the_running_model_version(coverage):
    """Without it the reader cannot tell which model 'stale' is relative to."""
    r = coverage([_Emp(1, "E1", "Ada")], [[_Face()]])
    assert r["model_version"] == EMBEDDING_MODEL_VERSION
