"""Deleting a leave allocation must not be able to erase leave already taken.

`used_days` is the only record that an employee's approved leave was spent
against a given allocation. The approved LeaveRequest rows stay on the books
whatever happens here, so removing an allocation that has been drawn down would
leave the requests pointing at nothing and silently restore days the employee
has already used. The endpoint therefore refuses, and these tests pin that
refusal along with the audit entry the successful path is required to write.
"""
from decimal import Decimal

import pytest
from fastapi import HTTPException

from app.api.routes.leave import delete_allocation
from app.models import Employee, LeaveAllocation, LeaveType


class _Alloc:
    def __init__(self, used, allocated=Decimal("12"), alloc_id=7):
        self.id = alloc_id
        self.employee_id = 55
        self.financial_year_id = 2
        self.leave_type_id = 3
        self.allocated_days = allocated
        self.used_days = used


class _FakeSession:
    """Returns a canned row per model class, and records what was deleted."""

    def __init__(self, rows):
        self._rows = rows
        self.deleted = []
        self.added = []
        self.commits = 0

    def query(self, model):
        session = self

        class _Q:
            def filter(self, *_a, **_k):
                return self

            def first(self):
                return session._rows.get(model)

        return _Q()

    def delete(self, obj):
        self.deleted.append(obj)

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1


class _User:
    id = 1


def _rows(alloc):
    emp = Employee(id=55, first_name="Seema", last_name="Chauhan")
    lt = LeaveType(id=3, code="PL", name="Paid Leave")
    return {LeaveAllocation: alloc, Employee: emp, LeaveType: lt}


@pytest.mark.parametrize("used", [Decimal("0.5"), Decimal("1"), Decimal("12")])
def test_refuses_to_delete_an_allocation_that_has_been_used(used):
    alloc = _Alloc(used=used)
    db = _FakeSession(_rows(alloc))

    with pytest.raises(HTTPException) as excinfo:
        delete_allocation(alloc.id, db=db, current_user=_User())

    assert excinfo.value.status_code == 400
    assert "already been used" in excinfo.value.detail
    assert db.deleted == [], "a used allocation must survive the refusal"
    assert db.commits == 0, "nothing may be committed on the refused path"


def test_deletes_an_untouched_allocation_and_records_it():
    alloc = _Alloc(used=Decimal("0"))
    db = _FakeSession(_rows(alloc))

    result = delete_allocation(alloc.id, db=db, current_user=_User())

    assert db.deleted == [alloc]
    assert "deleted" in result["detail"].lower()
    # The audit entry is the point: an allocation vanishing with no trace is
    # exactly the kind of change an HR audit log exists to explain.
    assert len(db.added) == 1, "the delete must be written to the audit log"
    entry = db.added[0]
    assert entry.action == "LEAVE_ALLOCATION_DELETE"
    assert entry.entity_id == str(alloc.id)
    assert "Seema" in entry.details and "PL" in entry.details


def test_missing_allocation_is_a_404():
    db = _FakeSession({LeaveAllocation: None})

    with pytest.raises(HTTPException) as excinfo:
        delete_allocation(999, db=db, current_user=_User())

    assert excinfo.value.status_code == 404
    assert db.deleted == []


def test_none_used_days_is_treated_as_zero_not_an_error():
    """Legacy rows can carry NULL rather than 0; those are deletable."""
    alloc = _Alloc(used=None)
    db = _FakeSession(_rows(alloc))

    delete_allocation(alloc.id, db=db, current_user=_User())

    assert db.deleted == [alloc]
