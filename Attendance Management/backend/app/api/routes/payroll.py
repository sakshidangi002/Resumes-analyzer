"""Payroll periods, salary structure, run payroll, payslips."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models import User, SalaryStructure, PayrollPeriod, Payslip, Employee, SalaryAdvance
from app.models.user import Role, user_roles

from app.schemas.payroll import (
    SalaryStructureCreate,
    SalaryStructureUpdate,
    SalaryStructureResponse,
    PayrollPeriodResponse,
    PayrollPeriodUpdate,
    PayslipResponse,
    PayslipCreate,
    PayslipUpdate,
    SalaryAdvanceCreate,
    SalaryAdvanceResponse,
)
from app.api.deps import require_roles
from app.services.payroll_service import run_payroll_for_period

router = APIRouter()


def _employee_has_role(db: Session, employee_id: int, role_name: str) -> bool:
  u = (
      db.query(User)
      .join(user_roles, User.id == user_roles.c.user_id)
      .join(Role, Role.id == user_roles.c.role_id)
      .filter(User.employee_id == employee_id, Role.name == role_name)
      .first()
  )
  return u is not None


@router.get("/periods", response_model=list[PayrollPeriodResponse])
def list_periods(db: Session = Depends(get_db)):
    return db.query(PayrollPeriod).order_by(PayrollPeriod.year.desc(), PayrollPeriod.month.desc()).all()


@router.post("/periods", response_model=PayrollPeriodResponse)
def create_period(
    month: int,
    year: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    if month < 1 or month > 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12")
    existing = (
        db.query(PayrollPeriod)
        .filter(PayrollPeriod.month == month, PayrollPeriod.year == year)
        .first()
    )
    if existing:
        return existing
    period = PayrollPeriod(month=month, year=year, status="OPEN")
    db.add(period)
    db.commit()
    db.refresh(period)
    return period


@router.patch("/periods/{period_id}", response_model=PayrollPeriodResponse)
def update_period(
    period_id: int,
    data: PayrollPeriodUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    period = db.query(PayrollPeriod).filter(PayrollPeriod.id == period_id).first()
    if not period:
        raise HTTPException(status_code=404, detail="Period not found")
    if data.status is not None:
        if data.status not in ("OPEN", "PROCESSED", "LOCKED"):
            raise HTTPException(status_code=400, detail="Invalid status")
        period.status = data.status
    if data.month is not None:
        if data.month < 1 or data.month > 12:
            raise HTTPException(status_code=400, detail="Month must be 1-12")
        period.month = data.month
    if data.year is not None:
        period.year = data.year
    db.commit()
    db.refresh(period)
    return period


@router.delete("/periods/{period_id}")
def delete_period(
    period_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    period = db.query(PayrollPeriod).filter(PayrollPeriod.id == period_id).first()
    if not period:
        raise HTTPException(status_code=404, detail="Period not found")

    # A LOCKED period has already been paid out. Deleting it destroys every
    # payslip for that month with no audit trail and no soft-delete, so refuse.
    if str(getattr(period, "status", "") or "").upper() == "LOCKED":
        raise HTTPException(
            status_code=409,
            detail="This payroll period is locked and cannot be deleted. "
                   "Unlock it first if you really need to remove it.",
        )

    # Cascade delete is enabled in the model, so deleting the period will delete all its payslips.
    # That cascade is ORM-level only; anything else still referencing these
    # payslips (e.g. salary advance repayments) raises an FK error. Roll back so
    # the payslips aren't left destroyed by a half-applied delete, and report
    # why instead of surfacing a bare 500.
    try:
        db.delete(period)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="This period still has records linked to it (for example salary "
                   "advance repayments). Remove those first, then delete the period.",
        )
    except Exception:
        db.rollback()
        raise
    return {"message": "Deleted"}


@router.post("/periods/{period_id}/run", response_model=list[PayslipResponse])
def run_payroll(
    period_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    period = db.query(PayrollPeriod).filter(PayrollPeriod.id == period_id).first()
    if not period:
        raise HTTPException(status_code=404, detail="Period not found")
    slips = run_payroll_for_period(db, period.month, period.year)
    return slips


@router.post("/salary-structures", response_model=SalaryStructureResponse)
def create_salary_structure(
    data: SalaryStructureCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["HR"])),
):
    s = SalaryStructure(
        employee_id=data.employee_id,
        basic=data.basic,
        hra=data.hra,
        medical=data.medical,
        travelling=data.travelling,
        miscellaneous=data.miscellaneous,
        allowances=data.allowances,
        deductions=data.deductions,
        effective_from=data.effective_from,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


@router.get("/salary-structures", response_model=list[SalaryStructureResponse])
def list_salary_structures(
    employee_id: int | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Employee"])),
):
    role_names = [r.name for r in current_user.roles]
    is_admin = "Admin" in role_names
    is_hr = "HR" in role_names

    q = db.query(SalaryStructure)

    # Employee (no HR/Admin): only own payroll
    if "Employee" in role_names and not (is_admin or is_hr):
        if current_user.employee_id is None:
            return []
        q = q.filter(SalaryStructure.employee_id == current_user.employee_id)
    elif employee_id is not None:
        q = q.filter(SalaryStructure.employee_id == employee_id)

    # HR & Admin: can view all salary structures
    return q.order_by(SalaryStructure.effective_from.desc()).all()


@router.patch("/salary-structures/{structure_id}", response_model=SalaryStructureResponse)
def update_salary_structure(
    structure_id: int,
    data: SalaryStructureUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["HR"])),
):
    s = db.query(SalaryStructure).filter(SalaryStructure.id == structure_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Salary structure not found")
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(s, k, v)
    db.commit()
    db.refresh(s)
    return s


@router.delete("/salary-structures/{structure_id}")
def delete_salary_structure(
    structure_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["HR"])),
):
    s = db.query(SalaryStructure).filter(SalaryStructure.id == structure_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Salary structure not found")
    db.delete(s)
    db.commit()
    return {"message": "Deleted"}


# ---------- Salary advances ----------
@router.post("/advances", response_model=SalaryAdvanceResponse)
def create_advance(
    data: SalaryAdvanceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Record a salary advance taken by an employee (recovered on next payroll)."""
    if data.amount is None or data.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be greater than 0")
    emp = db.query(Employee).filter(Employee.id == data.employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    adv = SalaryAdvance(
        employee_id=data.employee_id,
        amount=data.amount,
        date_taken=data.date_taken,
        reason=data.reason,
        status="PENDING",
        created_by_user_id=current_user.id,
        created_by_name=current_user.username,
    )
    db.add(adv)
    db.commit()
    db.refresh(adv)
    return adv


@router.get("/advances", response_model=list[SalaryAdvanceResponse])
def list_advances(
    employee_id: int | None = Query(None),
    status: str | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Employee"])),
):
    role_names = [r.name for r in current_user.roles]
    is_priv = any(r in role_names for r in ("Admin", "HR"))
    q = db.query(SalaryAdvance)
    if not is_priv:
        # Employee: only own advances
        if current_user.employee_id is None:
            return []
        q = q.filter(SalaryAdvance.employee_id == current_user.employee_id)
    elif employee_id is not None:
        q = q.filter(SalaryAdvance.employee_id == employee_id)
    if status:
        q = q.filter(SalaryAdvance.status == status)
    return q.order_by(SalaryAdvance.date_taken.desc(), SalaryAdvance.id.desc()).all()


@router.delete("/advances/{advance_id}")
def delete_advance(
    advance_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    adv = db.query(SalaryAdvance).filter(SalaryAdvance.id == advance_id).first()
    if not adv:
        raise HTTPException(status_code=404, detail="Advance not found")
    if adv.status == "DEDUCTED":
        raise HTTPException(
            status_code=400,
            detail="This advance was already recovered in payroll and cannot be deleted.",
        )
    db.delete(adv)
    db.commit()
    return {"message": "Deleted"}


@router.get("/payslips", response_model=list[PayslipResponse])
def list_payslips(
    employee_id: int | None = Query(None),
    period_id: int | None = Query(None),
    year: int | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Employee"])),
):
    q = db.query(Payslip)
    role_names = [r.name for r in current_user.roles]
    is_admin = "Admin" in role_names
    is_hr = "HR" in role_names

    # Employee: only own payslips
    if "Employee" in role_names and not (is_admin or is_hr):
        if current_user.employee_id is None:
            return []
        q = q.filter(Payslip.employee_id == current_user.employee_id)
    elif employee_id is not None:
        q = q.filter(Payslip.employee_id == employee_id)

    if period_id is not None:
        q = q.filter(Payslip.payroll_period_id == period_id)
    if year is not None:
        q = q.join(PayrollPeriod).filter(PayrollPeriod.year == year)

    # HR & Admin: can view all remaining records
    return q.order_by(Payslip.generated_at.desc()).all()


@router.get("/payslips/{payslip_id}", response_model=PayslipResponse)
def get_payslip(
    payslip_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Employee"])),
):
    slip = db.query(Payslip).filter(Payslip.id == payslip_id).first()
    if not slip:
        raise HTTPException(status_code=404, detail="Payslip not found")
    role_names = [r.name for r in current_user.roles]
    is_admin = "Admin" in role_names
    is_hr = "HR" in role_names
    if "Employee" in role_names and not (is_admin or is_hr) and current_user.employee_id != slip.employee_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return slip


@router.post("/payslips", response_model=PayslipResponse)
def create_payslip(
    data: PayslipCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin"])),
):
    """
    Manual payslip creation is disabled for normal use.
    Salary must come from attendance via the payroll run.
    """
    raise HTTPException(status_code=403, detail="Manual payslip creation is disabled; use payroll run.")


@router.patch("/payslips/{payslip_id}", response_model=PayslipResponse)
def update_payslip(
    payslip_id: int,
    data: PayslipUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin"])),
):
    """
    Manual payslip update is disabled for normal use.
    Salary must come from attendance via the payroll run.
    """
    raise HTTPException(status_code=403, detail="Manual payslip update is disabled; use payroll run.")


@router.delete("/payslips/{payslip_id}")
def delete_payslip(
    payslip_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """
    Allow Admin/HR to delete an individual payslip.
    Typically used when rerunning payroll or fixing mistakes.
    """
    slip = db.query(Payslip).filter(Payslip.id == payslip_id).first()
    if not slip:
        raise HTTPException(status_code=404, detail="Payslip not found")
    db.delete(slip)
    db.commit()
    return {"message": "Deleted"}
