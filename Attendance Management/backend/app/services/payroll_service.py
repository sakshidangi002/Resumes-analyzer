"""Payroll: calendar-month salary calculation, LOP, payslips."""
from datetime import date, timedelta
from decimal import Decimal
import json
from sqlalchemy.orm import Session
from app.models import (
    SalaryStructure,
    PayrollPeriod,
    Payslip,
    AttendanceRecord,
    Employee,
    LeaveRequest,
    LeaveType,
    CompanyConfig,
    Holiday,
)
from app.services.leave_service import get_current_financial_year


def _get_company_config(db: Session) -> CompanyConfig | None:
    return db.query(CompanyConfig).first()


def _is_weekly_off(d: date, weekly_off_days: str | None) -> bool:
    days_cfg = weekly_off_days or "SAT,SUN"
    day_num = d.weekday()  # Monday=0
    day_names = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    day_str = day_names[day_num]
    return day_str in days_cfg.upper().replace(" ", "").split(",")


def _is_holiday(db: Session, d: date) -> bool:
    return db.query(Holiday).filter(Holiday.date == d).first() is not None


def get_salary_structure_for_date(db: Session, employee_id: int, as_on: date) -> SalaryStructure | None:
    return db.query(SalaryStructure).filter(
        SalaryStructure.employee_id == employee_id,
        SalaryStructure.effective_from <= as_on,
        (SalaryStructure.effective_to == None) | (SalaryStructure.effective_to >= as_on),
    ).order_by(SalaryStructure.effective_from.desc()).first()


def run_payroll_for_period(
    db: Session,
    month: int,
    year: int,
) -> list[Payslip]:
    period = db.query(PayrollPeriod).filter(
        PayrollPeriod.month == month,
        PayrollPeriod.year == year,
    ).first()
    if not period:
        period = PayrollPeriod(month=month, year=year, status="OPEN")
        db.add(period)
        db.flush()
    if period.status == "LOCKED":
        return db.query(Payslip).filter(Payslip.payroll_period_id == period.id).all()
    employees = db.query(Employee).filter(Employee.employment_status == "Active").all()
    # Existing payslips for this period.
    #
    # Note: historically, duplicates could be created for the same (employee_id, payroll_period_id).
    # When that happens, we keep the newest and delete the extras so the UI doesn't show duplicates
    # and re-runs remain idempotent.
    slips_for_period = (
        db.query(Payslip)
        .filter(Payslip.payroll_period_id == period.id)
        .order_by(Payslip.employee_id.asc(), Payslip.generated_at.desc(), Payslip.id.desc())
        .all()
    )
    existing_slips: dict[int, Payslip] = {}
    duplicate_ids: list[int] = []
    for s in slips_for_period:
        if s.employee_id in existing_slips:
            duplicate_ids.append(s.id)
        else:
            existing_slips[s.employee_id] = s
    if duplicate_ids:
        (
            db.query(Payslip)
            .filter(Payslip.id.in_(duplicate_ids))
            .delete(synchronize_session=False)
        )
        db.flush()
    payslips = []
    config = _get_company_config(db)
    weekly_off_days = config.weekly_off_days if config else None
    for emp in employees:
        struct = get_salary_structure_for_date(db, emp.id, date(year, month, 1))
        if not struct:
            continue
        gross = struct.basic + struct.hra + struct.allowances
        # Count paid days and LOP over the relevant range.
        # For past months: full calendar month.
        # For the current month: only from the 1st up to today (no future days).
        month_start = date(year, month, 1)
        if month == 12:
            month_end = date(year, 12, 31)
        else:
            month_end = date(year, month + 1, 1) - timedelta(days=1)

        today = date.today()
        if year == today.year and month == today.month:
            # Exclude current day; calculate only up to yesterday
            end = min(month_end, today - timedelta(days=1))
        else:
            end = month_end
        start = month_start

        # Range of days we are actually evaluating (1st of month up to today or month end)
        actual_days = (end - start).days + 1
        days_in_range = Decimal(str(actual_days))

        # Approved leave requests (paid and unpaid) for the period.
        # We need this to determine if an ABSENT or ON_LEAVE status is payable.
        leave_unpaid_fraction_by_date: dict[date, Decimal] = {}
        leave_is_approved_by_date: dict[date, bool] = {}
        
        all_leave_reqs = (
            db.query(LeaveRequest, LeaveType)
            .join(LeaveType, LeaveType.id == LeaveRequest.leave_type_id)
            .filter(
                LeaveRequest.employee_id == emp.id,
                LeaveRequest.status == "APPROVED",
                LeaveRequest.start_date <= end,
                LeaveRequest.end_date >= start,
            )
            .all()
        )
        for req, lt in all_leave_reqs:
            print(f"DEBUG: Emp {emp.id} Leave {req.start_date} to {req.end_date}, Type: {lt.name} (is_paid={lt.is_paid})")
            d = max(req.start_date, start)
            last = min(req.end_date, end)
            while d <= last:
                leave_is_approved_by_date[d] = True
                if not lt.is_paid:
                    frac = Decimal("0.5") if (req.is_half_day and req.start_date == req.end_date) else Decimal("1")
                    leave_unpaid_fraction_by_date[d] = max(leave_unpaid_fraction_by_date.get(d, Decimal("0")), frac)
                else:
                    # Explicitly mark as 0 if paid, to avoid defaulting to 1 on ABSENT
                    if d not in leave_unpaid_fraction_by_date:
                        leave_unpaid_fraction_by_date[d] = Decimal("0")
                d = d + timedelta(days=1)

        # Count total weekly offs and holidays in the month (always payable)
        wo_and_holiday_days = Decimal("0")
        current = start
        for _ in range(actual_days):
            if _is_weekly_off(current, weekly_off_days) or _is_holiday(db, current):
                wo_and_holiday_days += Decimal("1")
            current = current + timedelta(days=1)
        records = db.query(AttendanceRecord).filter(
            AttendanceRecord.employee_id == emp.id,
            AttendanceRecord.date >= start,
            AttendanceRecord.date <= end,
        ).all()

        # If there is no meaningful attendance recorded at all for this employee
        # in the month (no Time In / Time Out / status), skip salary calculation.
        if not records:
            continue

        # --- LOP Calculation Logic ---
        # 1. Initialize full map for the range
        day_unpaid_frac = {}
        day_is_approved_leave = {}
        
        curr = start
        while curr <= end:
            # Default: Working days = 1 (Unpaid), Non-working (WO/Holiday) = 0 (Paid)
            if _is_weekly_off(curr, weekly_off_days) or _is_holiday(db, curr):
                day_unpaid_frac[curr] = Decimal("0")
            else:
                day_unpaid_frac[curr] = Decimal("1")
            curr += timedelta(days=1)

        # 2. Update with Attendance Records (Punches)
        expected_hours = Decimal(str(emp.expected_working_hours if getattr(emp, 'expected_working_hours', None) else 9.0))
        punch_missed_hours = {}
        for r in records:
            if r.status in ("PRESENT", "HALF_DAY", "SHORT"):
                if r.total_work_hours is not None:
                    worked_hours = Decimal(str(r.total_work_hours))
                    missed = expected_hours - worked_hours
                    punch_missed_hours[r.date] = missed
                    
                    if missed <= Decimal("0.25"): # 15 min grace
                        day_unpaid_frac[r.date] = Decimal("0")
                    elif missed <= Decimal("2.0"): # Short Leave (up to 2h)
                        day_unpaid_frac[r.date] = Decimal("0") # Evaluated by SL buffer
                    elif missed <= Decimal("4.5"): # Half Day (up to 4.5h)
                        day_unpaid_frac[r.date] = Decimal("0.5")
                    else: # Full Day
                        day_unpaid_frac[r.date] = Decimal("1.0")
                else:
                    if r.status == "PRESENT":
                        day_unpaid_frac[r.date] = Decimal("0")
                    elif r.status == "SHORT":
                        punch_missed_hours[r.date] = Decimal("2.0")
                        day_unpaid_frac[r.date] = Decimal("0")
                    else:
                        punch_missed_hours[r.date] = Decimal("4.5")
                        day_unpaid_frac[r.date] = Decimal("0.5")
            elif r.status in ("PAID_LEAVE", "HOLIDAY", "WEEKLY_OFF"):
                day_unpaid_frac[r.date] = Decimal("0")
            elif r.status in ("ABSENT", "ON_LEAVE"):
                day_unpaid_frac[r.date] = Decimal("1")

        # 3. Update with Approved Leave Requests (Overrides Punches)
        day_leave_type_code = {} # Track type for SL logic
        for req, lt in all_leave_reqs:
            d = max(req.start_date, start)
            last = min(req.end_date, end)
            while d <= last:
                if req.status == "APPROVED":
                    day_leave_type_code[d] = lt.code
                d = d + timedelta(days=1)

        for d, is_appr in leave_is_approved_by_date.items():
            if is_appr:
                frac = leave_unpaid_fraction_by_date.get(d, Decimal("0"))
                day_unpaid_frac[d] = frac
                day_is_approved_leave[d] = True

        # 4. Short Leave / Half Day Buffer Logic (2 SLs per month free)
        sl_buffer = 2 
        short_leaves_used = 0
        sorted_dates = sorted(day_unpaid_frac.keys())
        for d in sorted_dates:
            missed = punch_missed_hours.get(d, Decimal("0"))
            # SL is any absence up to 2 hours; HD is between 2 and 4.5 hours
            is_sl = day_leave_type_code.get(d) == "SL" or (Decimal("0.25") < missed <= Decimal("2.0"))
            is_hd = (Decimal("2.0") < missed <= Decimal("4.5")) or (d in day_unpaid_frac and day_unpaid_frac[d] == Decimal("0.5"))
            
            # Use buffer: 1 HD = 2 SLs, or 1 SL = 1 SL
            if is_hd and sl_buffer >= 2:
                short_leaves_used += 2
                sl_buffer -= 2
                day_unpaid_frac[d] = Decimal("0") # Paid via buffer
            elif is_sl and sl_buffer >= 1:
                short_leaves_used += 1
                sl_buffer -= 1
                day_unpaid_frac[d] = Decimal("0") # Paid via buffer
            else:
                # ALL OTHER ABSENCES (including SL/HD after buffer) are now fully proportional
                if missed > Decimal("0.25"):
                    day_unpaid_frac[d] = missed / expected_hours
                else:
                    # Grace period or worked full shift
                    # Only keep existing value if it was set to 1.0 by ABSENT status (no punches)
                    if d in day_unpaid_frac and day_unpaid_frac[d] == Decimal("1.0"):
                        pass 
                    else:
                        day_unpaid_frac[d] = Decimal("0")

        # 5. Apply Sandwich Leave Policy
        # Rule: If absent (unpaid) on both sides of a non-working block, the block becomes LOP.
        curr = start
        while curr <= end:
            if _is_weekly_off(curr, weekly_off_days) or _is_holiday(db, curr):
                block = []
                temp = curr
                while temp <= end and (_is_weekly_off(temp, weekly_off_days) or _is_holiday(db, temp)):
                    block.append(temp)
                    temp += timedelta(days=1)
                
                # Find boundary working days
                prev_wd = curr - timedelta(days=1)
                while prev_wd >= start and (_is_weekly_off(prev_wd, weekly_off_days) or _is_holiday(db, prev_wd)):
                    prev_wd -= timedelta(days=1)
                
                next_wd = temp
                while next_wd <= end and (_is_weekly_off(next_wd, weekly_off_days) or _is_holiday(db, next_wd)):
                    next_wd += timedelta(days=1)

                # Sandwich if both boundaries are within month and are ABSENT (Full LOP)
                if prev_wd in day_unpaid_frac and next_wd in day_unpaid_frac:
                    if day_unpaid_frac[prev_wd] == Decimal("1.0") and day_unpaid_frac[next_wd] == Decimal("1.0"):
                        # Only sandwich if they don't have an approved PAID leave on those days
                        for d in block:
                            day_unpaid_frac[d] = Decimal("1")
                
                curr = temp
            else:
                curr += timedelta(days=1)

        unpaid_attendance_days = sum(day_unpaid_frac.values(), Decimal("0"))
        
        if end == month_end:
            basis = Decimal("30")
        else:
            basis = Decimal(str(actual_days))
            
        lop_days = min(basis, unpaid_attendance_days)
        paid_days = basis - lop_days

        # Final check: Ensure we never count a Paid Leave day as LOP
        for d, is_appr in leave_is_approved_by_date.items():
            if is_appr and leave_unpaid_fraction_by_date.get(d, Decimal("0")) == 0:
                # If it's a paid leave, it MUST be 0 LOP, even if sandwich tried to change it
                day_unpaid_frac[d] = Decimal("0")

        # Salary calculation strictly on 30-day basis
        per_day = gross / Decimal("30")
        total_earnings = per_day * paid_days
        total_deductions = (struct.deductions / Decimal("30")) * paid_days
        net = total_earnings - total_deductions
        per_hour = (gross / Decimal("30")) / expected_hours
        breakdown = json.dumps({
            "basic": float(struct.basic),
            "hra": float(struct.hra),
            "allowances": float(struct.allowances),
            "deductions": float(struct.deductions),
            "paid_days": float(paid_days),
            "lop_days": float(lop_days),
            "lop_dates": [d.isoformat() for d, v in day_unpaid_frac.items() if v > 0],
            "short_leaves_used": int(short_leaves_used),
            "per_hour_salary": float(round(per_hour, 2)),
            "expected_hours": float(expected_hours)
        })
        slip = existing_slips.get(emp.id)
        if slip:
            slip.gross_salary = gross
            slip.total_earnings = total_earnings
            slip.total_deductions = total_deductions
            slip.net_salary = net
            slip.paid_days = paid_days
            slip.lop_days = lop_days
            slip.component_breakdown = breakdown
        else:
            slip = Payslip(
                employee_id=emp.id,
                payroll_period_id=period.id,
                gross_salary=gross,
                total_earnings=total_earnings,
                total_deductions=total_deductions,
                net_salary=net,
                paid_days=paid_days,
                lop_days=lop_days,
                component_breakdown=breakdown,
            )
            db.add(slip)
        payslips.append(slip)
    period.status = "PROCESSED"
    from app.core.datetime_utils import get_ist_now
    period.is_processed = True
    period.processed_at = get_ist_now()
    db.commit()
    for s in payslips:
        db.refresh(s)
    return payslips
