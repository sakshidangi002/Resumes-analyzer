"""HR letter templates and generation with Jinja2."""
from datetime import date
from jinja2 import Template
from sqlalchemy.orm import Session
from app.models import LetterTemplate, LetterInstance, Employee, SalaryStructure, Payslip
from app.services.payroll_service import get_salary_structure_for_date


def render_letter(
    db: Session,
    template_code: str,
    employee_id: int,
    extra_context: dict | None = None,
) -> tuple[str, str]:
    """Render subject and body for a letter template with employee data."""
    t = db.query(LetterTemplate).filter(LetterTemplate.code == template_code).first()
    if not t:
        raise ValueError(f"Template not found: {template_code}")
    emp = db.query(Employee).filter(Employee.id == employee_id).first()
    if not emp:
        raise ValueError("Employee not found")
    ctx = {
        "employee": emp,
        "employee_name": emp.full_name,
        "employee_code": emp.employee_code,
        "date_of_joining": emp.date_of_joining,
        "designation": emp.designation.title if emp.designation else "",
        "department": emp.department.name if emp.department else "",
        "date": date.today(),
    }
    struct = get_salary_structure_for_date(db, employee_id, date.today())
    if struct:
        ctx["basic"] = struct.basic
        ctx["hra"] = struct.hra
        ctx["allowances"] = struct.allowances
        ctx["gross"] = struct.basic + struct.hra + struct.allowances
    else:
        ctx["basic"] = ctx["hra"] = ctx["allowances"] = ctx["gross"] = 0
    if extra_context:
        ctx.update(extra_context)
    subject = Template(t.subject_template or "").render(**ctx) if t.subject_template else ""
    body = Template(t.body_template).render(**ctx)
    return subject, body


def create_letter_instance(
    db: Session,
    employee_id: int,
    template_id: int,
    generated_by_user_id: int | None,
    subject: str,
    body: str,
    data_snapshot: str | None = None,
    sent_via_email: bool = False,
    email_log_id: int | None = None,
) -> LetterInstance:
    inst = LetterInstance(
        employee_id=employee_id,
        template_id=template_id,
        generated_by_user_id=generated_by_user_id,
        subject=subject,
        body=body,
        data_snapshot=data_snapshot,
        sent_via_email=sent_via_email,
        email_log_id=email_log_id,
    )
    db.add(inst)
    db.commit()
    db.refresh(inst)
    return inst
