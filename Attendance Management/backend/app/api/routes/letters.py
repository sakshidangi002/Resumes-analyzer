"""HR letter templates and generate letters."""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models import User, LetterTemplate, LetterInstance, Employee, LetterReply
from app.schemas.letter import (
    LetterTemplateCreate,
    LetterTemplateResponse,
    LetterInstanceResponse,
    LetterPreviewRequest,
    LetterPreviewResponse,
    LetterGenerateOverrides,
    LetterReplyResponse,
    LetterReplyCreate,
)
from app.api.deps import get_current_user, require_roles
from app.services.letter_service import render_letter, create_letter_instance
from app.services.letter_templates_defaults import ensure_default_letter_templates
from app.services.email_service import send_notification
from app.services.letter_pdf import html_to_pdf_bytes, safe_pdf_filename
from app.services.notification_service import notify_user_for_employee


def _resolve_targets(emp: Employee, email_target: str) -> list[str]:
    """Pick official/personal email(s) for a letter delivery, raise HTTP 400 if missing."""
    official = (emp.official_email or "").strip() if emp else ""
    personal = (emp.personal_email or "").strip() if emp else ""
    targets: list[str] = []
    if email_target == "personal":
        if not personal:
            raise HTTPException(
                status_code=400,
                detail="Employee has no personal email saved. Add Personal Email in Employees and try again.",
            )
        targets.append(personal)
    elif email_target == "both":
        if not official and not personal:
            raise HTTPException(
                status_code=400,
                detail="Employee has neither official nor personal email saved.",
            )
        if official:
            targets.append(official)
        if personal and personal not in targets:
            targets.append(personal)
    else:  # default: official
        if not official:
            raise HTTPException(
                status_code=400,
                detail="Employee has no official email saved.",
            )
        targets.append(official)
    return targets


def _build_letter_attachment(
    emp: Employee | None,
    template_code: str | None,
    subject: str | None,
    body_html: str,
) -> list[tuple[str, bytes, str]]:
    """Render the letter HTML to a PDF and return it as an SMTP attachment list."""
    pdf_bytes = html_to_pdf_bytes(body_html, subject)
    if not pdf_bytes:
        return []
    emp_code = emp.employee_code if emp else None
    emp_name = emp.full_name if emp else None
    filename = safe_pdf_filename(template_code or "Letter", emp_code, emp_name)
    return [(filename, pdf_bytes, "pdf")]

router = APIRouter()

def _ensure_user_employee_link(db: Session, current_user: User) -> None:
    """
    If the logged-in user is not linked to an Employee, try to auto-link by official email.
    This helps Employee/HR/Admin accounts so personal views (My Letters) work without manual linking.
    """
    if current_user.employee_id:
        return
    candidate_email = (current_user.official_email or "").strip()
    if not candidate_email:
        candidate_email = (current_user.username or "").strip()
    if "@" not in candidate_email:
        return
    emp = db.query(Employee).filter(Employee.official_email == candidate_email).first()
    if not emp:
        return
    current_user.employee_id = emp.id
    db.add(current_user)
    db.commit()


@router.get("/templates", response_model=list[LetterTemplateResponse])
def list_templates(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    ensure_default_letter_templates(db)
    return db.query(LetterTemplate).order_by(LetterTemplate.code).all()


@router.post("/templates", response_model=LetterTemplateResponse)
def create_template(
    data: LetterTemplateCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    if db.query(LetterTemplate).filter(LetterTemplate.code == data.code).first():
        raise HTTPException(status_code=400, detail="Template code exists")
    t = LetterTemplate(
        code=data.code,
        name=data.name,
        subject_template=data.subject_template,
        body_template=data.body_template,
        is_editable=data.is_editable,
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@router.get("/templates/{template_id}", response_model=LetterTemplateResponse)
def get_template(
    template_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    t = db.query(LetterTemplate).filter(LetterTemplate.id == template_id).first()
    if not t:
        raise HTTPException(status_code=404, detail="Template not found")
    return t


@router.patch("/templates/{template_id}", response_model=LetterTemplateResponse)
def update_template(
    template_id: int,
    body_template: str | None = None,
    subject_template: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    t = db.query(LetterTemplate).filter(LetterTemplate.id == template_id).first()
    if not t:
        raise HTTPException(status_code=404, detail="Template not found")
    if not t.is_editable:
        raise HTTPException(status_code=400, detail="Template not editable")
    if body_template is not None:
        t.body_template = body_template
    if subject_template is not None:
        t.subject_template = subject_template
    db.commit()
    db.refresh(t)
    return t


@router.post("/preview", response_model=LetterPreviewResponse)
def preview_letter(
    data: LetterPreviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Return auto-generated subject and body without saving a letter instance."""
    try:
        subject, body = render_letter(
            db,
            data.template_code,
            data.employee_id,
            extra_context=data.extra_context or {},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return LetterPreviewResponse(subject=subject, body=body)


@router.post("/generate")
def generate_letter(
    template_code: str,
    employee_id: int,
    send_email: bool = False,
    email_target: str = "official",
    overrides: LetterGenerateOverrides | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """
    Generate a letter instance.

    - By default, subject/body are auto-generated from the template.
    - If overrides.body (and optionally overrides.subject) are provided,
      those values are used instead so HR can edit content before saving.
    """
    if overrides and overrides.body is not None:
        subject = overrides.subject or ""
        body = overrides.body
    else:
        try:
            subject, body = render_letter(db, template_code, employee_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    t = db.query(LetterTemplate).filter(LetterTemplate.code == template_code).first()
    inst = create_letter_instance(
        db, employee_id, t.id, current_user.id, subject, body, sent_via_email=False,
    )
    tname = t.name if t else template_code
    notify_user_for_employee(
        db,
        employee_id,
        f"New letter: {tname}",
        (subject or "").strip()[:300] or None,
        kind="LETTER",
        link_path="/letters",
        with_push=True,
        push_tag=f"letter-{inst.id}",
    )
    if send_email:
        emp = db.query(Employee).filter(Employee.id == employee_id).first()
        if emp:
            targets = _resolve_targets(emp, email_target)
            attachments = _build_letter_attachment(emp, template_code, subject, body)
            for addr in targets:
                ok, err = send_notification(
                    db,
                    addr,
                    subject,
                    body,
                    template_code=template_code,
                    related_entity_type="LetterInstance",
                    related_entity_id=str(inst.id),
                    from_email=(overrides.from_email.strip() if overrides and overrides.from_email else None),
                    attachments=attachments or None,
                )
                if not ok:
                    raise HTTPException(status_code=500, detail=f"Failed to send email to {addr}. {err or ''}".strip())
            if targets:
                inst.sent_via_email = True
                db.commit()
    return {"letter_instance_id": inst.id, "subject": subject, "sent_email": send_email}


@router.post("/generate-bulk")
def generate_letters_bulk(
    template_code: str,
    employee_ids: list[int],
    send_email: bool = False,
    email_target: str = "official",
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    results = []
    for eid in employee_ids:
        try:
            subject, body = render_letter(db, template_code, eid)
            t = db.query(LetterTemplate).filter(LetterTemplate.code == template_code).first()
            inst = create_letter_instance(db, eid, t.id, current_user.id, subject, body, sent_via_email=False)
            tname = t.name if t else template_code
            notify_user_for_employee(
                db,
                eid,
                f"New letter: {tname}",
                (subject or "").strip()[:300] or None,
                kind="LETTER",
                link_path="/letters",
                with_push=True,
                push_tag=f"letter-{inst.id}",
            )
            if send_email:
                emp = db.query(Employee).filter(Employee.id == eid).first()
                if emp:
                    # Bulk send: best-effort target collection (don't 400 on missing email,
                    # just skip so the rest of the batch continues).
                    targets: list[str] = []
                    official = (emp.official_email or "").strip()
                    personal = (emp.personal_email or "").strip()
                    if email_target == "personal":
                        if personal:
                            targets.append(personal)
                    elif email_target == "both":
                        if official:
                            targets.append(official)
                        if personal and personal not in targets:
                            targets.append(personal)
                    else:
                        if official:
                            targets.append(official)
                    attachments = _build_letter_attachment(emp, template_code, subject, body) if targets else []
                    for addr in targets:
                        send_notification(
                            db, addr, subject, body,
                            template_code=template_code,
                            related_entity_type="LetterInstance",
                            related_entity_id=str(inst.id),
                            attachments=attachments or None,
                        )
                    if targets:
                        inst.sent_via_email = True
            results.append({"employee_id": eid, "letter_instance_id": inst.id})
        except Exception as e:
            results.append({"employee_id": eid, "error": str(e)})
    db.commit()
    return {"results": results}


@router.get("/instances", response_model=list[LetterInstanceResponse])
def list_instances(
    employee_id: int | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    q = (
        db.query(LetterInstance, Employee)
        .join(Employee, Employee.id == LetterInstance.employee_id)
    )
    role_names = [r.name for r in current_user.roles]
    _ensure_user_employee_link(db, current_user)
    if "Employee" in role_names and "Manager" not in role_names and "HR" not in role_names and "Admin" not in role_names:
        if current_user.employee_id:
            q = q.filter(LetterInstance.employee_id == current_user.employee_id)
        else:
            return []
    elif employee_id is not None:
        q = q.filter(LetterInstance.employee_id == employee_id)
    rows = q.order_by(LetterInstance.generated_at.asc()).all()
    result: list[LetterInstanceResponse] = []
    for inst, emp in rows:
        result.append(
            LetterInstanceResponse(
                id=inst.id,
                employee_id=inst.employee_id,
                template_id=inst.template_id,
                generated_at=inst.generated_at,
                subject=inst.subject,
                sent_via_email=inst.sent_via_email,
                employee_code=emp.employee_code,
                employee_name=emp.full_name,
                employee_official_email=emp.official_email,
                employee_personal_email=emp.personal_email,
            )
        )
    return result


@router.get("/instances/{instance_id}/body", response_class=HTMLResponse)
def get_letter_body(
    instance_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    inst = db.query(LetterInstance).filter(LetterInstance.id == instance_id).first()
    if not inst:
        raise HTTPException(status_code=404, detail="Not found")
    _ensure_user_employee_link(db, current_user)
    role_names = [r.name for r in current_user.roles]
    is_mgmt = "Admin" in role_names or "HR" in role_names
    if not is_mgmt and current_user.employee_id != inst.employee_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return inst.body or ""


@router.get("/instances/{instance_id}/replies", response_model=list[LetterReplyResponse])
def list_replies(
    instance_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    inst = db.query(LetterInstance).filter(LetterInstance.id == instance_id).first()
    if not inst:
        raise HTTPException(status_code=404, detail="Letter not found")
    _ensure_user_employee_link(db, current_user)
    role_names = [r.name for r in current_user.roles]
    is_mgmt = "Admin" in role_names or "HR" in role_names
    if not is_mgmt and current_user.employee_id != inst.employee_id:
        raise HTTPException(status_code=403, detail="Access denied")
    replies = (
        db.query(LetterReply)
        .filter(LetterReply.letter_instance_id == instance_id)
        .order_by(LetterReply.created_at.asc())
        .all()
    )
    return [LetterReplyResponse.model_validate(r) for r in replies]


@router.post("/instances/{instance_id}/replies", response_model=LetterReplyResponse)
def create_reply(
    instance_id: int,
    data: LetterReplyCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    inst = db.query(LetterInstance).filter(LetterInstance.id == instance_id).first()
    if not inst:
        raise HTTPException(status_code=404, detail="Letter not found")
    _ensure_user_employee_link(db, current_user)
    role_names = [r.name for r in current_user.roles]
    is_mgmt = "Admin" in role_names or "HR" in role_names
    if not is_mgmt and current_user.employee_id != inst.employee_id:
        raise HTTPException(status_code=403, detail="Access denied")
    if not data.message or not data.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")
    reply = LetterReply(
        letter_instance_id=instance_id,
        author_employee_id=current_user.employee_id,
        author_user_id=current_user.id,
        message=data.message.strip(),
    )
    db.add(reply)
    db.commit()
    db.refresh(reply)
    return LetterReplyResponse.model_validate(reply)


@router.post("/instances/{instance_id}/email")
def email_letter_instance(
    instance_id: int,
    email_target: str = "official",
    from_email: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """
    Send (or resend) a previously generated letter as an email with a PDF attachment.

    `email_target` ∈ {"official", "personal", "both"}.
    Returns the list of addresses the letter was sent to.
    """
    inst = db.query(LetterInstance).filter(LetterInstance.id == instance_id).first()
    if not inst:
        raise HTTPException(status_code=404, detail="Letter not found")
    emp = db.query(Employee).filter(Employee.id == inst.employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found for this letter")
    template = db.query(LetterTemplate).filter(LetterTemplate.id == inst.template_id).first()
    template_code = template.code if template else None

    targets = _resolve_targets(emp, email_target)
    attachments = _build_letter_attachment(emp, template_code, inst.subject, inst.body or "")

    sent_to: list[str] = []
    last_err: str | None = None
    for addr in targets:
        ok, err = send_notification(
            db,
            addr,
            inst.subject or "",
            inst.body or "",
            template_code=template_code,
            related_entity_type="LetterInstance",
            related_entity_id=str(inst.id),
            from_email=(from_email.strip() if from_email else None),
            attachments=attachments or None,
        )
        if ok:
            sent_to.append(addr)
        else:
            last_err = err

    if not sent_to:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send email. {last_err or ''}".strip(),
        )

    inst.sent_via_email = True
    db.commit()
    return {
        "letter_instance_id": inst.id,
        "sent_to": sent_to,
        "with_pdf": bool(attachments),
    }


@router.delete("/instances/{instance_id}")
def delete_letter_instance(
    instance_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    inst = db.query(LetterInstance).filter(LetterInstance.id == instance_id).first()
    if not inst:
        raise HTTPException(status_code=404, detail="Letter not found")
    # Delete replies first, then the instance
    db.query(LetterReply).filter(LetterReply.letter_instance_id == instance_id).delete()
    db.delete(inst)
    db.commit()
    return {"message": "Letter deleted"}
