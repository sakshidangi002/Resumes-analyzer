"""Company policies (Feature 4).

Multiple named policies, each versioned. Publishing a new version appends a row
(never overwrites), so previous policies stay viewable. A version carries typed
text and/or an optional attached file. Viewable by everyone; only Admin/HR can
publish or delete.
"""
import uuid
from pathlib import Path
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import User, CompanyPolicy
from app.schemas.policy import PolicyVersionResponse, PolicyGroup, PolicyHistory, PolicyUpdate
from app.api.deps import require_roles

router = APIRouter()

# Stored under backend/data/policy_attachments (mirrors data/face_uploads).
BASE_DIR = Path(__file__).resolve().parents[3]
POLICY_DIR = BASE_DIR / "data" / "policy_attachments"


@router.get("", response_model=list[PolicyGroup])
def list_policies(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    """One entry per policy name = its current (latest) version + version count."""
    rows = (
        db.query(CompanyPolicy)
        .order_by(CompanyPolicy.name.asc(), CompanyPolicy.version.desc())
        .all()
    )
    counts = dict(
        db.query(CompanyPolicy.name, func.count(CompanyPolicy.id))
        .group_by(CompanyPolicy.name)
        .all()
    )
    groups: list[PolicyGroup] = []
    seen: set[str] = set()
    for r in rows:  # already sorted so the first row per name is the latest version
        if r.name in seen:
            continue
        seen.add(r.name)
        groups.append(
            PolicyGroup(
                name=r.name,
                category=r.category,
                current=PolicyVersionResponse.model_validate(r),
                versions_count=int(counts.get(r.name, 1)),
            )
        )
    return groups


@router.get("/history", response_model=PolicyHistory)
def policy_history(
    name: str = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    """All versions of a named policy, newest first."""
    versions = (
        db.query(CompanyPolicy)
        .filter(CompanyPolicy.name == name)
        .order_by(CompanyPolicy.version.desc())
        .all()
    )
    if not versions:
        raise HTTPException(status_code=404, detail="Policy not found")
    return PolicyHistory(name=name, versions=versions)


@router.post("", response_model=PolicyVersionResponse)
def create_policy_version(
    name: str = Form(...),
    effective_date: date = Form(...),
    title: str | None = Form(None),
    category: str | None = Form(None),
    content: str | None = Form(None),
    file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Publish a new version of a policy (creates the policy if the name is new)."""
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Policy name is required")
    has_text = bool(content and content.strip())
    has_file = file is not None and bool(file.filename)
    if not has_text and not has_file:
        raise HTTPException(status_code=400, detail="Provide policy text and/or an attachment")

    # Next version number for this policy name.
    last = (
        db.query(func.max(CompanyPolicy.version))
        .filter(CompanyPolicy.name == name)
        .scalar()
    )
    next_version = int(last or 0) + 1

    attachment_path = None
    attachment_name = None
    if has_file:
        POLICY_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename).suffix.lower()
        stored = POLICY_DIR / f"{uuid.uuid4().hex}{suffix}"
        stored.write_bytes(file.file.read())
        attachment_path = str(stored)
        attachment_name = file.filename

    policy = CompanyPolicy(
        name=name,
        title=(title or None),
        category=(category or None),
        content=(content or None),
        effective_date=effective_date,
        version=next_version,
        attachment_path=attachment_path,
        attachment_name=attachment_name,
        published_by_user_id=current_user.id,
        published_by_name=current_user.username,
    )
    db.add(policy)
    db.commit()
    db.refresh(policy)
    return policy


@router.patch("/{policy_id}", response_model=PolicyVersionResponse)
def update_policy_version(
    policy_id: int,
    data: PolicyUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Edit an existing policy version in place (title/category/content/date)."""
    policy = db.query(CompanyPolicy).filter(CompanyPolicy.id == policy_id).first()
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    patch = data.model_dump(exclude_unset=True)
    for field, value in patch.items():
        setattr(policy, field, value)
    db.commit()
    db.refresh(policy)
    return policy


@router.get("/{policy_id}/attachment")
def download_attachment(
    policy_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR", "Manager", "Employee"])),
):
    policy = db.query(CompanyPolicy).filter(CompanyPolicy.id == policy_id).first()
    if not policy or not policy.attachment_path:
        raise HTTPException(status_code=404, detail="Attachment not found")
    path = Path(policy.attachment_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Attachment file missing on server")
    return FileResponse(str(path), filename=policy.attachment_name or path.name)


@router.delete("/{policy_id}")
def delete_policy_version(
    policy_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    policy = db.query(CompanyPolicy).filter(CompanyPolicy.id == policy_id).first()
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    # Best-effort remove the stored file.
    if policy.attachment_path:
        try:
            Path(policy.attachment_path).unlink(missing_ok=True)
        except OSError:
            pass
    db.delete(policy)
    db.commit()
    return {"message": "Deleted"}
