"""Interview Questions.

HR/Admin upload interview-question PDFs, one per hiring position, and manage them
(list, filter, view/download, edit metadata, delete). PDFs are stored on disk
under backend/data/interview_questions and streamed back through the API so the
same auth guard that protects every other endpoint also protects the files.
"""
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models import User, InterviewQuestion
from app.schemas.interview_question import (
    InterviewQuestionResponse,
    InterviewQuestionUpdate,
)
from app.api.deps import require_roles

router = APIRouter()

# Stored under backend/data/interview_questions (mirrors data/policy_attachments).
BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = BASE_DIR / "data" / "interview_questions"

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# Single source of truth for the hiring-position dropdown. Extend this list to
# add a new position everywhere (upload form + filter) at once.
HIRING_POSITIONS = [
    ".NET Developer",
    "React Developer",
    "Python Developer",
    "Java Developer",
    "Node.js Developer",
    "Angular Developer",
    "Full Stack Developer",
    "QA Engineer",
    "DevOps Engineer",
    "Data Analyst",
    "UI/UX Designer",
    "HR Executive",
    "Other",
]


def _duplicate_exists(db: Session, position: str, title: str, exclude_id: int | None = None) -> bool:
    """A title must be unique within a hiring position (case-insensitive)."""
    q = db.query(InterviewQuestion).filter(
        func.lower(InterviewQuestion.position) == position.lower(),
        func.lower(InterviewQuestion.title) == title.lower(),
    )
    if exclude_id is not None:
        q = q.filter(InterviewQuestion.id != exclude_id)
    return db.query(q.exists()).scalar()


@router.get("/positions", response_model=list[str])
def list_positions(
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """The hiring positions available for the dropdown/filter."""
    return HIRING_POSITIONS


@router.get("", response_model=list[InterviewQuestionResponse])
def list_interview_questions(
    position: str | None = Query(None, description="Filter by exact hiring position"),
    search: str | None = Query(None, description="Search in title or position"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """All uploaded question sets, newest first, optionally filtered/searched."""
    q = db.query(InterviewQuestion)
    if position:
        q = q.filter(func.lower(InterviewQuestion.position) == position.lower())
    if search and search.strip():
        like = f"%{search.strip().lower()}%"
        q = q.filter(
            or_(
                func.lower(InterviewQuestion.title).like(like),
                func.lower(InterviewQuestion.position).like(like),
            )
        )
    return q.order_by(InterviewQuestion.created_at.desc(), InterviewQuestion.id.desc()).all()


@router.post("", response_model=InterviewQuestionResponse)
def upload_interview_question(
    position: str = Form(...),
    title: str = Form(...),
    description: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Upload a new interview-question PDF for a hiring position."""
    position = (position or "").strip()
    title = (title or "").strip()
    if not position:
        raise HTTPException(status_code=400, detail="Hiring position is required")
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="A PDF file is required")

    suffix = Path(file.filename).suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 20 MB")

    if _duplicate_exists(db, position, title):
        raise HTTPException(
            status_code=400,
            detail=f'A question set titled "{title}" already exists for {position}.',
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    stored.write_bytes(data)

    record = InterviewQuestion(
        position=position,
        title=title,
        description=(description.strip() if description and description.strip() else None),
        pdf_path=str(stored),
        pdf_name=file.filename,
        uploaded_by_user_id=current_user.id,
        uploaded_by_name=current_user.username,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.patch("/{item_id}", response_model=InterviewQuestionResponse)
def update_interview_question(
    item_id: int,
    data: InterviewQuestionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Edit an uploaded question set's metadata (position/title/description)."""
    record = db.query(InterviewQuestion).filter(InterviewQuestion.id == item_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Interview question not found")

    patch = data.model_dump(exclude_unset=True)
    new_position = (patch.get("position") or record.position).strip() if "position" in patch else record.position
    new_title = (patch.get("title") or record.title).strip() if "title" in patch else record.title
    if "position" in patch and not new_position:
        raise HTTPException(status_code=400, detail="Hiring position cannot be empty")
    if "title" in patch and not new_title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    if _duplicate_exists(db, new_position, new_title, exclude_id=record.id):
        raise HTTPException(
            status_code=400,
            detail=f'A question set titled "{new_title}" already exists for {new_position}.',
        )

    record.position = new_position
    record.title = new_title
    if "description" in patch:
        desc = patch.get("description")
        record.description = desc.strip() if desc and desc.strip() else None
    db.commit()
    db.refresh(record)
    return record


@router.get("/{item_id}/file")
def get_interview_question_file(
    item_id: int,
    download: bool = Query(False, description="Force download instead of inline view"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """Stream the stored PDF, inline for viewing or as an attachment to download."""
    record = db.query(InterviewQuestion).filter(InterviewQuestion.id == item_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Interview question not found")
    path = Path(record.pdf_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF file missing on server")
    disposition = "attachment" if download else "inline"
    return FileResponse(
        str(path),
        media_type="application/pdf",
        filename=record.pdf_name,
        headers={"Content-Disposition": f'{disposition}; filename="{record.pdf_name}"'},
    )


@router.delete("/{item_id}")
def delete_interview_question(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    record = db.query(InterviewQuestion).filter(InterviewQuestion.id == item_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Interview question not found")
    # Best-effort remove the stored file.
    if record.pdf_path:
        try:
            Path(record.pdf_path).unlink(missing_ok=True)
        except OSError:
            pass
    db.delete(record)
    db.commit()
    return {"message": "Deleted"}
