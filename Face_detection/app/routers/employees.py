from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from app.db import UPLOAD_DIR, get_db, embedding_to_blob, db_cursor
from app.models.employee import Employee
from app.services.embedding_cache import invalidate_embedding_cache
from app.services.employee_service import (
    process_face_uploads,
    save_employee_photo,
    serialize_employee_row,
)

router = APIRouter(prefix="/api/employees", tags=["employees"])


@router.get("")
def list_employees(db: Session = Depends(get_db)) -> dict:
    employees = db.query(Employee).order_by(Employee.created_at.desc(), Employee.id.desc()).all()
    return {"employees": [serialize_employee_row(emp) for emp in employees]}


@router.post("/register")
async def register_employee(
    employee_id: int = Form(...),
    name: str = Form(...),
    department: str = Form(None),
    images: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Employee name is required")

    if not 1 <= len(images) <= 5:
        raise HTTPException(status_code=400, detail="Upload between 1 and 5 face images")

    prepared_images = await process_face_uploads(images)
    embeddings = [item["embedding"] for item in prepared_images]
    averaged_embedding = np.mean(np.stack(embeddings), axis=0).astype(np.float32)

    with db_cursor() as (_, cur):
        emp = cur.execute("SELECT id FROM employees WHERE id = ?", (employee_id,)).fetchone()
        if not emp:
            raise HTTPException(status_code=404, detail=f"Employee with ID {employee_id} not found in HRMS database.")

        photo_path = save_employee_photo(employee_id, prepared_images[0]["bytes"], prepared_images[0]["filename"])
        
        cur.execute(
            """
            UPDATE employees
            SET embedding = ?, photo_path = ?, sample_count = ?
            WHERE id = ?
            """,
            (embedding_to_blob(averaged_embedding), photo_path, len(prepared_images), employee_id)
        )
    
    invalidate_embedding_cache()

    return JSONResponse(
        {
            "message": "Employee registered successfully",
            "employee": {
                "id": employee_id,
                "name": clean_name,
                "photo_path": photo_path,
                "sample_count": len(prepared_images)
            },
        }
    )


@router.put("/{employee_id}")
def update_employee(employee_id: int, name: str = Form(...), department: str = Form(None), db: Session = Depends(get_db)) -> dict:
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Employee name is required")

    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    employee.name = clean_name
    if department is not None:
        employee.department = department

    from app.models.event import AttendanceEvent
    db.commit()
    db.refresh(employee)

    invalidate_embedding_cache()
    return {"message": "Employee updated", "employee": serialize_employee_row(employee)}


@router.get("/{employee_id}/photo")
def get_employee_photo(employee_id: int, db: Session = Depends(get_db)) -> FileResponse:
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee or not employee.photo_path:
        raise HTTPException(status_code=404, detail="Photo not found")
        
    image_path = Path(employee.photo_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Sample image file missing")

    return FileResponse(image_path)


@router.delete("/{employee_id}")
def delete_employee(employee_id: int, db: Session = Depends(get_db)) -> dict:
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    db.delete(employee)
    db.commit()

    employee_dir = UPLOAD_DIR / str(employee_id)
    if employee_dir.exists():
        shutil.rmtree(employee_dir)

    invalidate_embedding_cache()
    return {"message": "Employee deleted", "employee_id": employee_id}

