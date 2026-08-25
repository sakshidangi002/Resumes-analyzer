"""Enrol an employee's face from a folder of photos.

Why a script and not just the UI
--------------------------------
The UI route (POST /api/employees/{id}/face) does the same thing, but it always
uses ``replace=True`` and reports only a count. When an employee is already
enrolled -- possibly under a superseded recognition model -- you want to see
what each photo scores BEFORE anything is written, because the write supersedes
their existing gallery and there is no undo.

So this runs as a DRY RUN by default. It reports, per photo, whether the face
passes the enrolment gate and what it would contribute, and writes nothing until
you pass --commit.

The pose warning
----------------
The enrolment gate deliberately does not restrict pose (FACE_ENROLL_MAX_YAW_DEG
defaults to 180) so the gallery can hold the angles a ceiling camera sees. But
an IN/OUT camera applies ``max_yaw_deg=40`` to the QUERY face before matching,
so a gallery vector captured at 60 degrees can never be hit by an attendance
camera: the query that would match it is discarded first. Such a photo is not
merely useless, it costs the employee score -- ``match.enrollment_bias_penalty``
grows with gallery diversity and is subtracted from every comparison.

Measured on this deployment's gallery, pruning the >40 degree photos was worth
+0.008 -- small, but the vectors are pure cost. This script flags them so you
can decide rather than discover it later.

Usage
-----
    python scripts/enrol_faces.py --employee 6 --dir enrol_drop/saloni
    python scripts/enrol_faces.py --employee 6 --dir enrol_drop/saloni --commit --replace

    --replace  supersede the employee's existing vectors (soft-deleted, audit
               trail preserved). Correct when re-enrolling after a model change.
               WITHOUT it, the new photos are ADDED to the existing gallery.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HRMS_BACKEND = REPO_ROOT / "Attendance Management" / "backend"
for _p in (str(HRMS_BACKEND), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# The bar an IN/OUT camera applies to a QUERY face. See module docstring.
RUNTIME_MAX_YAW = 40.0
RUNTIME_MAX_PITCH = 35.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--employee", type=int, required=True, help="employee id")
    ap.add_argument("--dir", required=True, help="folder of photos")
    ap.add_argument("--commit", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--replace", action="store_true",
                    help="supersede existing vectors instead of adding to them")
    args = ap.parse_args()

    import numpy as np
    from PIL import Image

    from app.db.session import SessionLocal
    from app.models.employee import Employee
    from app.services.employee_face_service import (
        ENROLL_LIMITS,
        _QUALITY_CHECK,
        enroll_embeddings,
        gallery_summary,
        save_employee_photo,
    )
    from app.services.face_quality import assess
    from app.services.face_service import EMBEDDING_MODEL_VERSION, extract_face_embeddings

    folder = Path(args.dir)
    if not folder.is_absolute():
        folder = REPO_ROOT / folder
    if not folder.is_dir():
        print(f"[error] not a folder: {folder}")
        return 2

    photos = sorted(p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not photos:
        print(f"[error] no images in {folder}")
        return 2

    with SessionLocal() as db:
        emp = db.query(Employee).filter(Employee.id == args.employee).first()
        if emp is None:
            print(f"[error] employee {args.employee} not found")
            return 2
        emp_name = emp.full_name
        emp_status = emp.employment_status

    print(f"Employee : {args.employee}  {emp_name}  ({emp_status})")
    print(f"Model    : {EMBEDDING_MODEL_VERSION}")
    print(f"Folder   : {folder}   ({len(photos)} image(s))")
    print(f"Mode     : {'COMMIT' if args.commit else 'DRY RUN — nothing will be written'}"
          f"{'  [replace]' if args.replace else '  [add to existing]'}")
    print(f"Gate     : face>={ENROLL_LIMITS.min_face_px:.0f}px det>={ENROLL_LIMITS.min_det_score:.2f} "
          f"blur>={ENROLL_LIMITS.min_blur_var:.0f} (quality check {'on' if _QUALITY_CHECK else 'OFF'})")
    print()

    before = gallery_summary(args.employee)
    print(f"Gallery before: {before['total']} active vector(s), "
          f"{before['stale_model_version']} on a superseded model")
    print()

    usable: list[dict] = []
    for photo in photos:
        try:
            pil = Image.open(photo).convert("RGB")
        except Exception as exc:
            print(f"  SKIP  {photo.name}: unreadable ({exc})")
            continue

        faces = extract_face_embeddings(pil)
        if not faces:
            print(f"  SKIP  {photo.name}: no detectable face")
            continue
        if len(faces) > 1:
            print(f"  SKIP  {photo.name}: {len(faces)} faces — use one face per image")
            continue

        face = faces[0]
        q = assess(face, np.asarray(pil), limits=ENROLL_LIMITS)
        if _QUALITY_CHECK and not q.ok:
            print(f"  SKIP  {photo.name}: {q.detail}")
            continue

        flag = ""
        if abs(q.yaw) > RUNTIME_MAX_YAW or abs(q.pitch) > RUNTIME_MAX_PITCH:
            # Not rejected: it is legitimate for a MONITOR camera. But an
            # attendance camera can never produce a query at this pose.
            flag = "  <-- outside the IN/OUT query pose gate; will never be matched there"

        print(f"  OK    {photo.name}: q={q.score:.3f} face={q.face_px:.0f}px "
              f"det={q.det_score:.2f} blur={q.blur_var:.0f} "
              f"yaw={q.yaw:+.0f} pitch={q.pitch:+.0f}{flag}")

        usable.append({
            "path": photo,
            "bytes": photo.read_bytes(),
            "embedding": face["embedding"],
            "aligned": bool(face.get("aligned", True)),
            "quality": q,
        })

    print()
    if not usable:
        print("No usable photo. Nothing to enrol.")
        return 1

    good_pose = sum(1 for u in usable
                    if abs(u["quality"].yaw) <= RUNTIME_MAX_YAW
                    and abs(u["quality"].pitch) <= RUNTIME_MAX_PITCH)
    print(f"{len(usable)} of {len(photos)} photo(s) usable; "
          f"{good_pose} within the IN/OUT query pose gate.")

    if not args.commit:
        print()
        print("DRY RUN — nothing written. Re-run with --commit to enrol"
              f"{' (add --replace to supersede the existing gallery)' if not args.replace else ''}.")
        return 0

    added = enroll_embeddings(
        employee_id=args.employee,
        observations=[
            {"embedding": u["embedding"], "aligned": u["aligned"], "quality": u["quality"]}
            for u in usable
        ],
        source="upload",
        replace=args.replace,
    )

    # Cover image, matching what the enrolment route stores. Only the first
    # photo gets a file; the rest exist as vectors only.
    with SessionLocal() as db:
        emp = db.query(Employee).filter(Employee.id == args.employee).first()
        if emp is not None:
            emp.photo_path = save_employee_photo(
                args.employee, usable[0]["bytes"], usable[0]["path"].name
            )
            db.commit()

    after = gallery_summary(args.employee)
    print()
    print(f"ENROLLED {added} vector(s).")
    print(f"Gallery after: {after['total']} active, "
          f"{after['stale_model_version']} on a superseded model, "
          f"model={after['model_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
