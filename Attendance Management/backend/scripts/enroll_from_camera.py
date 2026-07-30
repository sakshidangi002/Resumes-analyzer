"""Enrol an employee's face FROM THE CAMERA THAT WILL RECOGNISE THEM.

Why this exists
---------------
Recognition compares a live frame against enrolled reference photos. The
reference photos are studio-quality close-ups; the live frames are 27-45px faces
from a ceiling camera, downscaled and JPEG-compressed. Those are two different
image domains, and the model spends its discriminative power on the gap between
them rather than on identity.

MEASURED on this system's own enrolled photos (same person, same photo, only the
REFERENCE quality differing):

    reference              genuine   impostor   separation
    clean close-up           0.715      0.032        0.683
    degraded to ~30px        0.862      0.048        0.814

The degraded reference wins on BOTH axes — genuine similarity rises sharply
while impostor similarity barely moves. Matching the domain is worth more than
reference sharpness.

That is why the normal upload path's 90px minimum (FACE_ENROLL_MIN_FACE_PX)
rejects exactly the images that would help most here. This tool deliberately
takes a different path, with a lower but non-zero quality floor.

Deliberately TWO STEPS, because the tool cannot know who it is looking at:

    # 1. grab candidate faces and write them out for you to LOOK at
    python scripts/enroll_from_camera.py capture --camera 59 --shots 12

    # 2. after eyeballing the crops, attach the good ones to an employee
    python scripts/enroll_from_camera.py commit --dir <dir> --employee 4 --faces 1,3,5

New embeddings are APPENDED to the employee's existing stack, never replacing
it. The matcher takes the best over the stack, so adding camera-domain
references can only help genuine recall — and the enrollment-bias correction in
services/match.py keeps unequal photo counts from skewing the ranking.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.models.camera import CameraConfig  # noqa: E402
from app.models.employee import Employee  # noqa: E402
from app.services.camera_service import open_capture_with_timeout  # noqa: E402
from app.services.embedding_cache import (  # noqa: E402
    blob_to_embedding,
    embedding_to_blob,
    get_employee_candidates,
    invalidate_embedding_cache,
)
from app.services.face_service import extract_faces_from_rgb  # noqa: E402
from app.services.match import find_best_match  # noqa: E402

REVIEW_ROOT = Path(__file__).resolve().parents[1] / "data" / "enroll_review"

# Quality floor. Far below the upload path's 90px — the whole point is to accept
# camera-domain faces — but NOT zero: below ~20px an embedding carries no
# identity at all and would poison the gallery, matching everyone equally.
MIN_FACE_PX = int(__import__("os").getenv("ENROLL_CAM_MIN_FACE_PX", "20"))
MIN_DET_SCORE = float(__import__("os").getenv("ENROLL_CAM_MIN_DET_SCORE", "0.30"))
MAX_STACK = int(__import__("os").getenv("ENROLL_CAM_MAX_STACK", "20"))


def _face_px(face) -> float:
    box = face["box"]
    return float(box[2] - box[0])


def _resolve_source(camera_id: int | None, url: str | None) -> tuple[str, str]:
    if url:
        return url, f"url:{url[:40]}"
    with SessionLocal() as db:
        cam = db.query(CameraConfig).filter(CameraConfig.id == camera_id).first()
        if not cam:
            raise SystemExit(f"camera id {camera_id} not found")
        return cam.source_url, f"{cam.id} ({cam.name})"


def capture(args) -> int:
    source, label = _resolve_source(args.camera, args.url)
    cap = open_capture_with_timeout(source, "rtsp", args.camera or 0, timeout_sec=20)
    if cap is None:
        raise SystemExit("could not open the camera stream")

    out_dir = REVIEW_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    gallery = get_employee_candidates()

    print(f"camera {label}")
    print(f"capturing {args.shots} candidate face(s) every ~{args.every:.1f}s")
    print(f"quality floor: >= {MIN_FACE_PX}px, det_score >= {MIN_DET_SCORE}")
    print()

    manifest: list[dict] = []
    index = 0
    deadline = time.time() + args.shots * args.every + 30

    while len(manifest) < args.shots and time.time() < deadline:
        frame = None
        for _ in range(int(max(1, args.every * 10))):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        if frame is None:
            continue

        faces = extract_faces_from_rgb(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for face in sorted(faces, key=_face_px, reverse=True):
            px = _face_px(face)
            det = float(face.get("confidence", 0.0))
            if px < MIN_FACE_PX or det < MIN_DET_SCORE:
                continue

            index += 1
            x1, y1, x2, y2 = (int(v) for v in face["box"][:4])
            pad_x, pad_y = int((x2 - x1) * 0.4), int((y2 - y1) * 0.4)
            h, w = frame.shape[:2]
            crop = frame[max(0, y1 - pad_y):min(h, y2 + pad_y),
                         max(0, x1 - pad_x):min(w, x2 + pad_x)]
            if crop.size == 0:
                continue

            name = f"face_{index:02d}"
            cv2.imwrite(str(out_dir / f"{name}.jpg"), crop)
            np.save(out_dir / f"{name}.npy", face["embedding"])

            # Who does it currently look like? NOT used to assign anything —
            # only so the operator can sanity-check before committing.
            best = find_best_match(face["embedding"], gallery, threshold=0.0, min_margin=0.0)
            manifest.append({
                "id": index, "file": f"{name}.jpg", "face_px": round(px, 1),
                "det_score": round(det, 3),
                "current_best_match": best["employee_name"],
                "current_best_score": round(best["score"], 3),
            })
            print(f"  [{index:2d}] {px:5.0f}px  det={det:.2f}  "
                  f"looks most like: {best['employee_name']} ({best['score']:.3f})")
            break     # one face per frame — the largest

    cap.release()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    print(f"wrote {len(manifest)} candidate(s) to:")
    print(f"  {out_dir}")
    print()
    print("LOOK AT THE JPEGs, then commit only the ones that are the right person:")
    print(f"  python scripts/enroll_from_camera.py commit --dir \"{out_dir}\" "
          f"--employee <ID> --faces 1,2,3")
    return 0


def commit(args) -> int:
    review_dir = Path(args.dir)
    if not review_dir.is_dir():
        raise SystemExit(f"not a directory: {review_dir}")

    wanted = [int(x) for x in args.faces.split(",") if x.strip()]
    vectors = []
    for face_id in wanted:
        path = review_dir / f"face_{face_id:02d}.npy"
        if not path.exists():
            raise SystemExit(f"no such candidate: {path.name}")
        vectors.append(np.load(path).astype(np.float32))

    with SessionLocal() as db:
        emp = db.query(Employee).filter(Employee.id == args.employee).first()
        if not emp:
            raise SystemExit(f"employee {args.employee} not found")

        existing = None
        if emp.embedding is not None:
            existing = blob_to_embedding(emp.embedding)
            existing = existing[None, :] if existing.ndim == 1 else existing

        new_stack = np.stack(vectors).astype(np.float32)
        combined = new_stack if existing is None else np.vstack([existing, new_stack])

        # Keep the newest if the stack grows large — camera-domain references
        # are the ones we want to retain, and an unbounded stack slows matching.
        if combined.shape[0] > MAX_STACK:
            combined = combined[-MAX_STACK:]

        before = 0 if existing is None else existing.shape[0]
        print(f"employee {emp.id} — {emp.full_name}")
        print(f"  reference photos: {before} -> {combined.shape[0]} "
              f"(adding {new_stack.shape[0]} from the camera)")

        if args.dry_run:
            print("  DRY RUN — nothing written")
            return 0

        emp.embedding = embedding_to_blob(combined)
        emp.sample_count = int(combined.shape[0])
        db.commit()

    invalidate_embedding_cache()
    print("  committed; embedding cache invalidated")
    print()
    print("Recognition uses the BEST match across the stack, so the clean photos")
    print("still apply — these camera-domain references are additive.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap_p = sub.add_parser("capture", help="grab candidate faces for review")
    cap_p.add_argument("--camera", type=int, help="camera id from the cameras table")
    cap_p.add_argument("--url", help="use an RTSP url directly instead")
    cap_p.add_argument("--shots", type=int, default=10)
    cap_p.add_argument("--every", type=float, default=2.0, help="seconds between shots")
    cap_p.set_defaults(func=capture)

    com_p = sub.add_parser("commit", help="attach reviewed faces to an employee")
    com_p.add_argument("--dir", required=True)
    com_p.add_argument("--employee", type=int, required=True)
    com_p.add_argument("--faces", required=True, help="e.g. 1,3,5")
    com_p.add_argument("--dry-run", action="store_true")
    com_p.set_defaults(func=commit)

    args = parser.parse_args()
    if args.cmd == "capture" and not args.camera and not args.url:
        raise SystemExit("capture needs --camera or --url")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
