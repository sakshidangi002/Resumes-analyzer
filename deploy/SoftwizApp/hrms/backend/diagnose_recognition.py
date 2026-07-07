"""
diagnose_recognition.py
========================
Tells you EXACTLY why a person is / isn't recognised, using the same models and
embeddings the live cameras use.

Usage (from the backend folder, with the backend's Python env active):

    python diagnose_recognition.py path/to/photo.jpg [more.jpg ...]

For each photo it detects the face and prints the ranked cosine-similarity
scores against every enrolled employee, plus the margin over the runner-up and
whether it would pass the webcam / DVR thresholds.

Interpreting the output
-----------------------
* Your own name is #1 with a high score (>0.45) and a clear margin  -> good.
* Your own name is #1 but score is LOW (0.30-0.40)                  -> domain gap
  (enrol photos that look like what the camera sees) or threshold too high.
* SOMEONE ELSE is #1, or margin is tiny (<0.10)                     -> false match;
  raise the threshold / re-enrol the look-alikes with better photos.
* "No face detected"                                               -> the photo /
  camera frame is too small, dark, or side-on for the detector.
"""
from __future__ import annotations

import sys
from pathlib import Path

from app.core.config import get_settings
from app.services.embedding_cache import warm_embedding_cache, get_employee_candidates
from app.services.face_service import extract_faces_from_bytes
from app.services.match import cosine_similarity


def rank_photo(image_path: str, candidates: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(f"PHOTO: {image_path}")
    print("=" * 70)

    data = Path(image_path).read_bytes()
    faces = extract_faces_from_bytes(data)
    if not faces:
        print("  ❌ No face detected — photo too small / dark / side-on.")
        return

    if len(faces) > 1:
        print(f"  ⚠️  {len(faces)} faces found; scoring the largest one.")
    # Pick the largest face (most likely the subject).
    faces.sort(key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]), reverse=True)
    face = faces[0]
    print(f"  Detector confidence: {face['confidence']:.3f}")

    scored = []
    for c in candidates:
        s = cosine_similarity(face["embedding"], c["embedding"])
        scored.append((s, c["employee_name"], c["employee_id"]))
    scored.sort(reverse=True)

    if not scored:
        print("  ❌ No enrolled employees in the database.")
        return

    best_score = scored[0][0]
    runner_up = scored[1][0] if len(scored) > 1 else -1.0
    margin = best_score - runner_up if runner_up >= 0 else best_score

    print("\n  Rank  Score   Employee")
    print("  ----  ------  --------")
    for i, (s, name, eid) in enumerate(scored[:5], start=1):
        marker = "  <-- best" if i == 1 else ""
        print(f"  {i:>4}  {s:0.4f}  {name} (id={eid}){marker}")

    settings = get_settings()
    print(f"\n  Margin over runner-up: {margin:0.4f}  (need >= {settings.min_match_margin})")
    print(f"  Webcam threshold {settings.default_threshold}: "
          f"{'PASS' if best_score >= settings.default_threshold and margin >= settings.min_match_margin else 'FAIL'}")
    print(f"  DVR threshold    {settings.dvr_recognition_threshold}: "
          f"{'PASS' if best_score >= settings.dvr_recognition_threshold and margin >= settings.min_match_margin else 'FAIL'}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    n = warm_embedding_cache()
    print(f"Loaded {n} enrolled employee(s) from the database.")
    candidates = get_employee_candidates()

    for image_path in sys.argv[1:]:
        try:
            rank_photo(image_path, candidates)
        except FileNotFoundError:
            print(f"\n  ❌ File not found: {image_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n  ❌ Error on {image_path}: {exc}")


if __name__ == "__main__":
    main()
