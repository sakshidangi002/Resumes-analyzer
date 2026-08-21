"""Measure this deployment's recognition thresholds instead of guessing them.

WHY
---
Every threshold in the CCTV pipeline was justified by anecdote. The code
comments cite one mislabelled person ("a man was labelled Saloni Pathania at
77%") and a handful of remembered scores; `min_face_px` walked 45 -> 24 -> 16
because each tightening made the room cameras stop seeing anyone. There was no
measurement anywhere, so no one could say what the false-accept rate actually
was, or whether a change helped.

That anecdote also contains the key finding: correct matches scored 0.73-0.79
and the WRONG one scored 0.77. Score alone cannot separate them. This script
therefore sweeps the MARGIN and the minimum face size alongside the threshold,
because those are the axes that can.

INPUT
-----
A directory of labelled face crops captured from the real cameras:

    dataset/
      camera_2/                 <- directory name is free-form; --camera sets the id
        E001/  img1.jpg img2.jpg ...     <- subdirectory = employee_code
        E014/  ...
        _unknown/  ...          <- people NOT in the gallery (visitors, guests)

`_unknown` is optional but strongly recommended: without it the false-accept
rate can only be estimated from cross-identity confusions inside the gallery,
which understates the risk from strangers.

Collect it by pulling attendance snapshots (data/attendance_snapshots/) and
unknown-face crops (data/unknown_faces/) for one day and sorting them by hand.
An hour of sorting is worth more than a week of tuning by feel.

OUTPUT
------
Per camera: FAR / FRR / precision / recall across the sweep, the ROC and DET
points, and a RECOMMENDED (threshold, margin, min_face_px) chosen under an
explicit policy — the strictest operating point whose false accepts are at or
below `--max-far`, tie-broken by recall.

The recommendation is deliberately conservative for attendance cameras: a miss
is recoverable by HR keying the event in, a false accept writes the wrong
person's payroll data and is usually discovered weeks later, if at all.

USAGE
-----
    python scripts/calibrate_thresholds.py --dataset ./labelled --camera 2
    python scripts/calibrate_thresholds.py --dataset ./labelled --camera 2 \
        --max-far 0.0 --apply

`--apply` writes the recommendation into that camera's profile columns. Without
it the script only reports.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Make `app` importable when run as a script from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

UNKNOWN_DIR = "_unknown"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    """One labelled face crop, reduced to what the sweep needs."""

    label: Optional[str]      # employee_code, or None for a stranger
    embedding: np.ndarray
    face_px: float
    quality: float
    pose_ok: bool


@dataclass
class OperatingPoint:
    threshold: float
    margin: float
    min_face_px: float

    accepts: int = 0          # matched somebody
    true_accepts: int = 0     # matched the RIGHT somebody
    false_accepts: int = 0    # matched the WRONG somebody, or matched a stranger
    rejects: int = 0          # matched nobody
    false_rejects: int = 0    # was a known employee but matched nobody
    gated: int = 0            # never reached matching (failed the quality gate)

    @property
    def far(self) -> float:
        """P(accepting the wrong identity | an identity claim was possible).

        Denominator is every sample that could have produced a false accept —
        i.e. all of them, since a stranger and a mismatched employee are both
        false accepts.
        """
        total = self.accepts + self.rejects
        return (self.false_accepts / total) if total else 0.0

    @property
    def frr(self) -> float:
        """P(failing to recognise a genuine employee)."""
        known = self.true_accepts + self.false_rejects
        return (self.false_rejects / known) if known else 0.0

    @property
    def precision(self) -> float:
        return (self.true_accepts / self.accepts) if self.accepts else 1.0

    @property
    def recall(self) -> float:
        known = self.true_accepts + self.false_rejects
        return (self.true_accepts / known) if known else 0.0

    def as_row(self) -> dict:
        return {
            "threshold": round(self.threshold, 3),
            "margin": round(self.margin, 3),
            "min_face_px": round(self.min_face_px, 1),
            "FAR": round(self.far, 5),
            "FRR": round(self.frr, 5),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "true_accepts": self.true_accepts,
            "false_accepts": self.false_accepts,
            "false_rejects": self.false_rejects,
            "quality_gated": self.gated,
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_samples(camera_dir: Path, limits) -> list[Sample]:
    """Embed and grade every labelled crop under `camera_dir`.

    The SAME detector, embedder and quality module the live pipeline uses, so a
    threshold measured here means the same thing at runtime. Calibrating against
    a different preprocessing path is the classic way to produce numbers that do
    not transfer.
    """
    from PIL import Image

    from app.services.face_quality import assess
    from app.services.face_service import extract_faces_from_image

    samples: list[Sample] = []
    for label_dir in sorted(p for p in camera_dir.iterdir() if p.is_dir()):
        label = None if label_dir.name == UNKNOWN_DIR else label_dir.name
        for image_path in sorted(label_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                print(f"  ! unreadable: {image_path}", file=sys.stderr)
                continue

            faces = extract_faces_from_image(image)
            if not faces:
                continue
            # The largest face — a crop should contain one person, and any
            # bystander caught in the padding is smaller.
            face = max(
                faces,
                key=lambda f: abs(float(f["box"][2]) - float(f["box"][0])),
            )
            quality = assess(face, np.asarray(image), limits=limits)
            samples.append(
                Sample(
                    label=label,
                    embedding=np.asarray(face["embedding"], dtype=np.float32),
                    face_px=quality.face_px,
                    quality=quality.score,
                    pose_ok=quality.reason not in {"pose_yaw", "pose_pitch", "landmark_asym"},
                )
            )
    return samples


def load_gallery() -> list[dict]:
    """The live enrolled gallery, exactly as the matcher sees it."""
    from app.services.embedding_cache import get_employee_candidates

    return get_employee_candidates()


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def sweep(
    samples: Iterable[Sample],
    gallery: list[dict],
    thresholds: Iterable[float],
    margins: Iterable[float],
    face_sizes: Iterable[float],
) -> list[OperatingPoint]:
    from app.services.match import find_best_match

    samples = list(samples)
    code_of = {int(c["employee_id"]): str(c.get("employee_code") or "") for c in gallery}

    # Score every sample against the gallery ONCE. The sweep then only re-applies
    # thresholds to cached (best, runner-up) pairs, which turns an O(sweep x
    # samples x gallery) job into O(samples x gallery).
    scored = []
    for sample in samples:
        result = find_best_match(sample.embedding, gallery, threshold=-1.0, min_margin=0.0)
        scored.append(
            (
                sample,
                float(result.get("score") or -1.0),
                float(result.get("margin") or 0.0),
                code_of.get(int(result.get("employee_id") or -1), ""),
            )
        )

    points: list[OperatingPoint] = []
    for min_face in face_sizes:
        for threshold in thresholds:
            for margin in margins:
                point = OperatingPoint(threshold, margin, min_face)
                for sample, score, sample_margin, matched_code in scored:
                    if sample.face_px < min_face:
                        point.gated += 1
                        # A gated sample is not a false accept — but for a known
                        # employee it IS a miss, and hiding that would make a
                        # brutally strict min_face look free.
                        if sample.label is not None:
                            point.false_rejects += 1
                        continue

                    accepted = score >= threshold and sample_margin >= margin
                    if not accepted:
                        point.rejects += 1
                        if sample.label is not None:
                            point.false_rejects += 1
                        continue

                    point.accepts += 1
                    if sample.label is not None and matched_code == sample.label:
                        point.true_accepts += 1
                    else:
                        # Either a stranger was accepted, or an employee was
                        # accepted as somebody else. Both write the wrong
                        # person's attendance.
                        point.false_accepts += 1
                points.append(point)
    return points


def recommend(points: list[OperatingPoint], max_far: float) -> Optional[OperatingPoint]:
    """Best operating point under an explicit policy.

    Policy: among points whose FAR is at or below `max_far`, take the highest
    recall; break ties toward the LOWER minimum face size (keeps more people
    recognisable) and then the higher threshold (more headroom).

    Returns None when no point meets the FAR ceiling — which is itself the
    finding, and means the gallery or the camera view has to improve before any
    threshold is safe.
    """
    eligible = [p for p in points if p.far <= max_far]
    if not eligible:
        return None
    return max(eligible, key=lambda p: (p.recall, -p.min_face_px, p.threshold))


def det_curve(points: list[OperatingPoint]) -> list[dict]:
    """(FAR, FRR) pairs on the Pareto front — the DET curve.

    Only non-dominated points: one that is worse on both axes than another
    cannot be an operating point anybody would choose, and printing it just
    hides the real trade-off.
    """
    ordered = sorted(points, key=lambda p: (p.far, p.frr))
    front: list[OperatingPoint] = []
    best_frr = float("inf")
    for point in ordered:
        if point.frr < best_frr:
            front.append(point)
            best_frr = point.frr
    return [
        {
            "FAR": round(p.far, 5), "FRR": round(p.frr, 5),
            "threshold": round(p.threshold, 3), "margin": round(p.margin, 3),
            "min_face_px": round(p.min_face_px, 1),
        }
        for p in front
    ]


def apply_to_camera(camera_id: int, point: OperatingPoint) -> None:
    """Write a recommendation into the camera's profile columns."""
    from app.db.session import SessionLocal
    from app.models.camera import CameraConfig
    from app.services import camera_profile

    with SessionLocal() as db:
        row = db.query(CameraConfig).filter(CameraConfig.id == camera_id).first()
        if row is None:
            raise SystemExit(f"camera {camera_id} not found")
        row.threshold = float(point.threshold)
        row.match_margin = float(point.margin)
        row.min_face_px = int(round(point.min_face_px))
        db.commit()
    camera_profile.invalidate(camera_id)
    print(f"\nApplied to camera {camera_id}. Live within ~10s, no restart needed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Directory of labelled crops for ONE camera")
    parser.add_argument("--camera", required=True, type=int, help="Camera id")
    parser.add_argument("--max-far", type=float, default=0.0,
                        help="Highest acceptable false-accept rate (default 0.0)")
    parser.add_argument("--thresholds", default="0.35:0.70:0.025",
                        help="start:stop:step")
    parser.add_argument("--margins", default="0.05:0.35:0.025", help="start:stop:step")
    parser.add_argument("--face-sizes", default="16:60:4", help="start:stop:step")
    parser.add_argument("--json", type=Path, help="Write the full sweep to this file")
    parser.add_argument("--apply", action="store_true",
                        help="Write the recommendation into the camera profile")
    args = parser.parse_args()

    if not args.dataset.is_dir():
        raise SystemExit(f"not a directory: {args.dataset}")

    from app.services import camera_profile

    profile = camera_profile.get_profile(args.camera)
    print(f"Camera {args.camera}: {profile.describe()}\n")

    # Load with permissive limits so the sweep can explore below the current
    # gate. Calibrating inside the gate you are trying to calibrate would only
    # ever confirm the current setting.
    from dataclasses import replace as _replace

    permissive = _replace(
        profile.limits, min_face_px=1.0, min_det_score=0.0,
        max_yaw_deg=90.0, max_pitch_deg=90.0, max_landmark_asym=1.0, min_blur_var=0.0,
    )

    print(f"Loading {args.dataset} ...")
    samples = load_samples(args.dataset, permissive)
    known = sum(1 for s in samples if s.label is not None)
    print(f"  {len(samples)} faces ({known} labelled, {len(samples) - known} stranger)")
    if not samples:
        raise SystemExit("no usable faces found — check the dataset layout")
    if known == len(samples):
        print("  ! no _unknown/ directory: the false-accept rate will be measured")
        print("    only from cross-identity confusions and understates stranger risk.")

    gallery = load_gallery()
    print(f"  gallery: {len(gallery)} enrolled employees\n")
    if not gallery:
        raise SystemExit("the enrolled gallery is empty — nothing to calibrate against")

    points = sweep(
        samples, gallery,
        _span(args.thresholds), _span(args.margins), _span(args.face_sizes),
    )

    best = recommend(points, args.max_far)
    print(f"Swept {len(points)} operating points.\n")

    if best is None:
        print(f"NO operating point reaches FAR <= {args.max_far}.")
        print("That is the finding: no threshold makes this camera safe as it")
        print("stands. Improve the gallery first — enrol the affected employees")
        print("from THIS camera's viewpoint (the unknown-face review queue is the")
        print("fastest route) — then re-run.")
        lowest = min(points, key=lambda p: (p.far, -p.recall))
        print(f"\nLowest FAR available: {json.dumps(lowest.as_row(), indent=2)}")
    else:
        print("RECOMMENDED (strictest point meeting the FAR ceiling, best recall):")
        print(json.dumps(best.as_row(), indent=2))
        print(
            f"\n  Set threshold={best.threshold:.3f} margin={best.margin:.3f} "
            f"min_face_px={best.min_face_px:.0f} on camera {args.camera}."
        )

    print("\nDET curve (Pareto front):")
    for row in det_curve(points)[:15]:
        print(f"  {row}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "camera_id": args.camera,
                    "samples": len(samples),
                    "gallery": len(gallery),
                    "recommended": best.as_row() if best else None,
                    "det_curve": det_curve(points),
                    "sweep": [p.as_row() for p in points],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nFull sweep written to {args.json}")

    if args.apply:
        if best is None:
            raise SystemExit("refusing to apply: no point meets the FAR ceiling")
        apply_to_camera(args.camera, best)

    return 0


def _span(spec: str) -> list[float]:
    start, stop, step = (float(part) for part in spec.split(":"))
    if step <= 0:
        raise SystemExit(f"step must be positive: {spec}")
    values, current = [], start
    while current <= stop + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


if __name__ == "__main__":
    raise SystemExit(main())
