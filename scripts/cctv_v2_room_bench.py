"""Measure how many of the people who are REALLY in a room camera's view get
detected, and at what score -- so detector floor and tracker floor can be set
from evidence instead of from each other.

MANUAL / MEASUREMENT ONLY. Needs the live cameras (for `capture`) and the YOLO
weights (for `sweep`), so it is deliberately not importable by the automated
suite -- `pytest` must stay runnable on a box with no DVR and no model file.

WHY THIS EXISTS
---------------
The room cameras have been tuned by a sequence of one-off measurements recorded
in .env comments, and those comments now disagree with each other: one block
concludes `imgsz=960` finds three people where 640 finds two, a later block
concludes 480 beats all of them. Both were true of the frames they were measured
on. Neither was measured against a fixed, labelled set of the HARD cases -- the
seated, back-facing, half-occluded people this camera exists to count -- so
there was no way to tell a real improvement from a change of sample.

This fixes the sample. Frames are captured once, labelled by hand with the
number of people a human can actually see, and every configuration is scored
against that same set.

THE ONE MEASUREMENT TRICK THAT MATTERS
--------------------------------------
Detection floor and track-creation floor are SEPARATE numbers, and sweeping both
would multiply the run time for no information. So the detector is run ONCE per
(model, imgsz, crop) at a floor low enough to be effectively off (0.01), and
every candidate threshold is then evaluated ARITHMETICALLY against the recorded
scores.

That is not a shortcut, it is the more informative experiment: it yields the
score each real person actually achieved, which is the quantity a threshold has
to sit below. A conf sweep only ever tells you whether a person survived a
particular bar, never by how much.

RECALL IS COUNTED AGAINST BOXES, NOT TOTALS
-------------------------------------------
Ground truth is per-person boxes, not a headcount. A frame with 3 people where
the detector fires 3 times can still be wrong -- two boxes on one person and
none on another scores 3/3 on a headcount and 2/3 honestly. Matching is greedy
by score at IoU >= 0.3 (loose on purpose: a box that lands on the right person
counts even if it includes their chair, because presence is the question, not
segmentation).

    capture   grab spaced frames from the room cameras   -> data/cctv_v2_rooms/
    label     write/refresh the ground-truth stub for those frames
    sweep     score every configuration against the labels
    render    draw one configuration's detections for visual checking

Usage:
    python scripts/cctv_v2_room_bench.py capture --cameras 59,60 --count 10
    python scripts/cctv_v2_room_bench.py sweep --imgsz 480,640,960,1280
    python scripts/cctv_v2_room_bench.py render --frame 59_00003 --imgsz 960
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Attendance Management" / "backend"
for p in (str(BACKEND), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Which labelled set to work on. Overridable so a SECOND, independent set can
# be captured and scored without overwriting the first -- the frame names are
# just "{camera}_{index}.jpg", so two sessions in one directory would silently
# clobber each other and leave the labels pointing at different pixels.
FRAMES = REPO / "data" / os.environ.get("CCTV_BENCH_DIR", "cctv_v2_rooms")
TRUTH = FRAMES / "ground_truth.json"
RESULTS = FRAMES / "sweep.json"

# IoU at which a detection is accepted as covering a labelled person. Loose on
# purpose -- see the module docstring.
MATCH_IOU = 0.30

# The crop-assist window, copied from bytetrack_engine._add_crop_assist_tracks so
# this measures the crop the pipeline would actually run, not an idealised one.
CROP_X, CROP_Y = 0.15, 0.10

# Thresholds evaluated arithmetically against the recorded scores. Spans the
# range between "effectively off" and the strict doorway config.
THRESHOLDS = (0.010, 0.015, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30)


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------
def cmd_capture(args) -> int:
    import cv2

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    from app.db.session import SessionLocal
    from app.models.camera import CameraConfig

    cams = [int(c) for c in args.cameras.split(",") if c.strip()]
    FRAMES.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as db:
        urls = {
            c: db.query(CameraConfig).filter(CameraConfig.id == c).first()
            for c in cams
        }
    for cid, row in urls.items():
        if row is None or not row.source_url:
            print(f"camera {cid}: no source_url")
            return 1

    caps = {}
    for cid, row in urls.items():
        cap = cv2.VideoCapture(row.source_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"camera {cid}: could not open stream")
            return 1
        caps[cid] = cap
        print(f"camera {cid}: open")

    written = 0
    for i in range(args.count):
        for cid, cap in caps.items():
            # Drain the decoder so the frame written is CURRENT, not the oldest
            # buffered one. Without this every "spaced" frame is really the
            # frame that arrived seconds ago and the set has no variety.
            for _ in range(args.drain):
                cap.grab()
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"  camera {cid}: read failed at {i}")
                continue
            path = FRAMES / f"{cid}_{i:05d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
        print(f"  [{i + 1}/{args.count}] captured")
        if i + 1 < args.count:
            time.sleep(args.interval)

    for cap in caps.values():
        cap.release()
    print(f"\nwrote {written} frames to {FRAMES}")
    print("next: label them, then run `sweep`")
    return 0


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------
def cmd_label(args) -> int:
    """Create or refresh the ground-truth stub.

    Deliberately does NOT invent labels. A frame nobody has looked at gets
    `people: null`, and `sweep` refuses to score it -- an unlabelled frame
    silently treated as empty would make every configuration look perfect on it.
    """
    truth = _load_truth()
    names = sorted(p.stem for p in FRAMES.glob("*.jpg"))
    added = 0
    for name in names:
        if name not in truth:
            truth[name] = {"people": None, "boxes": [], "note": ""}
            added += 1
    TRUTH.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    labelled = sum(1 for v in truth.values() if v.get("people") is not None)
    print(f"{TRUTH}: {len(truth)} frames, {labelled} labelled, {added} new stubs")
    return 0


def _load_truth() -> dict:
    if TRUTH.exists():
        return json.loads(TRUTH.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------
def cmd_propose(args) -> int:
    """Render numbered candidate boxes at a near-zero floor, for adjudication.

    Labelling by reading pixel coordinates off a picture by eye is both slow and
    imprecise, and an imprecise box moves the IoU match it is supposed to be
    judging. So the detector proposes and a human disposes: every candidate is
    drawn with an index and a score, and the reviewer says which indices are
    real people.

    This CANNOT bias the recall measurement in the detector's favour, because
    the floor here is far below any threshold under test and the reviewer is
    also asked what the detector MISSED. A person with no candidate box is
    added by hand -- those are exactly the cases that matter, and they stay
    visible as `boxes` entries with no proposal behind them.
    """
    import cv2
    from ultralytics import YOLO

    model_path = _resolve_model(args.model)
    if model_path is None:
        print(f"model not found: {args.model}")
        return 1
    model = YOLO(model_path)

    names = [args.frame] if args.frame else sorted(
        p.stem for p in FRAMES.glob("*.jpg") if not p.stem.startswith(("render", "propose")))
    out_dir = FRAMES / "propose"
    out_dir.mkdir(parents=True, exist_ok=True)

    catalogue = {}
    for name in names:
        path = FRAMES / f"{name}.jpg"
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        dets, _, _ = _detect_with_crop(model, frame, args.imgsz, args.floor,
                                       args.iou, True)
        dets = sorted(dets, key=lambda d: -d[0])[:args.max_boxes]
        catalogue[name] = [
            {"i": i, "score": round(s, 4),
             "box": [round(v, 1) for v in b]}
            for i, (s, b) in enumerate(dets)
        ]
        canvas = frame.copy()
        for i, (score, (x1, y1, x2, y2)) in enumerate(dets):
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 220, 255), 2)
            tag = f"{i}:{score:.3f}"
            cv2.rectangle(canvas, (int(x1), int(y1) - 18),
                          (int(x1) + 9 * len(tag), int(y1)), (0, 0, 0), -1)
            cv2.putText(canvas, tag, (int(x1) + 2, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1,
                        cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{name}.jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{name}: {len(dets)} candidates")

    (out_dir / "candidates.json").write_text(
        json.dumps(catalogue, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir}/  (renders + candidates.json)")
    print("Review each render, then write the real ones into ground_truth.json")
    return 0


def cmd_accept(args) -> int:
    """Build the ground truth from an adjudication file.

    The file says, per frame, which proposed candidates are real people (`pick`)
    and which people the detector produced NO candidate for at all (`add`, boxes
    read off the picture by hand).

    `add` is the important half. A benchmark built only from candidates the
    detector offered can never score below 100% on the people it cannot see,
    because those people would not be in the denominator. The hand-added boxes
    ARE the misses, and they are what makes recall an honest number here.

    Anything proposed and not picked is, by definition, not a person -- so the
    false-positive evidence comes out of the same file for free.
    """
    cat_path = FRAMES / "propose" / "candidates.json"
    if not cat_path.exists():
        print("no candidates.json; run `propose` first")
        return 1
    catalogue = json.loads(cat_path.read_text(encoding="utf-8"))
    adj = json.loads(Path(args.file).read_text(encoding="utf-8"))
    truth = _load_truth()

    for name, spec in adj.items():
        if name.startswith("_"):          # comment keys
            continue
        by_i = {c["i"]: c["box"] for c in catalogue.get(name, [])}
        picks = spec.get("pick", [])
        missing = [i for i in picks if i not in by_i]
        if missing:
            print(f"{name}: no such candidate {missing}")
            return 1
        boxes = [by_i[i] for i in picks] + [list(map(float, b))
                                            for b in spec.get("add", [])]
        # Why each person is HARD, parallel to `boxes`. Recall averaged over a
        # mixed set hides the thing that matters: a set with one easy standing
        # person and three occluded seated ones reports 25% or 100% depending
        # only on which of them the detector happened to find.
        cats = spec.get("categories", [])
        if cats and len(cats) != len(boxes):
            print(f"{name}: {len(cats)} categories for {len(boxes)} boxes")
            return 1
        truth[name] = {
            "people": len(boxes),
            "boxes": boxes,
            "categories": cats,
            "detected_by_proposal": len(picks),
            "invisible_to_proposal": len(spec.get("add", [])),
            "note": spec.get("note", ""),
        }
    TRUTH.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    labelled = [v for v in truth.values() if v.get("people") is not None]
    total = sum(v["people"] for v in labelled)
    unseen = sum(v.get("invisible_to_proposal", 0) for v in labelled)
    print(f"{TRUTH}: {len(labelled)} frames, {total} people")
    print(f"  {unseen} of them produced NO box at all even at floor 0.005 "
          f"({unseen / max(1, total):.0%})")
    return 0


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _detect(model, frame, imgsz: int, conf: float, iou: float):
    """Person boxes for one frame. Returns [(score, box), ...]."""
    res = model.predict(frame, classes=[0], conf=conf, iou=iou,
                        imgsz=imgsz, verbose=False)[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    out = []
    for score, box in zip(boxes.conf.cpu().tolist(), boxes.xyxy.cpu().tolist()):
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        if x2 > x1 and y2 > y1:
            out.append((float(score), (x1, y1, x2, y2)))
    return out


def _detect_with_crop(model, frame, imgsz: int, conf: float, iou: float,
                      crop: bool):
    """Full-frame detections, optionally merged with one overlapping crop pass.

    The merge rule matches bytetrack_engine: a crop detection is kept only if it
    does not already overlap a full-frame detection, so the crop can ADD people
    but never duplicate one.
    """
    full = _detect(model, frame, imgsz, conf, iou)
    if not crop:
        return full, 0, 0

    h, w = frame.shape[:2]
    cx, cy = int(w * CROP_X), int(h * CROP_Y)
    sub = frame[cy:h, cx:w]
    if sub.size == 0:
        return full, 0, 0
    raw = _detect(model, sub, imgsz, max(0.02, conf * 0.5), iou)

    added, dupes = 0, 0
    merged = list(full)
    for score, (x1, y1, x2, y2) in raw:
        box = (x1 + cx, y1 + cy, x2 + cx, y2 + cy)
        if any(_iou(box, f[1]) > 0.45 for f in full):
            dupes += 1
            continue
        merged.append((score, box))
        added += 1
    return merged, added, dupes


def _score_frame(dets, gt_boxes):
    """Best detection score per labelled person, plus scores of unmatched boxes.

    Greedy by detection score: the most confident box claims its best person
    first. That is the order a tracker would see them in, and it stops a weak
    box stealing a person from a strong one.
    """
    claimed = {}
    unmatched = []
    for score, box in sorted(dets, key=lambda d: -d[0]):
        best_i, best_iou = None, MATCH_IOU
        for i, gt in enumerate(gt_boxes):
            if i in claimed:
                continue
            v = _iou(box, gt)
            if v >= best_iou:
                best_i, best_iou = i, v
        if best_i is None:
            unmatched.append((score, box))
        else:
            claimed[best_i] = score
    per_person = [claimed.get(i) for i in range(len(gt_boxes))]
    return per_person, unmatched


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def cmd_sweep(args) -> int:
    import cv2
    from ultralytics import YOLO

    truth = _load_truth()
    frames = []
    for name in sorted(truth):
        gt = truth[name]
        if gt.get("people") is None:
            continue
        cid = int(name.split("_")[0])
        if args.cameras and cid not in [int(c) for c in args.cameras.split(",")]:
            continue
        path = FRAMES / f"{name}.jpg"
        if not path.exists():
            continue
        frames.append((name, cid, path, gt))
    if not frames:
        print("no labelled frames; run `label`, fill in `people` and `boxes`")
        return 1

    sizes = [int(s) for s in args.imgsz.split(",")]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    crops = [False, True] if args.crop == "both" else [args.crop == "on"]

    print(f"{len(frames)} labelled frames, "
          f"{sum(f[3]['people'] for f in frames)} labelled people")
    print(f"models={models} imgsz={sizes} crop={crops} floor={args.floor}\n")

    loaded = {}
    results = []
    for model_name in models:
        path = _resolve_model(model_name)
        if path is None:
            print(f"model not found: {model_name}")
            return 1
        if model_name not in loaded:
            loaded[model_name] = YOLO(path)
        model = loaded[model_name]

        for imgsz in sizes:
            for crop in crops:
                row = _run_config(cv2, model, model_name, imgsz, crop,
                                  frames, args)
                results.append(row)
                _print_config(row)

    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS}")
    _print_summary(results)
    return 0


def _run_config(cv2, model, model_name, imgsz, crop, frames, args) -> dict:
    per_camera = {}
    total_ms = 0.0
    crop_added = crop_dupes = 0

    for name, cid, path, gt in frames:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        t0 = time.time()
        dets, added, dupes = _detect_with_crop(
            model, frame, imgsz, args.floor, args.iou, crop)
        total_ms += (time.time() - t0) * 1000.0
        if args.min_box > 0:
            # Discard boxes too small to be a person in THIS view.
            #
            # Not a confidence question. At a floor low enough to reach a seated
            # person, the detector also emits fragments a few pixels tall on
            # frame edges and decode-corruption bands; camera 60 produced seven
            # of them in one frame, all about 10x7. The smallest labelled real
            # person here is 46px tall, so a size floor separates the two
            # cleanly where no score threshold can -- some of that junk outscores
            # a genuinely seated person.
            dets = [d for d in dets if (d[1][3] - d[1][1]) >= args.min_box]
        crop_added += added
        crop_dupes += dupes

        gt_boxes = [tuple(b) for b in gt.get("boxes", [])]
        per_person, unmatched = _score_frame(dets, gt_boxes)
        cam = per_camera.setdefault(cid, {"person_scores": [], "fp_scores": [],
                                          "fp_heights": [],
                                          "frames": 0, "people": 0})
        cam["frames"] += 1
        cam["people"] += len(gt_boxes)
        cam["person_scores"].extend(per_person)
        cam["fp_scores"].extend(s for s, _ in unmatched)
        # Kept alongside the scores so a SIZE floor can be evaluated from the
        # same run as a SCORE floor. They reject different things: a size floor
        # removes the few-pixel fragments a low score floor lets through, and
        # some of those fragments outscore a genuinely seated person.
        cam["fp_heights"].extend(round(b[3] - b[1], 1) for _, b in unmatched)

    n_passes = sum(c["frames"] for c in per_camera.values())
    return {
        "model": model_name,
        "imgsz": imgsz,
        "crop_assist": crop,
        "avg_ms": round(total_ms / max(1, n_passes), 1),
        "crop_added": crop_added,
        "crop_duplicates_suppressed": crop_dupes,
        "cameras": {
            str(cid): {
                "frames": c["frames"],
                "people": c["people"],
                # None = the person produced NO box at any score. A threshold
                # cannot rescue those; only a better model or more pixels can.
                "person_scores": [None if s is None else round(s, 4)
                                  for s in c["person_scores"]],
                "fp_scores": [round(s, 4) for s in c["fp_scores"]],
                "fp_heights": c["fp_heights"],
                "recall_at": {
                    f"{t:.3f}": _recall(c["person_scores"], t)
                    for t in THRESHOLDS
                },
                "false_at": {
                    f"{t:.3f}": sum(1 for s in c["fp_scores"] if s >= t)
                    for t in THRESHOLDS
                },
            }
            for cid, c in sorted(per_camera.items())
        },
    }


def _recall(scores, threshold) -> float:
    if not scores:
        return 0.0
    found = sum(1 for s in scores if s is not None and s >= threshold)
    return round(found / len(scores), 3)


def _print_config(row) -> None:
    tag = (f"{row['model']:<14} imgsz={row['imgsz']:<5} "
           f"crop={'on ' if row['crop_assist'] else 'off'} "
           f"{row['avg_ms']:>7.0f}ms")
    print(tag)
    for cid, c in row["cameras"].items():
        scored = [s for s in c["person_scores"] if s is not None]
        weakest = f"weakest {min(scored):.3f}" if scored else "nothing scored"
        print(f"    cam{cid}: {len(scored)}/{c['people']} people produced a "
              f"box   {weakest}")
        r = c["recall_at"]
        f = c["false_at"]
        print("        recall  " + "  ".join(
            f"@{t}={r[t]:.2f}" for t in ("0.020", "0.030", "0.050", "0.100", "0.200")))
        print("        false   " + "  ".join(
            f"@{t}={f[t]:<3d}" for t in ("0.020", "0.030", "0.050", "0.100", "0.200")))


def _print_summary(results) -> None:
    print("\n" + "=" * 78)
    print("RECALL BY CONFIG, AT THREE CANDIDATE TRACK-CREATION FLOORS")
    print("=" * 78)
    for t in ("0.010", "0.020", "0.030"):
        print(f"\n--- track-creation floor {t} " + "-" * 40)
        print(f"{'model':<20}{'imgsz':>6}{'crop':>6}{'cam59':>8}{'cam60':>8}"
              f"{'false':>8}{'ms':>9}")
        for row in results:
            c59 = row["cameras"].get("59", {}).get("recall_at", {}).get(t, 0.0)
            c60 = row["cameras"].get("60", {}).get("recall_at", {}).get(t, 0.0)
            fp = sum(c.get("false_at", {}).get(t, 0)
                     for c in row["cameras"].values())
            print(f"{row['model']:<20}{row['imgsz']:>6}"
                  f"{'on' if row['crop_assist'] else 'off':>6}"
                  f"{c59:>8.2f}{c60:>8.2f}{fp:>8d}{row['avg_ms']:>9.0f}")


def _resolve_model(name: str):
    for cand in (Path(name), BACKEND / name, BACKEND / "models" / name):
        if cand.exists():
            return str(cand)
    return None


# ---------------------------------------------------------------------------
# assoc -- which part of a person says WHICH SEAT they are in
# ---------------------------------------------------------------------------
# Candidate regions, as fractions of the person's box measured from its top.
# Each returns the sub-box that is tested against a seat zone.
#
# These exist because the shipped rule -- the LOWER HALF of the box -- was
# reasoned about rather than measured, and the reasoning assumed a standing
# person whose feet are the part in contact with the floor. A person seated at a
# desk is not that: their box bottom is where the desk edge or the chair base
# cuts them off, several tens of pixels below the seat, and the half of the box
# above it is torso, not lap.
SEAT_REGIONS = {
    "whole_box":   (0.00, 1.00),
    "lower_half":  (0.50, 1.00),   # what ships today
    "lower_third": (0.67, 1.00),
    "torso_band":  (0.30, 0.75),
    "mid_band":    (0.40, 0.90),
    "upper_half":  (0.00, 0.50),
}


def _region_box(box, lo: float, hi: float):
    x1, y1, x2, y2 = box
    h = y2 - y1
    return (x1, y1 + h * lo, x2, y1 + h * hi)


def _assoc_for(rule: str, person_box, zones, w, h, min_overlap: float):
    """The seat this rule assigns a person to, or None.

    Mirrors RoomOccupancy.update: best seat by fraction-of-seat covered, and
    only if it clears `min_overlap`. Everything that differs between rules is in
    WHICH PART of the person is tested, so a difference in the result is a
    difference in that choice and nothing else.
    """
    lo, hi = SEAT_REGIONS[rule]
    rx1, ry1, rx2, ry2 = _region_box(person_box, lo, hi)
    region = (rx1 / w, ry1 / h, rx2 / w, ry2 / h)
    best_id, best = None, 0.0
    for chair_id, zone in zones.items():
        px1, py1, px2, py2 = region
        sx1, sy1, sx2, sy2 = zone
        ix1, iy1 = max(px1, sx1), max(py1, sy1)
        ix2, iy2 = min(px2, sx2), min(py2, sy2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        seat_area = max(1e-9, (sx2 - sx1) * (sy2 - sy1))
        frac = (iw * ih) / seat_area
        if frac > best:
            best, best_id = frac, chair_id
    return (best_id, best) if best >= min_overlap else (None, best)


def _assign_frame(rule, boxes, zones, w, h, min_overlap, max_drop):
    """Assign a whole frame's people to seats, exactly as RoomOccupancy does.

    Scored per FRAME rather than per person because the real algorithm resolves
    competition between people: each person claims at most their single best
    seat, and where two people claim one seat the stronger claim wins. Judging
    each person in isolation misses both effects, and it was that isolated view
    which made every rule look like it seats a standing person -- in the real
    algorithm the person actually sitting there often outbids them.
    """
    claims = {}                     # chair_id -> (frac, person_index)
    for i, box in enumerate(boxes):
        chair_id, frac = _assoc_for(rule, box, zones, w, h, min_overlap)
        if chair_id is None:
            continue
        if max_drop is not None:
            # How far the person's lowest visible point falls BELOW the seat.
            # Someone sitting in a chair cannot extend far under it; someone
            # standing in the aisle in front of it can, and in a receding view
            # that is most of what separates the two.
            if (box[3] / h) - zones[chair_id][3] > max_drop:
                continue
        prev = claims.get(chair_id)
        if prev is None or frac > prev[0]:
            claims[chair_id] = (frac, i)
    return {i: cid for cid, (_, i) in claims.items()}


def cmd_assoc(args) -> int:
    """Score every candidate seat-association rule against hand-labelled seats.

    Run on GROUND-TRUTH person boxes, never on detections. Association and
    detection are separate failures and mixing them makes both unreadable: a
    rule cannot be blamed for a person the detector never produced.
    """
    truth = _load_truth()
    seats = json.loads((FRAMES / "seat_truth.json").read_text(encoding="utf-8"))
    from app.cctv_v2.config.geometry import room_geometry

    drops = [None] + [float(d) for d in args.max_drop.split(",")
                      if d.strip()] if args.max_drop else [None]
    rows = []
    for rule in args.rules.split(","):
        rule = rule.strip()
        if rule not in SEAT_REGIONS:
            print(f"unknown rule {rule!r}; have {sorted(SEAT_REGIONS)}")
            return 1
        for drop in drops:
            correct = wrong = missed = spurious = 0
            for name, spec in sorted(seats.items()):
                if name.startswith("_"):
                    continue
                gt = truth.get(name)
                if not gt:
                    continue
                cid = int(name.split("_")[0])
                zones = {c.chair_id: c.box for c in room_geometry(cid).chairs}
                if not zones:
                    continue
                got = _assign_frame(rule, gt["boxes"], zones,
                                    args.width, args.height,
                                    args.min_overlap, drop)
                for i, want in enumerate(spec):
                    want = want or None
                    mine = got.get(i)
                    if want is None and mine is None:
                        continue                  # correctly left unassigned
                    if want is None:
                        spurious += 1             # forced into a seat
                    elif mine is None:
                        missed += 1               # really seated, not assigned
                    elif mine == want:
                        correct += 1
                    else:
                        wrong += 1                # assigned the WRONG seat
            total = correct + wrong + missed
            rows.append({
                "rule": rule, "region": SEAT_REGIONS[rule], "max_drop": drop,
                "correct": correct, "wrong_seat": wrong, "missed": missed,
                "spurious": spurious,
                "accuracy": round(correct / total, 3) if total else 0.0,
            })

    print(f"{'rule':<13}{'region':>12}{'drop':>7}{'correct':>9}{'wrong':>7}"
          f"{'missed':>8}{'spurious':>10}{'acc':>7}")
    for r in rows:
        lo, hi = r["region"]
        drop = "-" if r["max_drop"] is None else f"{r['max_drop']:.2f}"
        print(f"{r['rule']:<13}{f'{lo:.2f}-{hi:.2f}':>12}{drop:>7}"
              f"{r['correct']:>9}{r['wrong_seat']:>7}{r['missed']:>8}"
              f"{r['spurious']:>10}{r['accuracy']:>7.2f}")
    print("\ncorrect  = right seat        wrong = a DIFFERENT mapped seat")
    print("missed   = seated, no seat   spurious = not seated, given a seat")
    (FRAMES / "assoc.json").write_text(json.dumps(rows, indent=2),
                                       encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# persons -- per-frame recall is not what the live count experiences
# ---------------------------------------------------------------------------
def cmd_persons(args) -> int:
    """Break recall down PER PERSON, and estimate what the tracker sees.

    Per-frame recall understates the live count, and by a lot. A track survives
    several analysis ticks without a detection (ByteTrack `track_buffer`, plus
    the room profile's 30s hold), so somebody found on one pass in three is
    counted CONTINUOUSLY, not a third of the time. Tuning on per-frame recall
    alone would buy expensive settings for people who were never going to be
    dropped from the count.

    What matters is the per-person gap structure:

        rate        fraction of passes this person was found on
        max_gap     longest run of consecutive passes they were missed on
        held        whether that gap fits inside the track hold

    A person with rate 0.3 and max_gap 2 is tracked the whole time. A person
    with rate 0.3 whose misses are one long run is not, and neither is a person
    with rate 0.0 -- no threshold and no hold can rescue somebody who never
    produces a box at all.

    People are identified by their labelled SEAT (seat_truth.json); everyone not
    in a seat is pooled as "standing", which is fine because the standing people
    in this set are the easy case.
    """
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    truth = _load_truth()
    seats = json.loads((FRAMES / "seat_truth.json").read_text(encoding="utf-8"))

    # Rebuild the order _run_config walked, so the flat score lists line up.
    order: dict[int, list[tuple[str, int]]] = {}
    for name in sorted(truth):
        gt = truth[name]
        if gt.get("people") is None or not (FRAMES / f"{name}.jpg").exists():
            continue
        cid = int(name.split("_")[0])
        for i in range(len(gt["boxes"])):
            order.setdefault(cid, []).append((name, i))

    for row in results:
        if args.imgsz and row["imgsz"] != args.imgsz:
            continue
        if args.crop is not None and row["crop_assist"] != args.crop:
            continue
        print(f"\n{row['model']}  imgsz={row['imgsz']}  "
              f"crop={'on' if row['crop_assist'] else 'off'}  "
              f"floor={args.at}")
        for cam, c in row["cameras"].items():
            scores = c["person_scores"]
            seq = order.get(int(cam), [])
            if len(seq) != len(scores):
                print(f"    cam{cam}: cannot align ({len(seq)} vs {len(scores)})")
                continue
            per_person: dict[str, list[bool]] = {}
            for (name, i), score in zip(seq, scores):
                label = (seats.get(name) or [None] * (i + 1))[i] or "standing"
                found = score is not None and score >= args.at
                per_person.setdefault(label, []).append(found)

            print(f"    cam{cam}")
            for label, hits in sorted(per_person.items()):
                rate = sum(hits) / len(hits)
                gap = best = 0
                for h in hits:
                    gap = 0 if h else gap + 1
                    best = max(best, gap)
                held = "TRACKED" if (sum(hits) and best <= args.hold) else (
                    "NEVER SEEN" if not sum(hits) else f"DROPS (gap {best})")
                print(f"        {label:<10} seen {sum(hits):>2}/{len(hits):<2} "
                      f"rate {rate:.2f}  max_gap {best}  {held}")
    return 0


# ---------------------------------------------------------------------------
# recall -- by how hard each person is to see
# ---------------------------------------------------------------------------
def cmd_recall(args) -> int:
    """Per-category recall from a completed sweep.

    Three failures are reported separately because they have different fixes and
    are routinely confused:

        invisible   the person produced NO box at any confidence. Neither a
                    threshold nor a tracker can help; only more pixels or a
                    model that knows this viewpoint.
        below floor a box existed but scored under the track-creation bar. A
                    threshold CAN help, at a stated cost in false tracks.
        detected    a box cleared the bar. Anything lost after this point is a
                    tracking or association failure, not a detection one.
    """
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    truth = _load_truth()

    order = {}
    for name in sorted(truth):
        gt = truth[name]
        if gt.get("people") is None or not (FRAMES / f"{name}.jpg").exists():
            continue
        cid = int(name.split("_")[0])
        cats = gt.get("categories") or ["unlabelled"] * len(gt["boxes"])
        for i in range(len(gt["boxes"])):
            order.setdefault(cid, []).append(cats[i])

    for row in results:
        if args.imgsz and row["imgsz"] != args.imgsz:
            continue
        if args.crop is not None and row["crop_assist"] != args.crop:
            continue
        print("")
        print(f"{row['model']}  imgsz={row['imgsz']}  "
              f"crop={'on' if row['crop_assist'] else 'off'}  "
              f"floor={args.at}")
        print(f"    {'category':<22}{'people':>7}{'detected':>10}"
              f"{'below floor':>13}{'invisible':>11}{'recall':>8}")
        overall = defaultdict(lambda: [0, 0, 0, 0])
        for cam, c in row["cameras"].items():
            cats = order.get(int(cam), [])
            scores = c["person_scores"]
            if len(cats) != len(scores):
                print(f"    cam{cam}: cannot align")
                continue
            per = defaultdict(lambda: [0, 0, 0, 0])   # n, detected, below, none
            for cat, sc in zip(cats, scores):
                for bucket in (per[cat], per["ALL"], overall[cat], overall["ALL"]):
                    bucket[0] += 1
                    if sc is None:
                        bucket[3] += 1
                    elif sc >= args.at:
                        bucket[1] += 1
                    else:
                        bucket[2] += 1
            print(f"  camera {cam}")
            for cat, (n, det, low, none) in sorted(per.items()):
                print(f"    {cat:<22}{n:>7}{det:>10}{low:>13}{none:>11}"
                      f"{det / n:>8.0%}")
    return 0


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------
def cmd_render(args) -> int:
    import cv2
    from ultralytics import YOLO

    path = FRAMES / f"{args.frame}.jpg"
    if not path.exists():
        print(f"no such frame: {path}")
        return 1
    model_path = _resolve_model(args.model)
    if model_path is None:
        print(f"model not found: {args.model}")
        return 1

    frame = cv2.imread(str(path))
    model = YOLO(model_path)
    dets, added, _ = _detect_with_crop(model, frame, args.imgsz, args.floor,
                                       args.iou, args.crop)

    truth = _load_truth().get(args.frame, {})
    for box in truth.get("boxes", []):
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

    for score, (x1, y1, x2, y2) in dets:
        # Green above the threshold under test, grey below it: the point of the
        # picture is to show WHICH people the floor is throwing away.
        colour = (0, 220, 0) if score >= args.mark else (150, 150, 150)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
        cv2.putText(frame, f"{score:.3f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    out = FRAMES / (f"render_{args.frame}_{Path(args.model).stem}"
                    f"_{args.imgsz}{'_crop' if args.crop else ''}.jpg")
    cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{len(dets)} detections ({added} from crop) -> {out}")
    print("  YELLOW = labelled person, GREEN = detection above "
          f"{args.mark}, GREY = below")
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture")
    c.add_argument("--cameras", default="59,60")
    c.add_argument("--count", type=int, default=10)
    c.add_argument("--interval", type=float, default=20.0)
    c.add_argument("--drain", type=int, default=25,
                   help="frames to discard so the saved one is current")
    c.set_defaults(func=cmd_capture)

    lb = sub.add_parser("label")
    lb.set_defaults(func=cmd_label)

    pr = sub.add_parser("propose")
    pr.add_argument("--frame", default=None)
    pr.add_argument("--model", default="models/yolo11m.pt")
    pr.add_argument("--imgsz", type=int, default=960)
    pr.add_argument("--floor", type=float, default=0.005)
    pr.add_argument("--iou", type=float, default=0.5)
    pr.add_argument("--max-boxes", type=int, default=12)
    pr.set_defaults(func=cmd_propose)

    ac = sub.add_parser("accept")
    ac.add_argument("--file", default=str(FRAMES / "adjudication.json"),
                    help="per-frame {pick: [indices], add: [[x1,y1,x2,y2]]}")
    ac.set_defaults(func=cmd_accept)

    s = sub.add_parser("sweep")
    s.add_argument("--models", default="models/yolo11m.pt")
    s.add_argument("--imgsz", default="480,640,960")
    s.add_argument("--crop", choices=["on", "off", "both"], default="both")
    s.add_argument("--cameras", default="")
    s.add_argument("--floor", type=float, default=0.01,
                   help="detector conf; keep very low, thresholds are applied "
                        "arithmetically afterwards")
    s.add_argument("--iou", type=float, default=0.5)
    s.add_argument("--min-box", type=float, default=0.0,
                   help="discard detections shorter than this many pixels")
    s.set_defaults(func=cmd_sweep)

    asc = sub.add_parser("assoc")
    asc.add_argument("--rules",
                     default=",".join(SEAT_REGIONS))
    asc.add_argument("--min-overlap", type=float, default=0.15)
    asc.add_argument("--max-drop", default="",
                     help="comma-separated ceilings on how far a person's box "
                          "bottom may fall below the seat's, normalised; "
                          "empty = test without the check only")
    asc.add_argument("--width", type=float, default=960.0)
    asc.add_argument("--height", type=float, default=1080.0)
    asc.set_defaults(func=cmd_assoc)

    pp = sub.add_parser("persons")
    pp.add_argument("--at", type=float, default=0.03,
                    help="track-creation floor to evaluate at")
    pp.add_argument("--hold", type=int, default=8,
                    help="analysis passes a track survives without a detection "
                         "(ByteTrack track_buffer)")
    pp.add_argument("--imgsz", type=int, default=0)
    pp.add_argument("--crop", type=lambda v: v == "on", default=None)
    pp.set_defaults(func=cmd_persons)

    rc = sub.add_parser("recall")
    rc.add_argument("--at", type=float, default=0.02)
    rc.add_argument("--imgsz", type=int, default=0)
    rc.add_argument("--crop", type=lambda v: v == "on", default=None)
    rc.set_defaults(func=cmd_recall)

    r = sub.add_parser("render")
    r.add_argument("--frame", required=True)
    r.add_argument("--model", default="models/yolo11m.pt")
    r.add_argument("--imgsz", type=int, default=960)
    r.add_argument("--crop", action="store_true")
    r.add_argument("--floor", type=float, default=0.01)
    r.add_argument("--iou", type=float, default=0.5)
    r.add_argument("--mark", type=float, default=0.03)
    r.set_defaults(func=cmd_render)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
