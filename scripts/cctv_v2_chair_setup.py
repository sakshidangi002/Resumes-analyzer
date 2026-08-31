"""Help a human place chair zones for a room camera.

WHY A HUMAN
-----------
COCO chair detection was measured against these two rooms and cannot enumerate
the seats: camera 59 returned between 1 and 9 chairs depending on input size and
confidence, against roughly 10 visible; camera 60 returned 3 to 10 against 6-8.
Auto-generating a chair map from numbers that move like that would produce a
configuration that looks authoritative and is wrong, and every occupancy figure
built on it would inherit the error silently.

So this tool does not decide anything. It renders the camera's view with a
labelled coordinate grid and the detector's chair guesses drawn on top, so a
person can read normalised coordinates off the picture and write them into
`config/geometry.py`. The detector's opinion is shown as a hint, clearly marked
as unreliable, not as an answer.

    python scripts/cctv_v2_chair_setup.py --camera 59
    -> data/cctv_v2_geometry/chairs59_setup.jpg

Read the grid, then add to ROOM_GEOMETRY in config/geometry.py:

    59: RoomGeometry(chairs=(
            ChairZone("C1", (0.18, 0.24, 0.30, 0.41)),
            ChairZone("C2", (0.30, 0.30, 0.42, 0.48)),
        )),

A zone should cover the SEAT AND THE SPACE A SEATED PERSON OCCUPIES, not the
chair furniture alone -- occupancy is decided by how much of the zone a person's
lower body covers.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Attendance Management" / "backend"
for p in (str(BACKEND), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = REPO / "data" / "cctv_v2_geometry"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--grid", type=int, default=20)
    # Placing zones against a SAVED frame matters as much as against the live
    # one: the benchmark set in data/cctv_v2_rooms is what every occupancy
    # measurement is scored on, so the map has to be checkable against those
    # exact pixels. Grabbing a fresh frame instead would put the zones on a
    # picture no measurement ever used.
    ap.add_argument("--frame", default=None,
                    help="path to a saved frame; omit to grab one live")
    # The COCO hint costs a YOLO pass. When a benchmark is running on the same
    # box that pass competes for the cores whose timings are being measured, and
    # the hint is the least valuable thing on the picture anyway -- it is the
    # detector that was already shown not to enumerate these seats.
    ap.add_argument("--no-hints", action="store_true",
                    help="skip the COCO chair pass (no model load, no CPU)")
    # Placing ten seats is not one edit, it is a loop: draw, look, nudge, look
    # again. Running that loop through geometry.py would mean editing shipped
    # configuration a dozen times and re-importing between each, so a candidate
    # map can live in a scratch JSON until it is right.
    ap.add_argument("--zones", default=None,
                    help='JSON of candidate zones {"C1": [x1,y1,x2,y2], ...}, '
                         "drawn in CYAN alongside the configured ones")
    args = ap.parse_args()

    import logging
    logging.disable(logging.INFO)
    import cv2

    from app.cctv_v2.capture.grabber import CameraGrabber
    from app.cctv_v2.config.geometry import room_geometry
    from app.cctv_v2.pipeline.detect import detect_chairs
    from app.db.session import SessionLocal
    from app.models.camera import CameraConfig
    if not args.no_hints:
        from ultralytics import YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    if args.frame:
        frame = cv2.imread(args.frame)
        if frame is None:
            print(f"could not read frame: {args.frame}")
            return 1
        print(f"using saved frame {args.frame}")
    else:
        with SessionLocal() as db:
            row = db.query(CameraConfig).filter(
                CameraConfig.id == args.camera).first()
        if row is None or not row.source_url:
            print(f"camera {args.camera} has no source_url")
            return 1

        g = CameraGrabber(args.camera, row.source_url)
        g.start()
        frame = None
        for _ in range(8):
            time.sleep(4.0)
            snap = g.get_latest()
            if snap is not None:
                frame = snap.frame
                break
        g.stop()
        if frame is None:
            print("no frame captured")
            return 1

    h, w = frame.shape[:2]
    canvas = frame.copy()

    # Grid, so coordinates can be read straight off the picture.
    # Every line is drawn, but only every other one is LABELLED once the grid is
    # fine -- at 0.05 spacing the labels overlap each other and the picture
    # becomes harder to read off than a coarser grid would have been.
    label_every = 2 if args.grid > 10 else 1
    for i in range(1, args.grid):
        f = i / args.grid
        x, y = int(f * w), int(f * h)
        major = (i % label_every == 0)
        shade = (110, 110, 110) if major else (70, 70, 70)
        cv2.line(canvas, (x, 0), (x, h), shade, 1)
        cv2.line(canvas, (0, y), (w, y), shade, 1)
        if major:
            cv2.putText(canvas, f"{f:.2f}", (x + 3, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1,
                        cv2.LINE_AA)
            cv2.putText(canvas, f"{f:.2f}", (3, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1,
                        cv2.LINE_AA)

    if args.no_hints:
        guesses = ()
    else:
        model = YOLO(str(BACKEND / "models" / "yolo11m.pt"))
        guesses = detect_chairs(model, frame, args.imgsz, args.conf, args.camera)
    for (x1, y1, x2, y2) in guesses:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 140, 255), 2)

    # Anything already configured, so an edit can be checked against the view.
    #
    # NUMBERED, and numbered at the BOTTOM of each zone. Two reasons, both
    # learned by getting this wrong: a reviewer counting chairs needs to say
    # "there is one more after number 8" without ambiguity, and the bottom of a
    # zone sits near the chair's five-star base, which is the part of a chair
    # that stays visible when somebody is sitting on it.
    existing = room_geometry(args.camera).chairs
    for i, c in enumerate(existing, start=1):
        x1, y1, x2, y2 = c.box
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        cv2.rectangle(canvas, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cx, cy = (px1 + px2) // 2, min(h - 14, py2 - 12)
        cv2.circle(canvas, (cx, cy), 15, (0, 0, 0), -1)
        cv2.circle(canvas, (cx, cy), 15, (0, 255, 0), 2)
        label = str(i)
        cv2.putText(canvas, label, (cx - 7 * len(label), cy + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, c.chair_id, (px1 + 4, py1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    candidates = {}
    if args.zones:
        import json
        candidates = {k: v for k, v in
                      json.loads(Path(args.zones).read_text(encoding="utf-8")).items()
                      if not k.startswith("_")}
        for chair_id, box in candidates.items():
            x1, y1, x2, y2 = box
            cv2.rectangle(canvas, (int(x1 * w), int(y1 * h)),
                          (int(x2 * w), int(y2 * h)), (255, 220, 0), 2)
            cv2.putText(canvas, chair_id, (int(x1 * w) + 4, int(y1 * h) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2,
                        cv2.LINE_AA)

    # Kept short enough to FIT. An earlier version listed every id and ran off
    # the right edge, so the reviewer could not read the part that mattered.
    banner = [
        f"camera {args.camera}   {w}x{h}   grid = normalised 0..1",
        f"GREEN = {len(existing)} mapped chairs, numbered 1..{len(existing)} "
        f"along the row"
        + (f"   CYAN = {len(candidates)} candidates" if candidates else "")
        + (f"   ORANGE = {len(guesses)} COCO guesses" if guesses else ""),
        "Missing one? Say where: 'one more after 8', 'one before 1'.",
    ]
    for i, line in enumerate(banner):
        cv2.rectangle(canvas, (0, h - 78 + i * 26), (w, h - 52 + i * 26), (0, 0, 0), -1)
        cv2.putText(canvas, line, (8, h - 58 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    path = OUT / f"chairs{args.camera}_setup.jpg"
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {path}")
    print(f"  COCO chair guesses: {len(guesses)} (orange) -- treat as a hint")
    print(f"  already configured: {len(existing)} (green)")
    print()
    print("Read zones off the grid and add them to ROOM_GEOMETRY in")
    print("  Attendance Management/backend/app/cctv_v2/config/geometry.py")
    print("Cover the seat AND the space a seated person occupies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
