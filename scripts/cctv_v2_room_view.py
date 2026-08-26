"""Draw what the room camera currently believes: seats, people, and who is where.

MANUAL / DEBUG ONLY. Writes an annotated JPEG; runs the real pipeline to make it.

WHY THIS EXISTS
---------------
Occupancy has been correct in a table for a while -- "4 chairs, 3 occupied, 1
free" -- and a table cannot tell you whether it is describing the right chairs.
The numbers only become checkable when they are drawn on the room they claim to
describe.

It also keeps the remaining gap visible rather than leaving it in a comment.
The chair map is now hand-placed and covers every seat the camera sees whole --
13 on camera 59, 5 on camera 60 -- but a person standing, walking, or in a chair
cut off by the frame edge is still drawn UNASSIGNED. That is the honest picture:
nobody is snapped to a nearby seat to make the render look tidy.

    GREEN  box    mapped seat, FREE
    RED    box    mapped seat, OCCUPIED (labelled with the occupant's track)
    BLUE   box    tracked person, sitting in a mapped seat
    YELLOW box    tracked person, in no mapped seat -- UNASSIGNED

Usage:
    python scripts/cctv_v2_room_view.py --camera 59 --seconds 60
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

GREEN, RED, BLUE, YELLOW, BLACK = ((0, 200, 0), (0, 0, 255), (255, 140, 0),
                                   (0, 215, 255), (0, 0, 0))
# Undecided -- the smoothing has not settled this seat either way yet.
GREY = (150, 150, 150)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=59)
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="how long to let occupancy settle before drawing")
    args = ap.parse_args()

    import logging
    logging.disable(logging.INFO)
    import cv2

    from app.cctv_v2.capture.grabber import CameraGrabber, FrameSnapshot
    from app.cctv_v2.config.geometry import room_geometry
    from app.cctv_v2.pipeline.detect import PersonDetector
    from app.cctv_v2.pipeline.occupancy import RoomOccupancy
    from app.cctv_v2.pipeline.track import CameraTracker
    from app.db.session import SessionLocal
    from app.models.camera import CameraConfig

    OUT.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db:
        row = db.query(CameraConfig).filter(CameraConfig.id == args.camera).first()
    if row is None or not row.source_url:
        print(f"camera {args.camera}: no source_url")
        return 1

    det = PersonDetector()
    tracker = CameraTracker(args.camera)
    occ = RoomOccupancy(args.camera)
    geom = room_geometry(args.camera)

    g = CameraGrabber(args.camera, row.source_url)
    g.start()
    print(f"settling occupancy for {args.seconds:.0f}s "
          f"({len(geom.chairs)} mapped seats)...")
    time.sleep(6.0)

    frame = snap_shot = None
    end = time.time() + args.seconds
    last_seq, seq = -1, 0
    while time.time() < end:
        s = g.get_latest()
        if s is None or s.sequence == last_seq:
            time.sleep(0.2)
            continue
        last_seq = s.sequence
        seq += 1
        # Chair zones are normalised, so occupancy has to be told the REAL frame
        # size. It defaults to the 960x1080 these cameras happen to produce, and
        # a default that is right by coincidence is a wrong answer waiting for
        # somebody to change the stream profile.
        fh, fw = s.frame.shape[:2]
        occ.frame_w, occ.frame_h = float(fw), float(fh)
        result = det.detect(args.camera, s)
        tracker.update(result)
        snapshot = occ.update(tracker.active_tracks(), s.timestamp)
        frame, snap_shot = s.frame.copy(), snapshot
        print(f"  people {snapshot.people_count:>2}   "
              f"occupied {snapshot.occupied_chairs}/{snapshot.total_chairs}")
    g.stop()

    if frame is None:
        print("no frame captured")
        return 1

    h, w = frame.shape[:2]
    canvas = frame.copy()
    by_id = {c["id"]: c for c in snap_shot.chairs}
    occupant_of = {c["occupant_track_id"]: c["id"]
                   for c in snap_shot.chairs if c["occupant_track_id"]}

    # Seats
    for zone in geom.chairs:
        st = by_id.get(zone.chair_id, {})
        # THREE states. Painting an undecided seat as FREE was a straight
        # contradiction of the totals printed underneath, which -- correctly --
        # refuse to count an undecided seat as free.
        state = st.get("state", "unknown")
        colour = {"occupied": RED, "free": GREEN}.get(state, GREY)
        x1, y1, x2, y2 = (int(zone.box[0] * w), int(zone.box[1] * h),
                          int(zone.box[2] * w), int(zone.box[3] * h))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 3)
        label = (f"{zone.chair_id} OCCUPIED trk{st.get('occupant_track_id')}"
                 if state == "occupied" else f"{zone.chair_id} {state.upper()}")
        cv2.rectangle(canvas, (x1, y1 - 22), (x1 + 9 * len(label), y1), BLACK, -1)
        cv2.putText(canvas, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    # People
    for t in tracker.active_tracks():
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        seat = occupant_of.get(t.track_id)
        colour = BLUE if seat else YELLOW
        label = f"trk{t.track_id} -> {seat}" if seat else f"trk{t.track_id} UNASSIGNED"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        cv2.rectangle(canvas, (x1, y2), (x1 + 9 * len(label), y2 + 20), BLACK, -1)
        cv2.putText(canvas, label, (x1 + 3, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)

    unassigned = snap_shot.people_count - snap_shot.occupied_chairs
    banner = [
        f"camera {args.camera}   people {snap_shot.people_count}   "
        f"chairs {snap_shot.total_chairs}   occupied {snap_shot.occupied_chairs}"
        f"   free {snap_shot.free_chairs}"
        + (f"   undecided {snap_shot.unknown_chairs}"
           if snap_shot.unknown_chairs else ""),
        f"GREEN=free  RED=occupied  GREY=undecided  BLUE=person in a seat  "
        f"YELLOW=person, no mapped seat ({max(0, unassigned)})",
        f"{snap_shot.total_chairs} seats mapped BY HAND; a YELLOW box is "
        f"standing, walking, or in a chair the camera only half sees",
    ]
    for i, line in enumerate(banner):
        y = h - 74 + i * 25
        cv2.rectangle(canvas, (0, y - 18), (w, y + 6), BLACK, -1)
        cv2.putText(canvas, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    path = OUT / f"room{args.camera}_occupancy.jpg"
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print()
    print(f"wrote {path}")
    print(f"  people {snap_shot.people_count}   chairs {snap_shot.total_chairs}"
          f"   occupied {snap_shot.occupied_chairs}   free {snap_shot.free_chairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
