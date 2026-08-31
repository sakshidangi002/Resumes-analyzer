"""Does the answer change when the room does not?

MANUAL / MEASUREMENT ONLY. Needs the live cameras or a saved frame plus the YOLO
weights, so it is deliberately not importable by the automated suite.

THE QUESTION
------------
Occupancy was observed flipping between FREE and OCCUPIED, and swapping between
neighbouring chairs, while nobody in the room moved. "Sometimes right" is not a
tuning problem to be smoothed away -- it means some input to the decision is
changing when the scene is not, and the first job is to find WHICH.

So this measures the pipeline one stage at a time and reports where the variance
enters:

    poll     does the SAME observation, read repeatedly, change the answer?
    static   does the SAME PIXELS, re-detected, give the same tracks and claims?
    live     over minutes on a real camera, how often does a chair change state
             while the track set is identical?

The three separate deliberately, because they have different fixes. Variance in
`poll` is a state-machine bug and no detector change can help it. Variance in
`static` is the detector or the tracker. Variance in `live` that does NOT show up
in `static` is the scene genuinely changing, or box jitter crossing a boundary.

WHAT COUNTS AS A FALSE CHANGE
-----------------------------
A chair changing state while the set of track ids AND their boxes are unchanged
from the previous pass. That is unambiguous: same input, different output.

A change while the boxes moved slightly is reported separately, because it is a
tolerance question rather than a bug -- see the jitter and margin sections.

Usage:
    python scripts/cctv_v2_stability.py poll
    python scripts/cctv_v2_stability.py static --frame data/cctv_v2_rooms/59_00005.jpg --passes 20
    python scripts/cctv_v2_stability.py live --camera 59 --minutes 5
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Attendance Management" / "backend"
for p in (str(BACKEND), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = REPO / "data" / "cctv_v2_stability"


# ---------------------------------------------------------------------------
# poll -- is the state machine driven by evidence or by reads?
# ---------------------------------------------------------------------------
def cmd_poll(args) -> int:
    """Read one unchanging observation repeatedly and watch the state move.

    This needs no camera and no model: it drives the API path directly with a
    fake V1 worker whose track list is frozen. If the state changes, nothing
    about detection can be to blame.
    """
    from app.cctv_v2.pipeline import v1_bridge

    class _Lock:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Track:
        def __init__(self, tid, box):
            self.track_id, self.box = tid, box
            self.confidence, self.last_seen = 0.5, time.time()

    class _State:
        """V1 stamps `updated_at` once per completed analysis pass, under the
        same lock that publishes the tracks. That stamp is what identifies an
        observation."""
        def __init__(self): self.updated_at = 1000.0

    class _Worker:
        def __init__(self, tracks):
            self._latest_tracks = tracks
            self._latest_frame = None
            self._frame_lock = _Lock()
            self.state = _State()

    from app.cctv_v2.config.geometry import room_geometry
    seat = room_geometry(args.camera).chairs[args.seat]
    x1, y1, x2, y2 = seat.box
    box = (x1 * 960, y1 * 1080, x2 * 960, y2 * 1080)

    worker = _Worker([_Track(1, box)])
    v1_bridge._v1_worker = lambda cid: worker
    v1_bridge._registry = type(v1_bridge._registry)()
    v1_bridge._last_observed.clear()
    v1_bridge._last_payload.clear()

    def read(label):
        snap = v1_bridge.occupancy_snapshot(args.camera)
        st = next(c for c in snap["chairs"]
                  if c["id"] == seat.chair_id)["state"]
        fresh = "new" if snap.get("from_new_observation") else "cached"
        print(f"  {label:<28}{st:<12}{fresh}")
        return st

    print(f"camera {args.camera}, seat {seat.chair_id}\n")

    # -- settle, one observation per pass -----------------------------------
    print("A. the person sits down; each line below is a NEW analysis pass")
    states = []
    for i in range(3):
        worker.state.updated_at += 1.0
        states.append(read(f"pass {i + 1}"))

    # -- the reported symptom -----------------------------------------------
    print(f"\nB. the dashboard polls {args.reads} times. NO new analysis pass "
          f"has completed,\n   so every one of these is a re-read of pass 3:")
    frozen = []
    for i in range(args.reads):
        frozen.append(read(f"poll {i + 1}"))
        time.sleep(args.interval)

    # -- and a real departure still works -----------------------------------
    print("\nC. the person really leaves. New passes now, and the seat must "
          "hold\n   OCCUPIED until confirm_free of them agree:")
    worker._latest_tracks = []
    leaving = []
    for i in range(7):
        worker.state.updated_at += 1.0
        leaving.append(read(f"pass {i + 4}"))

    frozen_changes = sum(1 for a, b in zip(frozen, frozen[1:]) if a != b)
    print(f"\n{frozen_changes} state change(s) across {args.reads} polls of an "
          f"observation that never changed.")
    print("   (must be 0 -- a poll is not evidence)")
    if "free" in leaving:
        print(f"seat went FREE after {leaving.index('free') + 1} new passes "
              f"showing nobody")
    else:
        print("seat never went FREE -- confirm_free may be too high")
    return 0 if frozen_changes == 0 else 1


# ---------------------------------------------------------------------------
# shared: run the real pipeline and record what each stage produced
# ---------------------------------------------------------------------------
def _claims(room, tracks):
    """Per chair, every track's coverage of it -- not just the winner.

    The margin between the best and second-best claim is the quantity that
    decides whether box jitter can swap two neighbouring seats, and it is
    invisible if only the winner is recorded.
    """
    from app.cctv_v2.pipeline.occupancy import (
        _overlap_fraction_of_seat, _seat_region)

    out = {}
    for track in tracks:
        region = _seat_region(track, room.frame_w, room.frame_h)
        scored = sorted(
            ((_overlap_fraction_of_seat(region, s.box), s.chair_id)
             for s in room.geometry.chairs),
            reverse=True,
        )
        out[track.track_id] = scored[:3]
    return out


def _record(passes: list, path: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(passes, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


def _analyse(passes: list, elapsed_min: float) -> None:
    """Report state changes, and separate the ones with no input change."""
    print("\n" + "=" * 74)
    print("STATE CHANGES")
    print("=" * 74)

    per_chair = defaultdict(lambda: {"changes": 0, "false": 0})
    for prev, cur in zip(passes, passes[1:]):
        same_input = (prev["tracks"] == cur["tracks"])
        for chair_id, state in cur["states"].items():
            if prev["states"].get(chair_id) != state:
                per_chair[chair_id]["changes"] += 1
                if same_input:
                    per_chair[chair_id]["false"] += 1

    total = sum(v["changes"] for v in per_chair.values())
    false = sum(v["false"] for v in per_chair.values())
    if not per_chair:
        print("no chair changed state at all")
    else:
        print(f"{'chair':<8}{'changes':>9}{'false':>7}{'per min':>10}")
        for chair_id, v in sorted(per_chair.items(),
                                  key=lambda kv: -kv[1]["changes"]):
            rate = v["changes"] / elapsed_min if elapsed_min else 0.0
            print(f"{chair_id:<8}{v['changes']:>9}{v['false']:>7}{rate:>10.2f}")
    print(f"\ntotal {total} change(s), {false} of them with an IDENTICAL track "
          f"set ({total / elapsed_min if elapsed_min else 0:.2f}/min)")

    # -- B. track id churn ---------------------------------------------------
    ids_per_pass = [set(p["tracks"].keys()) for p in passes]
    churn = sum(len(a ^ b) for a, b in zip(ids_per_pass, ids_per_pass[1:]))
    seen = set().union(*ids_per_pass) if ids_per_pass else set()
    print(f"\nTRACK IDS: {len(seen)} distinct ids over {len(passes)} passes, "
          f"{churn} appearances/disappearances")
    if len(seen) > 4:
        print("  id churn is high -- a person changing id changes which chair "
              "records them as occupant")

    # -- C. box jitter -------------------------------------------------------
    moves = defaultdict(list)
    for prev, cur in zip(passes, passes[1:]):
        for tid, box in cur["tracks"].items():
            if tid in prev["tracks"]:
                a, b = prev["tracks"][tid], box
                dx = ((a[0] + a[2]) - (b[0] + b[2])) / 2.0
                dy = ((a[1] + a[3]) - (b[1] + b[3])) / 2.0
                moves[tid].append((dx ** 2 + dy ** 2) ** 0.5)
    if moves:
        allm = [m for v in moves.values() for m in v]
        print(f"\nBOX JITTER: centre moves between passes, px  "
              f"median {statistics.median(allm):.0f}  "
              f"p90 {sorted(allm)[int(len(allm) * 0.9)]:.0f}  "
              f"max {max(allm):.0f}")

    # -- D. how close are the top two chairs? --------------------------------
    margins = []
    for p in passes:
        for tid, scored in p["claims"].items():
            if len(scored) >= 2 and scored[0][0] > 0:
                margins.append(scored[0][0] - scored[1][0])
    if margins:
        margins.sort()
        tight = sum(1 for m in margins if m < 0.05)
        print(f"\nCHAIR AMBIGUITY: best-vs-second claim margin  "
              f"median {statistics.median(margins):.3f}  "
              f"min {margins[0]:.3f}")
        print(f"  {tight}/{len(margins)} claims decided by less than 0.05 -- "
              f"those are the ones box jitter can swap")


def _states_of(snap) -> dict:
    return {c["id"]: c["state"] for c in snap.chairs}


# ---------------------------------------------------------------------------
# static -- same pixels, over and over
# ---------------------------------------------------------------------------
def cmd_static(args) -> int:
    """Re-run detection on ONE saved frame. The scene cannot change."""
    import logging
    logging.disable(logging.INFO)
    import cv2

    from app.cctv_v2.capture.grabber import FrameSnapshot
    from app.cctv_v2.pipeline.detect import PersonDetector
    from app.cctv_v2.pipeline.occupancy import RoomOccupancy
    from app.cctv_v2.pipeline.track import CameraTracker

    frame = cv2.imread(args.frame)
    if frame is None:
        print(f"could not read {args.frame}")
        return 1
    camera = int(Path(args.frame).stem.split("_")[0])
    h, w = frame.shape[:2]

    det = PersonDetector()
    tracker = CameraTracker(camera)
    room = RoomOccupancy(camera, (float(w), float(h)))

    print(f"camera {camera}, {args.passes} passes over the SAME frame "
          f"({Path(args.frame).name}, {w}x{h})\n")
    print(f"{'pass':>5}{'people':>8}{'occupied':>10}   states")
    passes = []
    for i in range(args.passes):
        snapshot = FrameSnapshot(camera_id=camera, frame=frame,
                                 timestamp=time.time(), sequence=i)
        result = det.detect(camera, snapshot)
        tracker.update(result)
        live = tracker.active_tracks()
        snap = room.update(live, float(i))
        passes.append({
            "pass": i,
            "tracks": {str(t.track_id): [round(v, 1) for v in t.bbox]
                       for t in live},
            "claims": {str(k): v for k, v in _claims(room, live).items()},
            "states": _states_of(snap),
        })
        occ = "".join("O" if c["occupied"] else "." for c in snap.chairs)
        print(f"{i:>5}{snap.people_count:>8}{snap.occupied_chairs:>10}   {occ}")

    _record(passes, OUT / f"static{camera}.json")
    _analyse(passes, elapsed_min=max(1e-9, args.passes / 60.0))
    print("\nNOTE: the frame is identical every pass, so ANY change above is "
          "internal -- detector nondeterminism, tracker state, or the smoothing.")
    return 0


# ---------------------------------------------------------------------------
# live -- a real camera, for minutes
# ---------------------------------------------------------------------------
def cmd_live(args) -> int:
    import logging
    logging.disable(logging.INFO)

    from app.cctv_v2.capture.grabber import CameraGrabber
    from app.cctv_v2.pipeline.detect import PersonDetector
    from app.cctv_v2.pipeline.occupancy import RoomOccupancy
    from app.cctv_v2.pipeline.track import CameraTracker
    from app.db.session import SessionLocal
    from app.models.camera import CameraConfig

    with SessionLocal() as db:
        row = db.query(CameraConfig).filter(
            CameraConfig.id == args.camera).first()
    if row is None or not row.source_url:
        print(f"camera {args.camera}: no source_url")
        return 1

    det = PersonDetector()
    tracker = CameraTracker(args.camera)
    room = RoomOccupancy(args.camera)

    g = CameraGrabber(args.camera, row.source_url)
    g.start()
    time.sleep(6.0)
    print(f"camera {args.camera}: {args.minutes} minutes. Leave the room alone.\n")
    print(f"{'t+s':>6}{'people':>8}{'occupied':>10}   states")

    passes = []
    last_seq = -1
    started = time.time()
    end = started + args.minutes * 60.0
    while time.time() < end:
        s = g.get_latest()
        if s is None or s.sequence == last_seq:
            time.sleep(0.2)
            continue
        last_seq = s.sequence
        fh, fw = s.frame.shape[:2]
        room.frame_w, room.frame_h = float(fw), float(fh)
        result = det.detect(args.camera, s)
        tracker.update(result)
        live = tracker.active_tracks()
        snap = room.update(live, s.timestamp)
        passes.append({
            "t": round(time.time() - started, 1),
            "tracks": {str(t.track_id): [round(v, 1) for v in t.bbox]
                       for t in live},
            "claims": {str(k): v for k, v in _claims(room, live).items()},
            "states": _states_of(snap),
        })
        occ = "".join("O" if c["occupied"] else "." for c in snap.chairs)
        print(f"{time.time() - started:>6.0f}{snap.people_count:>8}"
              f"{snap.occupied_chairs:>10}   {occ}")
    g.stop()

    elapsed_min = (time.time() - started) / 60.0
    _record(passes, OUT / f"live{args.camera}.json")
    print(f"\n{len(passes)} passes in {elapsed_min:.1f} min "
          f"= one every {elapsed_min * 60 / max(1, len(passes)):.1f}s")
    _analyse(passes, elapsed_min)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("poll")
    p.add_argument("--camera", type=int, default=59)
    p.add_argument("--seat", type=int, default=7, help="index into the chair map")
    p.add_argument("--reads", type=int, default=8)
    p.add_argument("--interval", type=float, default=0.15)
    p.set_defaults(func=cmd_poll)

    s = sub.add_parser("static")
    s.add_argument("--frame", required=True)
    s.add_argument("--passes", type=int, default=20)
    s.set_defaults(func=cmd_static)

    l = sub.add_parser("live")
    l.add_argument("--camera", type=int, default=59)
    l.add_argument("--minutes", type=float, default=5.0)
    l.set_defaults(func=cmd_live)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
