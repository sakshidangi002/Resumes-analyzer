"""Where the seconds go between a person leaving a chair and the dashboard saying so.

MANUAL / MEASUREMENT ONLY.

WHY THIS EXISTS
---------------
Occupancy is stable now but slow: a chair keeps saying OCCUPIED long after its
occupant has walked away. "Slow" has at least five candidate causes in this
pipeline and they are not equally to blame, so the first job is to apportion the
delay rather than to start lowering thresholds.

    camera frame  -> analysis pass      how often V1 even looks
    analysis pass -> observation        the motion gate and the coast
    observation   -> state change       confirm_occupied / confirm_free
    state change  -> API                nothing; the bridge caches
    API           -> dashboard          the 2s poll

The total is a PRODUCT, not a sum: a state change needs N observations, and the
observation interval is itself gated. So the interesting quantity is not the
confirmation count on its own but

    latency  ~=  confirmations  x  observation interval  +  poll

THE ONE THING THAT IS EASY TO MISS
----------------------------------
V1 skips analysis entirely on a MONITOR camera whose picture has not changed:

    if static and w.is_monitor:
        if time.time() - self._last_analysed_ts < _MONITOR_COAST_SEC:
            continue                      # no inference, and NO new observation

That is a good rule -- it stops two rooms burning the single inference slot on
furniture. But it fires exactly when occupancy most needs to move: the moment
somebody leaves, the room becomes still, so the observations that would confirm
the seat FREE are the ones the coast suppresses.

So this measures the motion signal V1 actually uses, on the real cameras, and
replays V1's gate over it. No YOLO: the question is how often an observation
HAPPENS, which is decided before any inference runs.

    motion    sample the cameras and report V1's static/moving verdict
    budget    turn that into a projected latency, per direction

Usage:
    python scripts/cctv_v2_latency.py motion --camera 59 --minutes 4
    python scripts/cctv_v2_latency.py budget
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Attendance Management" / "backend"
for p in (str(BACKEND), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = REPO / "data" / "cctv_v2_latency"


def _v1_settings() -> dict:
    """The numbers V1 is really running with, read from its own modules."""
    from app.core.config import get_settings
    from app.services import camera_service as cs

    s = get_settings()
    return {
        "motion_threshold": cs._MOTION_THRESHOLD,
        "monitor_coast_sec": cs._MONITOR_COAST_SEC,
        "monitor_interval_sec": float(s.monitor_analysis_interval),
        "monitor_imgsz": int(s.yolo_monitor_imgsz or s.yolo_person_imgsz),
    }


# ---------------------------------------------------------------------------
# motion -- the gate that decides whether an observation happens at all
# ---------------------------------------------------------------------------
def cmd_motion(args) -> int:
    """Sample a camera and apply V1's static test to consecutive samples.

    Deliberately mirrors V1: greyscale, mean absolute difference against the
    PREVIOUS analysis iteration's frame, compared with _MOTION_THRESHOLD. The
    sampling interval is V1's monitor analysis interval, so the verdicts line up
    with the ones V1 would reach.
    """
    import cv2
    import numpy as np

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    from app.db.session import SessionLocal
    from app.models.camera import CameraConfig

    cfg = _v1_settings()
    with SessionLocal() as db:
        row = db.query(CameraConfig).filter(
            CameraConfig.id == args.camera).first()
    if row is None or not row.source_url:
        print(f"camera {args.camera}: no source_url")
        return 1

    cap = cv2.VideoCapture(row.source_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"camera {args.camera}: could not open stream")
        return 1

    interval = args.interval or cfg["monitor_interval_sec"]
    print(f"camera {args.camera}: sampling every {interval:.1f}s for "
          f"{args.minutes:.0f} min")
    print(f"V1 static test: mean |frame - prev| < {cfg['motion_threshold']}")
    print(f"V1 coast:       skip analysis while static, up to "
          f"{cfg['monitor_coast_sec']:.0f}s\n")
    print(f"{'t+s':>6}{'motion':>9}   verdict      would V1 analyse?")

    prev = None
    samples = []
    # Replays V1's coast state so the "would analyse" column is V1's decision,
    # not merely the static verdict.
    last_analysed = 0.0
    started = time.time()
    end = started + args.minutes * 60.0
    while time.time() < end:
        for _ in range(args.drain):
            cap.grab()
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(interval)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now = time.time()
        if prev is not None and prev.shape == gray.shape:
            motion = float(np.mean(cv2.absdiff(gray, prev)))
            static = motion < cfg["motion_threshold"]
            coasted = now - last_analysed
            analyse = (not static) or coasted >= cfg["monitor_coast_sec"]
            if analyse:
                last_analysed = now
            samples.append({"t": round(now - started, 1),
                            "motion": round(motion, 3),
                            "static": static, "analysed": analyse})
            print(f"{now - started:>6.0f}{motion:>9.3f}   "
                  f"{'STATIC ' if static else 'moving ':<12} "
                  f"{'YES' if analyse else 'no  (coasting)'}")
        prev = gray
        time.sleep(interval)
    cap.release()

    if not samples:
        print("no samples")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"motion{args.camera}.json"
    path.write_text(json.dumps({"settings": cfg, "samples": samples}, indent=2),
                    encoding="utf-8")

    n = len(samples)
    n_static = sum(1 for s in samples if s["static"])
    analysed = [s for s in samples if s["analysed"]]
    gaps = [b["t"] - a["t"] for a, b in zip(analysed, analysed[1:])]
    print("\n" + "=" * 66)
    print(f"{n} samples over {(time.time() - started) / 60:.1f} min")
    print(f"  judged STATIC     {n_static}/{n}  ({n_static / n:.0%})")
    print(f"  V1 would analyse  {len(analysed)}/{n}")
    if gaps:
        print(f"  OBSERVATION INTERVAL  median {statistics.median(gaps):.1f}s"
              f"   max {max(gaps):.1f}s")
        print(f"\n  An occupancy state change needs N of those in a row, so at")
        print(f"  the median interval the latency floor is:")
        med = statistics.median(gaps)
        for name, n_conf in (("confirm_occupied", 2), ("confirm_free", 5)):
            print(f"      {name:<18} {n_conf} x {med:.1f}s = {n_conf * med:>5.1f}s")
    print(f"\nwrote {path}")
    return 0


# ---------------------------------------------------------------------------
# budget -- put the whole chain on one page
# ---------------------------------------------------------------------------
def cmd_budget(args) -> int:
    from app.cctv_v2.config.geometry import RoomGeometry

    cfg = _v1_settings()
    g = RoomGeometry()
    obs = args.observation_interval

    print("OCCUPANCY LATENCY BUDGET")
    print("=" * 66)
    print(f"observation interval assumed: {obs:.1f}s "
          f"(measure it with `motion`)\n")
    print(f"{'stage':<44}{'occupied->free':>12}{'free->occupied':>16}")
    print("-" * 72)
    rows = [
        ("camera frame -> analysis pass (frame age)", 0.1, 0.1),
        ("analysis pass -> observation (motion gate)",
         0.0 if args.moving else cfg["monitor_coast_sec"] - obs, 0.0),
        ("observation -> state change",
         g.confirm_free * obs, g.confirm_occupied * obs),
        ("state change -> API (cached, no work)", 0.0, 0.0),
        ("API -> dashboard (2s poll, average)", 1.0, 1.0),
    ]
    for name, a, b in rows:
        print(f"{name:<44}{a:>11.1f}s{b:>15.1f}s")
    print("-" * 72)
    print(f"{'TOTAL':<44}{sum(r[1] for r in rows):>11.1f}s"
          f"{sum(r[2] for r in rows):>15.1f}s")
    print()
    print("The dominant term is confirmations x observation interval. Both")
    print("halves matter: halving either halves the latency.")
    return 0


# ---------------------------------------------------------------------------
# simulate -- how many observations each transition costs, end to end
# ---------------------------------------------------------------------------
def cmd_simulate(args) -> int:
    """Drive the REAL state machine through the real API path and count.

    Measured in OBSERVATIONS, which is exact and repeatable, then converted to
    seconds using the interval measured from production. Staging a person to sit
    and stand on cue would give one noisy sample of the same number; this gives
    the number itself, and the interval is what varies.

    The path exercised is the one the dashboard uses -- v1_bridge, cache and
    all -- so anything the caching gets wrong shows up here.
    """
    from app.cctv_v2.config.geometry import RoomGeometry, room_geometry
    from app.cctv_v2.pipeline import v1_bridge
    from app.cctv_v2.pipeline.occupancy import OccupancyRegistry

    class _Lock:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _State:
        def __init__(self): self.updated_at = 1000.0

    class _T:
        def __init__(self, tid, box):
            self.track_id, self.box = tid, box
            self.confidence, self.last_seen = 0.5, 1000.0

    class _W:
        def __init__(self):
            self._latest_tracks, self._latest_frame = [], None
            self._frame_lock, self.state = _Lock(), _State()

        def pass_with(self, tracks):
            self._latest_tracks = list(tracks)
            self.state.updated_at += 1.0

    cam = args.camera
    seat = room_geometry(cam).chairs[args.seat]
    box = tuple(v * s for v, s in zip(seat.box, (960, 1080, 960, 1080)))

    w = _W()
    v1_bridge._v1_worker = lambda cid: w
    v1_bridge._registry = OccupancyRegistry()
    v1_bridge._last_observed.clear()
    v1_bridge._last_payload.clear()

    def state():
        s = v1_bridge.occupancy_snapshot(cam)
        return next(c for c in s["chairs"] if c["id"] == seat.chair_id)["state"]

    def run_until(target, tracks, cap=40):
        """Observations needed to reach `target`. Polls MORE than once per pass
        on purpose -- extra polls must not count, and if they ever do this
        number silently shrinks."""
        for i in range(1, cap + 1):
            w.pass_with(tracks)
            for _ in range(args.polls_per_pass):
                v1_bridge.occupancy_snapshot(cam)
            if state() == target:
                return i
        return None

    g = RoomGeometry()
    print(f"camera {cam}, seat {seat.chair_id}")
    print(f"confirm_occupied={g.confirm_occupied}  confirm_free={g.confirm_free}")
    print(f"polling {args.polls_per_pass}x per pass (must not count)\n")

    sit = run_until("occupied", [_T(1, box)])
    leave = run_until("free", [])
    sit2 = run_until("occupied", [_T(2, box)])
    leave2 = run_until("free", [])

    iv = args.interval
    print(f"{'transition':<34}{'observations':>14}{'seconds @' + f'{iv}s':>16}")
    print("-" * 64)
    for name, n in (("free -> occupied  (person sits)", sit),
                    ("occupied -> free  (person leaves)", leave),
                    ("free -> occupied  (repeat)", sit2),
                    ("occupied -> free  (repeat)", leave2)):
        secs = "never" if n is None else f"{n * iv:.1f}s"
        print(f"{name:<34}{str(n):>14}{secs:>16}")
    print()
    print("Add the tracker's own hold before this: V1 keeps a lost interior")
    print("track for up to CCTV_TRACK_HOLD_MAX_SEC (10s) before dropping it, so")
    print("the seat does not even lose its claim until then.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("motion")
    m.add_argument("--camera", type=int, default=59)
    m.add_argument("--minutes", type=float, default=4.0)
    m.add_argument("--interval", type=float, default=0.0,
                   help="0 = use V1's monitor analysis interval")
    m.add_argument("--drain", type=int, default=8)
    m.set_defaults(func=cmd_motion)

    b = sub.add_parser("budget")
    b.add_argument("--observation-interval", type=float, default=11.0)
    b.add_argument("--moving", action="store_true",
                   help="assume the room is never judged static")
    b.set_defaults(func=cmd_budget)

    sim = sub.add_parser("simulate")
    sim.add_argument("--camera", type=int, default=59)
    sim.add_argument("--seat", type=int, default=-1)
    sim.add_argument("--interval", type=float, default=8.1,
                     help="measured observation interval, seconds")
    sim.add_argument("--polls-per-pass", type=int, default=6)
    sim.set_defaults(func=cmd_simulate)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
