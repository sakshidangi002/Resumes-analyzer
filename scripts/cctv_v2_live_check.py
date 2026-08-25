"""Live RTSP validation for the CCTV V2 grabber + scheduler.

MANUAL / INTEGRATION ONLY. This needs four real cameras and is deliberately not
importable by the automated suite -- `pytest` must stay runnable on a machine
with no DVR.

WHAT IT PROVES (and what it does not)
-------------------------------------
It exercises exactly:

    RTSP -> grabber -> latest-frame slot -> scheduler -> stub callback

There is no YOLO, no tracking, no recognition and no attendance. The stub sleeps
for the MEASURED cost of a real inference pass (doorway 3.4s at 640, room 2.2s
at 480) because without that the scheduler would spin thousands of times a
second and any fairness result would be meaningless -- fairness only becomes a
question when the worker is scarce.

Two metrics are collected and must not be conflated:

  * frame age    = selection_time - frame_timestamp.
                   How out of date the picture was. Answers "is RTSP feeding us
                   fresh frames?"
  * service wait = time between one camera's consecutive selections.
                   How long a camera queued. Answers "is the scheduler fair?"

V1 reported neither, which is why a 0.12s target could sit in the config while
the real cadence was 4.5-9s.

Usage:
    python scripts/cctv_v2_live_check.py --seconds 240
    python scripts/cctv_v2_live_check.py --seconds 60 --fault 59
"""
from __future__ import annotations

import argparse
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

# Measured YOLO cost per role on this hardware. The stub burns this so the
# scheduler faces realistic scarcity.
INFERENCE_COST = {"doorway": 3.4, "room": 2.2}


def _pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--fault", type=int, default=None,
                    help="camera id to kill mid-run, to prove fault isolation")
    args = ap.parse_args()

    import logging
    logging.disable(logging.INFO)

    from app.cctv_v2.capture.grabber import CameraGrabber
    from app.cctv_v2.scheduler.loop import InferenceScheduler
    from app.cctv_v2.config.cameras import CAMERA_ROLE, role_for
    from app.db.session import SessionLocal
    from app.models.camera import CameraConfig

    with SessionLocal() as db:
        rows = {c.id: c.source_url for c in db.query(CameraConfig).all()}

    grabbers = {}
    for cid in sorted(CAMERA_ROLE):
        url = rows.get(cid)
        if not url:
            print(f"  camera {cid}: no source_url in DB, skipping")
            continue
        grabbers[cid] = CameraGrabber(cid, url)

    # Per-selection records, for the two distinct metrics.
    frame_ages: dict[int, list[float]] = defaultdict(list)
    service_gaps: dict[int, list[float]] = defaultdict(list)
    last_selected: dict[int, float] = {}
    consumed: dict[int, int] = defaultdict(int)

    def process(camera_id, snap):
        now = time.time()
        frame_ages[camera_id].append(now - snap.timestamp)
        if camera_id in last_selected:
            service_gaps[camera_id].append(now - last_selected[camera_id])
        last_selected[camera_id] = now
        consumed[camera_id] += 1
        # Stand in for the real inference pass.
        time.sleep(INFERENCE_COST[role_for(camera_id)])

    sched = InferenceScheduler(grabbers.values(), process=process)

    print(f"starting {len(grabbers)} grabbers + 1 scheduler for {args.seconds:.0f}s")
    print("  (no YOLO; the stub sleeps for the measured inference cost)")
    for g in grabbers.values():
        g.start()
    time.sleep(3.0)                      # let streams come up
    sched._started_at = time.time()
    sched.start()

    faulted = False
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        time.sleep(2.0)
        if args.fault and not faulted and time.time() - t0 > args.seconds * 0.4:
            print(f"\n  >>> killing camera {args.fault} to test isolation\n")
            grabbers[args.fault].stop()
            faulted = True

    # Snapshot health BEFORE stopping: health() reports live connection state,
    # and reading it after stop() would always show connected=False.
    health = {cid: g.health() for cid, g in grabbers.items()}
    sched.stop()
    for g in grabbers.values():
        g.stop()
    elapsed = time.time() - t0

    print()
    print("=" * 72)
    print(f"A. RTSP HEALTH   ({elapsed:.0f}s)")
    print("=" * 72)
    print(f"{'cam':>4}{'role':>9}{'conn':>7}{'received':>10}{'replaced':>10}"
          f"{'in fps':>8}{'reconn':>8}  last_error")
    for cid, g in sorted(grabbers.items()):
        h = health[cid]
        fps = h.frames_grabbed / elapsed if elapsed else 0
        print(f"{cid:>4}{role_for(cid):>9}{str(h.connected):>7}{h.frames_grabbed:>10}"
              f"{h.frames_dropped:>10}{fps:>8.1f}{h.reconnects:>8}  {h.last_error or '-'}")

    print()
    print("=" * 72)
    print("B. FRAME FRESHNESS  (selection_time - frame_timestamp)")
    print("=" * 72)
    print(f"{'cam':>4}{'n':>6}{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}")
    for cid in sorted(grabbers):
        v = frame_ages[cid]
        if not v:
            print(f"{cid:>4}{0:>6}{'-':>9}{'-':>9}{'-':>9}{'-':>9}")
            continue
        print(f"{cid:>4}{len(v):>6}{statistics.mean(v):>9.3f}"
              f"{statistics.median(v):>9.3f}{_pct(v,0.95):>9.3f}{max(v):>9.3f}")

    print()
    print("=" * 72)
    print("C. SCHEDULER FAIRNESS  (service wait = gap between this camera's turns)")
    print("=" * 72)
    total = sum(consumed.values()) or 1
    print(f"{'cam':>4}{'role':>9}{'served':>8}{'share':>8}{'avg wait':>10}"
          f"{'p95':>9}{'max':>9}{'starved':>9}")
    for cid in sorted(grabbers):
        v = service_gaps[cid]
        st = sched.stats[cid]
        avg = statistics.mean(v) if v else 0.0
        print(f"{cid:>4}{role_for(cid):>9}{consumed[cid]:>8}"
              f"{100*consumed[cid]/total:>7.1f}%{avg:>10.2f}"
              f"{_pct(v,0.95):>9.2f}{(max(v) if v else 0):>9.2f}"
              f"{st.starvation_selections:>9}")

    print()
    print("=" * 72)
    print("D. FRAMES RECEIVED vs CONSUMED  (proves the slot discards, not queues)")
    print("=" * 72)
    print(f"{'cam':>4}{'received':>10}{'consumed':>10}{'discarded':>11}{'discard %':>11}")
    for cid, g in sorted(grabbers.items()):
        h = health[cid]
        disc = h.frames_grabbed - consumed[cid]
        pct = 100 * disc / h.frames_grabbed if h.frames_grabbed else 0
        print(f"{cid:>4}{h.frames_grabbed:>10}{consumed[cid]:>10}{disc:>11}{pct:>10.1f}%")

    print()
    print(f"throughput: {total/elapsed:.3f} selections/s over {elapsed:.0f}s")
    if args.fault:
        others = [c for c in grabbers if c != args.fault]
        after = all(consumed[c] > 0 for c in others)
        print(f"fault isolation: camera {args.fault} killed; "
              f"others still served = {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
