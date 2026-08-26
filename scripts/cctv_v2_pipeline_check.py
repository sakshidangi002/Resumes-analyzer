"""Live end-to-end check: detect -> track -> crossing + occupancy.

MANUAL / INTEGRATION ONLY. Needs four real cameras and the yolo11m weights.

    grabber -> slot -> scheduler -> YOLO -> tracker -> crossing   (57, 58)
                                                    -> occupancy  (59, 60)

No recognition, no attendance, no database. This measures whether the counting
layers work on real footage, and reports what they cannot establish rather than
filling the gap with a guess.

WHAT IT CANNOT TELL YOU
-----------------------
Transits DETECTED is not transits that HAPPENED. A person who crosses entirely
between two inference passes -- 5.34s apart, measured -- is never seen on both
sides of the line, so no crossing exists to find and nothing here can count
them. The miss rate needs ground truth from outside the pipeline; see
scripts/cctv_v2_ground_truth.py.

Usage:
    python scripts/cctv_v2_pipeline_check.py --seconds 300
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Attendance Management" / "backend"
for p in (str(BACKEND), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dump", type=str, default=None,
                    help="JSONL file for every transit event and pass")
    args = ap.parse_args()

    import logging
    logging.disable(logging.INFO)

    from app.cctv_v2.capture.grabber import CameraGrabber
    from app.cctv_v2.config.cameras import CAMERA_ROLE, profile_for, role_for
    from app.cctv_v2.config.geometry import crossing_line, has_chair_map
    from app.cctv_v2.pipeline.crossing import CrossingRegistry
    from app.cctv_v2.pipeline.detect import PersonDetector
    from app.cctv_v2.pipeline.occupancy import OccupancyRegistry
    from app.cctv_v2.pipeline.track import TrackerRegistry
    from app.cctv_v2.scheduler.loop import InferenceScheduler
    from app.core.config import get_settings
    from app.db.session import SessionLocal
    from app.models.camera import CameraConfig

    try:
        import psutil
        proc = psutil.Process()
    except Exception:
        psutil, proc = None, None

    with SessionLocal() as db:
        rows = {c.id: c.source_url for c in db.query(CameraConfig).all()}
    grabbers = {cid: CameraGrabber(cid, rows[cid])
                for cid in sorted(CAMERA_ROLE) if rows.get(cid)}
    workers = args.workers or get_settings().cctv_v2_inference_workers

    tracks = TrackerRegistry()
    crossings = CrossingRegistry()
    occupancy = OccupancyRegistry()
    state_lock = threading.Lock()

    latency = defaultdict(list)
    ages = defaultdict(list)
    det_counts = defaultdict(list)
    people_series = defaultdict(list)
    consumed = defaultdict(int)
    cpu = []
    lock = threading.Lock()
    dump_fh = open(args.dump, "w", encoding="utf-8") if args.dump else None

    def make_worker():
        det = PersonDetector()

        def process(camera_id, snap):
            result = det.detect(camera_id, snap)
            with state_lock:
                seen = tracks.update(result)
                tr = tracks.get(camera_id)
                if role_for(camera_id) == "doorway":
                    fired = crossings.update(camera_id, seen,
                                             result.frame_timestamp,
                                             result.frame_sequence)
                    if dump_fh is not None:
                        for ev in fired:
                            dump_fh.write(json.dumps({
                                "kind": "transit", "t": ev.timestamp,
                                "camera_id": ev.camera_id, "track_id": ev.track_id,
                                "direction": ev.direction.value,
                                "travel": round(ev.travel, 4),
                                "hits": ev.track_hits,
                                "identity": ev.identity,
                            }) + chr(10))
                        dump_fh.write(json.dumps({
                            "kind": "pass", "t": result.frame_timestamp,
                            "camera_id": camera_id, "n": result.count,
                            "tracks": len(seen),
                        }) + chr(10))
                        dump_fh.flush()
                    crossings.get(camera_id).forget(
                        tid for tid in list(crossings.get(camera_id)._state)
                        if tid not in {t.track_id for t in tr.all_tracks()}
                    )
                    present = len(tr.active_tracks())
                else:
                    snapshot = occupancy.update(
                        camera_id, tr.active_tracks(), result.frame_timestamp)
                    present = snapshot.people_count
            with lock:
                latency[camera_id].append(result.inference_ms)
                ages[camera_id].append(result.frame_age_ms)
                det_counts[camera_id].append(result.count)
                people_series[camera_id].append(present)
                consumed[camera_id] += 1
        return process

    sched = InferenceScheduler(grabbers.values(),
                               process_factory=make_worker, workers=workers)
    print(f"starting {len(grabbers)} cameras, {workers} workers, "
          f"full pipeline, {args.seconds:.0f}s")
    for g in grabbers.values():
        g.start()
    time.sleep(3.0)
    if proc:
        proc.cpu_percent(None)
    sched._started_at = time.time()
    sched.start()

    t0 = time.time()
    while time.time() - t0 < args.seconds:
        time.sleep(2.0)
        if proc:
            cpu.append(proc.cpu_percent(None))
    if dump_fh is not None:
        dump_fh.close()
    health = {c: g.health() for c, g in grabbers.items()}
    sched.stop()
    for g in grabbers.values():
        g.stop()
    elapsed = time.time() - t0

    print()
    print("=" * 78)
    print(f"A. DOORWAY TRANSITS   ({elapsed:.0f}s)")
    print("=" * 78)
    print(f"{'cam':>4}{'line':>7}{'passes':>8}{'tracks':>8}{'IN':>5}{'OUT':>5}"
          f"{'unk IN':>8}{'unk OUT':>9}{'rej':>6}")
    for cid in sorted(c for c in grabbers if role_for(c) == "doorway"):
        d = crossings.get(cid).summary()
        tr = tracks.get(cid).stats() if cid in tracks.cameras() else {"created": 0}
        line = crossing_line(cid)
        rej = (d["rejected_unconfirmed"] + d["rejected_short_travel"]
               + d["rejected_cooldown"])
        print(f"{cid:>4}{line.position:>7.2f}{consumed[cid]:>8}{tr['created']:>8}"
              f"{d['people_in']:>5}{d['people_out']:>5}"
              f"{d['unknown_in']:>8}{d['unknown_out']:>9}{rej:>6}")
    print("  Every transit is unattributed by construction -- IN == unknown IN.")
    print("  DETECTED transits only. Crossings completed between two passes")
    print("  leave no trace; that miss rate needs external ground truth.")

    print()
    print("=" * 78)
    print("B. ROOM OCCUPANCY   (per camera -- never summed)")
    print("=" * 78)
    print(f"{'cam':>4}{'passes':>8}{'det/pass':>10}{'people now':>12}"
          f"{'people mean':>13}{'max':>5}{'chairs':>8}{'occ':>5}{'free':>6}")
    for cid in sorted(c for c in grabbers if role_for(c) == "room"):
        s = occupancy.snapshot().get(cid)
        ppl = people_series[cid]
        if not s:
            print(f"{cid:>4}  no occupancy snapshot")
            continue
        print(f"{cid:>4}{consumed[cid]:>8}"
              f"{(statistics.mean(det_counts[cid]) if det_counts[cid] else 0):>10.2f}"
              f"{s['people_count']:>12}"
              f"{(statistics.mean(ppl) if ppl else 0):>13.2f}"
              f"{(max(ppl) if ppl else 0):>5}"
              f"{s['chairs_total']:>8}{s['chairs_occupied']:>5}{s['chairs_free']:>6}")
        if not s["chair_map_configured"]:
            print(f"       camera {cid}: NO CHAIR MAP CONFIGURED -- chair columns")
            print(f"       are 0 because no seats are defined, not because the")
            print(f"       room is empty. Run scripts/cctv_v2_chair_setup.py.")
    for cid, row in occupancy.snapshot().items():
        if row.get("shares_room_with"):
            print(f"  camera {cid} shares a room with {row['shares_room_with']}; "
                  f"counts are NOT added.")

    print()
    print("=" * 78)
    print("C. PIPELINE HEALTH")
    print("=" * 78)
    print(f"{'cam':>4}{'role':>9}{'infer ms':>10}{'frame age':>11}"
          f"{'over cutoff':>13}{'stale':>7}{'reconn':>8}")
    for cid in sorted(grabbers):
        lat, ag = latency[cid], ages[cid]
        cut = profile_for(cid).max_frame_age * 1000.0
        print(f"{cid:>4}{role_for(cid):>9}"
              f"{(statistics.mean(lat) if lat else 0):>10.0f}"
              f"{(statistics.mean(ag) if ag else 0):>11.1f}"
              f"{sum(1 for v in ag if v > cut):>13}"
              f"{sched.stats[cid].stale_skips:>7}{health[cid].reconnects:>8}")

    if cpu and psutil:
        print()
        print(f"CPU mean {statistics.mean(cpu):.0f}%  max {max(cpu):.0f}%  "
              f"({psutil.cpu_count()} logical)")
    print(f"throughput {sum(consumed.values()) / elapsed:.3f} passes/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
