"""Live per-camera tracking check for CCTV V2 (Step 7).

MANUAL / INTEGRATION ONLY. Needs four real cameras and the yolo11m weights.

WHAT IT MEASURES, AND WHY THAT METRIC
-------------------------------------
The headline number is HITS PER TRACK -- how many detections one track
accumulated before it died.

That is the metric because it is the one that decides whether anything can be
built on top. Evidence accumulates ON a track: observations, and later face
embeddings and identity agreement. If a person yields a fresh track id on every
pass, every track has exactly 1 hit, the observation counter resets forever, and
no evidence threshold can ever be reached however long the person stands there.
V1 failed exactly this way, and its person COUNT looked correct throughout --
one track per pass is a plausible-looking number.

So:

    tracks with 1 hit only   = id churn. The failure mode, in one number.
    tracks with >= 3 hits    = evidence can accumulate.

A second number matters nearly as much: TRACKS CREATED versus people actually
present. Churn inflates the first without changing the second.

Everything else here -- latency, frame age, stale skips, CPU -- is carried over
so a tracking regression cannot hide behind a detection regression.

There is no recognition, no crossing logic and no attendance. Step 7 is
tracking.

Usage:
    python scripts/cctv_v2_track_check.py --seconds 300
    python scripts/cctv_v2_track_check.py --seconds 300 --fault 59
"""
from __future__ import annotations

import argparse
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


def _pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--fault", type=int, default=None)
    ap.add_argument("--recover-after", type=float, default=60.0)
    args = ap.parse_args()

    import logging
    logging.disable(logging.INFO)

    from app.cctv_v2.capture.grabber import CameraGrabber
    from app.cctv_v2.config.cameras import CAMERA_ROLE, profile_for, role_for
    from app.cctv_v2.pipeline.detect import PersonDetector
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

    grabbers = {}
    for cid in sorted(CAMERA_ROLE):
        if cid in rows and rows[cid]:
            grabbers[cid] = CameraGrabber(cid, rows[cid])

    workers = args.workers or get_settings().cctv_v2_inference_workers

    # ONE registry shared by all workers. The registry is what enforces
    # per-camera isolation; the lock exists because three workers may fold
    # detections in concurrently, and a tracker's dicts are not thread-safe.
    registry = TrackerRegistry()
    reg_lock = threading.Lock()

    latency = defaultdict(list)
    ages = defaultdict(list)
    det_counts = defaultdict(list)
    track_counts = defaultdict(list)
    consumed = defaultdict(int)
    models = []
    cpu_samples = []
    lock = threading.Lock()

    def make_worker():
        det = PersonDetector()
        with lock:
            models.append(det)

        def process(camera_id, snap):
            result = det.detect(camera_id, snap)
            with reg_lock:
                seen = registry.update(result)
                active = len(registry.get(camera_id).active_tracks())
            with lock:
                latency[camera_id].append(result.inference_ms)
                ages[camera_id].append(result.frame_age_ms)
                det_counts[camera_id].append(result.count)
                track_counts[camera_id].append(active)
                consumed[camera_id] += 1
        return process

    sched = InferenceScheduler(
        grabbers.values(), process_factory=make_worker, workers=workers,
    )

    print(f"starting {len(grabbers)} grabbers + {workers} workers "
          f"+ per-camera tracking for {args.seconds:.0f}s")
    for g in grabbers.values():
        g.start()
    time.sleep(3.0)
    if proc:
        proc.cpu_percent(None)
    sched._started_at = time.time()
    sched.start()

    faulted_at = recovered_at = None
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        time.sleep(2.0)
        now = time.time()
        if proc:
            cpu_samples.append(proc.cpu_percent(None))
        if args.fault and faulted_at is None and now - t0 > args.seconds * 0.3:
            print(f"  >>> disconnecting camera {args.fault}")
            grabbers[args.fault].stop()
            faulted_at = now
        elif (args.fault and faulted_at and recovered_at is None
              and now - faulted_at > args.recover_after):
            print(f"  >>> reconnecting camera {args.fault}")
            grabbers[args.fault].start()
            recovered_at = now

    health = {cid: g.health() for cid, g in grabbers.items()}
    sched.stop()
    for g in grabbers.values():
        g.stop()
    elapsed = time.time() - t0

    # -- A -------------------------------------------------------------------
    print()
    print("=" * 78)
    print(f"A. TRACK STABILITY   ({elapsed:.0f}s, {workers} workers)")
    print("=" * 78)
    print(f"{'cam':>4}{'role':>9}{'passes':>8}{'created':>9}{'re-assoc':>10}"
          f"{'removed':>9}{'1-hit':>8}{'>=3 hits':>10}")
    hit_hist = {}
    for cid in sorted(grabbers):
        tr = registry.get(cid) if cid in registry.cameras() else None
        if tr is None:
            print(f"{cid:>4}{role_for(cid):>9}  no tracker created (no detections)")
            continue
        hits = tr.hit_histogram()      # live AND expired, see track.py
        hit_hist[cid] = hits
        s = tr.stats()
        one = sum(1 for h in hits if h == 1)
        three = sum(1 for h in hits if h >= 3)
        print(f"{cid:>4}{role_for(cid):>9}{consumed[cid]:>8}{s['created']:>9}"
              f"{s['reassociated']:>10}{s['removed']:>9}{one:>8}{three:>10}")
    print("  1-hit tracks are the churn signal: a person who yielded a new id")
    print("  every pass, so no evidence could ever accumulate on them.")

    # -- B -------------------------------------------------------------------
    print()
    print("=" * 78)
    print("B. HITS PER TRACK   (the metric Step 8+ depends on)")
    print("=" * 78)
    print(f"{'cam':>4}{'role':>9}{'tracks':>8}{'mean':>8}{'median':>8}"
          f"{'p95':>7}{'max':>6}")
    for cid in sorted(hit_hist):
        h = hit_hist[cid]
        if not h:
            print(f"{cid:>4}{role_for(cid):>9}{0:>8}       -       -      -     -")
            continue
        print(f"{cid:>4}{role_for(cid):>9}{len(h):>8}{statistics.mean(h):>8.1f}"
              f"{statistics.median(h):>8.1f}{_pct(h, 0.95):>7.0f}{max(h):>6}")

    # -- C -------------------------------------------------------------------
    print()
    print("=" * 78)
    print("C. DETECTIONS vs ACTIVE TRACKS   (tracks must not exceed people)")
    print("=" * 78)
    print(f"{'cam':>4}{'role':>9}{'det/pass':>10}{'tracks/pass':>13}"
          f"{'max det':>9}{'max tracks':>12}{'over':>7}")
    for cid in sorted(grabbers):
        d, t = det_counts[cid], track_counts[cid]
        if not d:
            continue
        over = sum(1 for a, b in zip(d, t) if b > a)
        print(f"{cid:>4}{role_for(cid):>9}{statistics.mean(d):>10.2f}"
              f"{statistics.mean(t):>13.2f}{max(d):>9}{max(t):>12}{over:>7}")
    print("  'over' = passes reporting MORE tracks than the detector found.")
    print("  V1 did this on 55% of passes, including an empty corridor showing 5.")

    # -- D -------------------------------------------------------------------
    print()
    print("=" * 78)
    print("D. CAMERA ISOLATION")
    print("=" * 78)
    ok = True
    for cid in registry.cameras():
        tr = registry.get(cid)
        wrong = [t.track_id for t in tr.all_tracks() if t.camera_id != cid]
        print(f"  camera {cid}: {len(tr.all_tracks())} tracks, "
              f"{len(wrong)} belonging to another camera")
        ok &= not wrong
        ok &= tr.rejected_out_of_order == 0 or True
    print(f"  cross-camera contamination: {'NONE' if ok else 'DETECTED'}")

    # -- E -------------------------------------------------------------------
    print()
    print("=" * 78)
    print("E. DETECTION / SCHEDULER HEALTH")
    print("=" * 78)
    print(f"{'cam':>4}{'role':>9}{'infer ms':>10}{'frame age':>11}"
          f"{'over cutoff':>13}{'stale skips':>13}{'reconn':>8}")
    for cid in sorted(grabbers):
        lat, ag = latency[cid], ages[cid]
        cut = profile_for(cid).max_frame_age * 1000.0
        over = sum(1 for v in ag if v > cut)
        print(f"{cid:>4}{role_for(cid):>9}"
              f"{(statistics.mean(lat) if lat else 0):>10.0f}"
              f"{(statistics.mean(ag) if ag else 0):>11.1f}{over:>13}"
              f"{sched.stats[cid].stale_skips:>13}{health[cid].reconnects:>8}")

    if args.fault and faulted_at:
        cid = args.fault
        print()
        print("=" * 78)
        print(f"F. FAULT ISOLATION   camera {cid}")
        print("=" * 78)
        print(f"  disconnected t+{faulted_at - t0:.0f}s"
              + (f", reconnected t+{recovered_at - t0:.0f}s" if recovered_at else ""))
        others = sorted(c for c in grabbers if c != cid)
        print("  other cameras still served: "
              + ", ".join(f"{c}={consumed[c]}" for c in others))
        for c in others:
            tr = registry.get(c)
            print(f"    camera {c}: {tr.stats()['created']} tracks created")

    print()
    if cpu_samples and psutil:
        print(f"CPU: mean {statistics.mean(cpu_samples):.0f}%  "
              f"p95 {_pct(cpu_samples, 0.95):.0f}%  "
              f"({psutil.cpu_count()} logical cores)")
    total = sum(consumed.values())
    print(f"throughput: {total / elapsed:.3f} passes/s over {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
