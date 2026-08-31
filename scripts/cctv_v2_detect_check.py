"""Live YOLO detection check for the CCTV V2 pipeline (Step 6).

MANUAL / INTEGRATION ONLY. Needs four real cameras and the yolo11m weights, so
it is deliberately not importable by the automated suite -- `pytest` must stay
runnable on a machine with no DVR and no model file.

WHAT IT EXERCISES
-----------------
    RTSP -> grabber -> latest-frame slot -> scheduler -> YOLO -> person boxes

Everything up to the scheduler was validated in the previous step with a stub
that slept for an assumed inference cost. This replaces the stub with the real
detector, which answers the question that assumption was standing in for: does
V2 feed the correct fresh frames to YOLO at the correct role settings, and what
does it actually cost?

There is no tracking, no recognition and no attendance. Step 6 is detection.

WHAT TO LOOK AT
---------------
Doorway and room are reported separately throughout. They run at different input
sizes against different scenes, so a pooled average would describe no camera
that exists.

Two rates are reported and must not be confused:

  * inference FPS   1000/avg_ms. What one role could sustain if it had the
                    worker to itself.
  * effective FPS   passes actually completed / elapsed. What each camera got,
                    sharing one worker with three others.

The gap between them is the scheduler doing its job, not a fault. Reporting only
the first is how V1 came to believe a doorway was analysed every 0.12s.

Usage:
    python scripts/cctv_v2_detect_check.py --seconds 180
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


def _pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def _fmt(values, scale=1.0):
    if not values:
        return f"{'-':>9}{'-':>9}{'-':>9}{'-':>9}"
    return (f"{statistics.mean(values) * scale:>9.1f}"
            f"{statistics.median(values) * scale:>9.1f}"
            f"{_pct(values, 0.95) * scale:>9.1f}"
            f"{max(values) * scale:>9.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--workers", type=int, default=None,
                    help="inference workers; each builds its OWN YOLO instance. "
                         "Defaults to cctv_v2_inference_workers (3).")
    ap.add_argument("--fault", type=int, default=None,
                    help="camera id to disconnect mid-run")
    ap.add_argument("--recover-after", type=float, default=60.0)
    ap.add_argument("--dump", type=str, default=None,
                    help="write detection to this JSONL file")
    args = ap.parse_args()

    import logging
    logging.disable(logging.INFO)

    from app.cctv_v2.capture.grabber import CameraGrabber
    from app.cctv_v2.config.cameras import CAMERA_ROLE, profile_for, role_for
    from app.cctv_v2.pipeline.detect import PersonDetector
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
        url = rows.get(cid)
        if not url:
            print(f"  camera {cid}: no source_url in DB, skipping")
            continue
        grabbers[cid] = CameraGrabber(cid, url)

    workers = args.workers or get_settings().cctv_v2_inference_workers

    latency = defaultdict(list)        # camera -> [ms]
    ages = defaultdict(list)           # camera -> [ms] frame age at inference
    counts = defaultdict(list)         # camera -> [people per frame]
    consumed = defaultdict(int)
    cpu_samples = []
    models = []
    lock = threading.Lock()
    dump_fh = open(args.dump, "w", encoding="utf-8") if args.dump else None

    def make_worker():
        """Called ONCE PER WORKER, inside that worker's thread.

        Each call builds a fresh PersonDetector and therefore a fresh YOLO
        instance. Sharing one across workers measured SLOWER than a single
        worker (0.280 vs 0.403 passes/s), so this must not be hoisted.
        """
        det = PersonDetector()
        with lock:
            models.append(det)

        def process(camera_id, snap):
            result = det.detect(camera_id, snap)
            with lock:
                if dump_fh is not None:
                    dump_fh.write(json.dumps({
                        "t": time.time(), "camera_id": camera_id,
                        "frame_ts": result.frame_timestamp,
                        "seq": result.frame_sequence,
                        "n": result.count,
                        "confs": [round(d.confidence, 4) for d in result.detections],
                        "ms": round(result.inference_ms, 1),
                        "age_ms": round(result.frame_age_ms, 1),
                    }) + chr(10))
                    dump_fh.flush()
                latency[camera_id].append(result.inference_ms)
                ages[camera_id].append(result.frame_age_ms)
                counts[camera_id].append(result.count)
                consumed[camera_id] += 1
        return process

    sched = InferenceScheduler(
        grabbers.values(), process_factory=make_worker, workers=workers,
    )

    print(f"starting {len(grabbers)} grabbers + {workers} inference worker(s), "
          f"each with its OWN YOLO instance, for {args.seconds:.0f}s")
    for g in grabbers.values():
        g.start()
    time.sleep(3.0)
    if proc:
        proc.cpu_percent(None)         # prime the counter
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

    if dump_fh is not None:
        dump_fh.close()
    health = {cid: g.health() for cid, g in grabbers.items()}
    sched.stop()
    for g in grabbers.values():
        g.stop()
    elapsed = time.time() - t0

    # -- A ------------------------------------------------------------------
    print()
    print("=" * 78)
    print(f"A. INFERENCE LATENCY PER CAMERA   ({elapsed:.0f}s, {workers} worker(s), {len(models)} YOLO instances)")
    print("=" * 78)
    print(f"{'cam':>4}{'role':>9}{'imgsz':>7}{'conf':>7}{'n':>6}"
          f"{'mean ms':>9}{'med ms':>9}{'p95 ms':>9}{'max ms':>9}")
    for cid in sorted(grabbers):
        prof = profile_for(cid)
        print(f"{cid:>4}{role_for(cid):>9}{prof.input_size:>7}"
              f"{prof.predict_conf:>7.2f}{len(latency[cid]):>6}"
              + _fmt(latency[cid]))

    # -- B ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("B. DETECTIONS PER FRAME")
    print("=" * 78)
    print(f"{'cam':>4}{'role':>9}{'frames':>8}{'people':>8}{'per frame':>11}"
          f"{'max':>6}{'frames w/ people':>18}")
    for cid in sorted(grabbers):
        c = counts[cid]
        total = sum(c)
        withp = sum(1 for n in c if n > 0)
        print(f"{cid:>4}{role_for(cid):>9}{len(c):>8}{total:>8}"
              f"{(total / len(c) if c else 0):>11.2f}{(max(c) if c else 0):>6}"
              f"{withp:>12} ({100 * withp / len(c) if c else 0:.0f}%)")

    # -- C ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("C. INFERENCE FPS: capability vs what each camera actually got")
    print("=" * 78)
    print(f"{'cam':>4}{'role':>9}{'inference fps':>15}{'effective fps':>15}"
          f"{'sec/pass':>10}")
    for cid in sorted(grabbers):
        lat = latency[cid]
        cap = 1000.0 / statistics.mean(lat) if lat else 0.0
        eff = len(lat) / elapsed if elapsed else 0.0
        print(f"{cid:>4}{role_for(cid):>9}{cap:>15.2f}{eff:>15.3f}"
              f"{(1 / eff if eff else 0):>10.1f}")
    print("  inference fps = if this role had the worker to itself")
    print("  effective fps = what it got, sharing one worker with three others")

    # -- D ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("D. FRAME AGE AT INFERENCE   (ms; the cutoff is per role)")
    print("=" * 78)
    print(f"{'cam':>4}{'role':>9}{'cutoff':>9}{'mean':>9}{'med':>9}"
          f"{'p95':>9}{'max':>9}{'over cutoff':>13}")
    for cid in sorted(grabbers):
        a = ages[cid]
        cutoff_ms = profile_for(cid).max_frame_age * 1000.0
        over = sum(1 for v in a if v > cutoff_ms)
        print(f"{cid:>4}{role_for(cid):>9}{cutoff_ms:>9.0f}" + _fmt(a)
              + f"{over:>13}")

    # -- E ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("E. FRAMES DROPPED AND SKIPPED")
    print("=" * 78)
    print(f"{'cam':>4}{'role':>9}{'received':>10}{'to YOLO':>9}{'discarded':>11}"
          f"{'discard %':>11}{'stale skips':>13}{'reconn':>8}")
    for cid in sorted(grabbers):
        h = health[cid]
        disc = h.frames_grabbed - consumed[cid]
        pct = 100 * disc / h.frames_grabbed if h.frames_grabbed else 0
        print(f"{cid:>4}{role_for(cid):>9}{h.frames_grabbed:>10}"
              f"{consumed[cid]:>9}{disc:>11}{pct:>10.1f}%"
              f"{sched.stats[cid].stale_skips:>13}{h.reconnects:>8}")
    print("  discarded = superseded in the latest-frame slot before YOLO saw them.")
    print("  This number SHOULD be large: it is the queue that never formed.")

    # -- F ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("F. DOORWAY vs ROOM")
    print("=" * 78)
    merged = {}
    for role in ("doorway", "room"):
        cams = [c for c in grabbers if role_for(c) == role]
        lat = [v for c in cams for v in latency[c]]
        det = [v for c in cams for v in counts[c]]
        merged[role] = {
            "passes": len(lat),
            "avg_ms": statistics.mean(lat) if lat else 0.0,
            "p95_ms": _pct(lat, 0.95),
            "inference_fps": (1000.0 / statistics.mean(lat)) if lat else 0.0,
            "detections_per_frame": (sum(det) / len(det)) if det else 0.0,
            "failures": sum(m.timing[role].failures for m in models),
        }
    summary = merged
    print(f"{'role':>9}{'imgsz':>7}{'conf':>7}{'passes':>8}{'avg ms':>9}"
          f"{'p95 ms':>9}{'fps':>7}{'ppl/frame':>11}{'fails':>7}")
    for role in ("doorway", "room"):
        s = summary[role]
        cid = next(c for c in grabbers if role_for(c) == role)
        prof = profile_for(cid)
        print(f"{role:>9}{prof.input_size:>7}{prof.predict_conf:>7.2f}"
              f"{s['passes']:>8}{s['avg_ms']:>9.1f}{s['p95_ms']:>9.1f}"
              f"{s['inference_fps']:>7.2f}{s['detections_per_frame']:>11.2f}"
              f"{s['failures']:>7}")

    d, r = summary["doorway"], summary["room"]
    if d["avg_ms"] and r["avg_ms"]:
        print(f"  a doorway pass costs {d['avg_ms'] / r['avg_ms']:.2f}x a room pass "
              f"({d['avg_ms']:.0f}ms at 640px vs {r['avg_ms']:.0f}ms at 480px)")

    # -- G ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("G. CPU")
    print("=" * 78)
    if cpu_samples:
        cores = psutil.cpu_count(logical=False) or 1
        logical = psutil.cpu_count() or 1
        print(f"  process CPU:  mean {statistics.mean(cpu_samples):.0f}%  "
              f"p95 {_pct(cpu_samples, 0.95):.0f}%  max {max(cpu_samples):.0f}%  "
              f"(n={len(cpu_samples)})")
        print(f"  machine:      {cores} physical / {logical} logical cores "
              f"-- 100% is one core, {100 * logical}% is all of them")
        print(f"  saturation:   {statistics.mean(cpu_samples) / logical:.0f}% "
              f"of total capacity")
    else:
        print("  psutil not available -- no CPU measurement")

    if args.fault and faulted_at:
        cid = args.fault
        print()
        print("=" * 78)
        print(f"H. FAULT ISOLATION   camera {cid}")
        print("=" * 78)
        print(f"  disconnected at        t+{faulted_at - t0:.0f}s")
        if recovered_at:
            print(f"  reconnected at         t+{recovered_at - t0:.0f}s")
        print(f"  stale skips:           {sched.stats[cid].stale_skips}")
        print(f"  starvation selections: {sched.stats[cid].starvation_selections}")
        others = sorted(c for c in grabbers if c != cid)
        print("  other cameras served:  "
              + ", ".join(f"{c}={consumed[c]}" for c in others))

    total = sum(consumed.values())
    print()
    print(f"workers: {workers}   independent YOLO instances: {len(models)}")
    print(f"throughput: {total / elapsed:.3f} detections/s over {elapsed:.0f}s "
          f"({total} passes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
