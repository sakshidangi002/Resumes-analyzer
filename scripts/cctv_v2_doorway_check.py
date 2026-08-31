"""Step 6.5 validation: is the doorway detector working, and can the scheduler
sample fast enough to see a transit?

MANUAL / INTEGRATION ONLY. Needs real cameras and the yolo11m weights.

WHY THIS EXISTS
---------------
The Step 6 live run found zero people on camera 57 across 32 passes and four on
camera 58. That number has three completely different explanations and the run
could not tell them apart:

    A. the corridor was empty
    B. YOLO works, but the scheduler samples too slowly to catch a 2s transit
    C. YOLO at 640/0.15 fails on doorway frames

Telling them apart needs GROUND TRUTH -- what YOLO would see if it looked at
every frame -- against which the scheduler's actual sampling can be compared.
So this records frames at a fixed rate first and runs inference offline
afterwards. Inference costs 3.4s per pass and the cameras deliver 12fps; there
is no way to do both at once, and trying is what makes a live measurement
measure the measurement.

    record   capture frames from the doorways while somebody walks through
    analyse  run YOLO on EVERY recorded frame -> ground truth
    cpubench CPU cost per pass, and 1 vs 2 concurrent inferences
    schedsim what the real measured costs do to the scheduler's fairness

The 640/0.15 doorway configuration is NOT changed here. Measure first.

Usage:
    python scripts/cctv_v2_doorway_check.py record --seconds 90 --fps 3
    python scripts/cctv_v2_doorway_check.py analyse
    python scripts/cctv_v2_doorway_check.py cpubench
    python scripts/cctv_v2_doorway_check.py schedsim
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

CLIP_DIR = REPO / "data" / "cctv_v2_step65"
DOORWAYS = (57, 58)

# Measured in the Step 6 live run; used by schedsim.
REAL_COST = {"doorway": 3.406, "room": 2.632}


def _pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def _stats(values, fmt="{:.3f}"):
    if not values:
        return "no samples"
    return (f"min {fmt.format(min(values))}  mean {fmt.format(statistics.mean(values))}"
            f"  median {fmt.format(statistics.median(values))}"
            f"  p95 {fmt.format(_pct(values, 0.95))}  max {fmt.format(max(values))}")


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------
def cmd_record(args) -> int:
    import cv2

    global CLIP_DIR
    if getattr(args, "out", None):
        CLIP_DIR = REPO / "data" / args.out

    from app.cctv_v2.capture.grabber import CameraGrabber
    from app.db.session import SessionLocal
    from app.models.camera import CameraConfig

    out = CLIP_DIR
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    for old in out.glob("*.json"):
        old.unlink()

    with SessionLocal() as db:
        rows = {c.id: c.source_url for c in db.query(CameraConfig).all()}

    grabbers = {}
    for cid in DOORWAYS:
        if cid not in rows:
            print(f"  camera {cid}: not in DB, skipping")
            continue
        grabbers[cid] = CameraGrabber(cid, rows[cid])
        grabbers[cid].start()

    print(f"connecting to cameras {sorted(grabbers)}...")
    time.sleep(4.0)

    print()
    print("=" * 70)
    print(f"  RECORDING FOR {args.seconds:.0f} SECONDS AT {args.fps} FPS")
    print(f"  WALK THROUGH CAMERA 57 AND CAMERA 58 SEVERAL TIMES NOW.")
    print("  Walk at normal speed. Cross the full width of the view.")
    print("=" * 70)
    print()

    interval = 1.0 / args.fps
    manifest = []
    t0 = time.time()
    last_seq = {cid: -1 for cid in grabbers}
    n = 0
    next_tick = t0

    while time.time() - t0 < args.seconds:
        now = time.time()
        if now < next_tick:
            time.sleep(min(0.02, next_tick - now))
            continue
        next_tick += interval

        for cid, g in grabbers.items():
            snap = g.get_latest()
            if snap is None or snap.sequence == last_seq[cid]:
                continue
            last_seq[cid] = snap.sequence
            name = f"{cid}_{n:05d}.jpg"
            cv2.imwrite(str(out / name), snap.frame)
            manifest.append({
                "file": name, "camera_id": cid,
                "timestamp": snap.timestamp, "sequence": snap.sequence,
                "t_rel": snap.timestamp - t0,
            })
            n += 1

        el = time.time() - t0
        if int(el) % 15 == 0 and abs(el - round(el)) < 0.05:
            print(f"  ...{el:.0f}s / {args.seconds:.0f}s  ({n} frames)")

    for g in grabbers.values():
        g.stop()

    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    per_cam = defaultdict(int)
    for m in manifest:
        per_cam[m["camera_id"]] += 1
    print()
    print(f"recorded {n} frames to {out}")
    for cid in sorted(per_cam):
        print(f"  camera {cid}: {per_cam[cid]} frames")
    print()
    print("next:  python scripts/cctv_v2_doorway_check.py analyse")
    return 0


# ---------------------------------------------------------------------------
# analyse
# ---------------------------------------------------------------------------
def cmd_analyse(args) -> int:
    import cv2

    global CLIP_DIR
    if getattr(args, "dir", None):
        CLIP_DIR = REPO / "data" / args.dir

    from app.cctv_v2.capture.grabber import FrameSnapshot
    from app.cctv_v2.config.cameras import profile_for
    from app.cctv_v2.pipeline.detect import PersonDetector

    manifest_path = CLIP_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"no recording at {CLIP_DIR}; run `record` first")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    det = PersonDetector()
    if not det.available:
        print("YOLO could not be loaded")
        return 1

    by_cam = defaultdict(list)
    for m in manifest:
        by_cam[m["camera_id"]].append(m)

    total = len(manifest)
    print(f"running YOLO on all {total} recorded frames at the UNCHANGED "
          f"doorway settings (640/0.15)")
    print("this is the expensive part -- roughly "
          f"{total / (0.6 if args.workers >= 3 else 0.3) / 60:.0f} minutes "
          f"at {args.workers} workers")
    print()

    # Analysed by a pool with INDEPENDENT detectors, for the same reason the
    # live scheduler uses one: sharing a model across threads measured slower
    # than not parallelising at all.
    import threading

    pool = [det] + [PersonDetector() for _ in range(max(0, args.workers - 1))]
    print(f"  using {len(pool)} independent YOLO instances")

    jobs = []
    for cid in sorted(by_cam):
        for m in sorted(by_cam[cid], key=lambda x: x["t_rel"]):
            jobs.append((cid, m))

    out_rows = [None] * len(jobs)
    next_i = [0]
    lock = threading.Lock()
    done = [0]

    def worker(d):
        while True:
            with lock:
                if next_i[0] >= len(jobs):
                    return
                i = next_i[0]
                next_i[0] += 1
            cid, m = jobs[i]
            img = cv2.imread(str(CLIP_DIR / m["file"]))
            if img is None:
                continue
            r = d.detect(cid, FrameSnapshot(cid, img, m["timestamp"], m["sequence"]))
            out_rows[i] = (cid, {
                "t": m["t_rel"],
                "n": r.count,
                "confs": [x.confidence for x in r.detections],
                "boxes": [x.bbox for x in r.detections],
                "ms": r.inference_ms,
                "shape": img.shape[:2],
            })
            with lock:
                done[0] += 1
                if done[0] % 25 == 0:
                    print(f"  ...{done[0]}/{total}")

    threads = [threading.Thread(target=worker, args=(d,)) for d in pool]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    results = defaultdict(list)
    for entry in out_rows:
        if entry is not None:
            results[entry[0]].append(entry[1])
    for cid in results:
        results[cid].sort(key=lambda r: r["t"])
    results = dict(results)

    _report_analysis(results, profile_for)
    (CLIP_DIR / "analysis.json").write_text(json.dumps(results), encoding="utf-8")
    return 0


def _transits(rows, gap_sec=1.5):
    """Group consecutive person-frames into transits.

    A gap longer than `gap_sec` with nobody visible ends one transit and starts
    another. 1.5s is deliberately longer than a single missed frame at 3fps, so
    one dropped detection mid-walk does not split one person into two.
    """
    out, cur = [], None
    for r in rows:
        if r["n"] > 0:
            if cur is None:
                cur = {"start": r["t"], "end": r["t"], "frames": 1,
                       "peak": r["n"], "confs": list(r["confs"])}
            elif r["t"] - cur["end"] <= gap_sec:
                cur["end"] = r["t"]
                cur["frames"] += 1
                cur["peak"] = max(cur["peak"], r["n"])
                cur["confs"].extend(r["confs"])
            else:
                out.append(cur)
                cur = {"start": r["t"], "end": r["t"], "frames": 1,
                       "peak": r["n"], "confs": list(r["confs"])}
    if cur:
        out.append(cur)
    return out


def _missed(transits, sample_interval, phases=20):
    """How many transits a scheduler sampling every `sample_interval` seconds
    would see nothing of.

    Averaged over `phases` starting offsets, because whether a given transit is
    caught depends entirely on where the sampling happens to land, and a single
    phase would report luck rather than behaviour.
    """
    if not transits:
        return 0.0, 0.0
    rates = []
    for k in range(phases):
        offset = sample_interval * k / phases
        seen = 0
        for tr in transits:
            lo, hi = tr["start"], tr["end"]
            # is there a sample instant inside [lo, hi]?
            first = offset
            if hi < first:
                continue
            import math
            idx = math.ceil((lo - offset) / sample_interval)
            t = offset + idx * sample_interval
            if lo <= t <= hi:
                seen += 1
        rates.append(seen / len(transits))
    return statistics.mean(rates), 1.0 - statistics.mean(rates)


def _report_analysis(results, profile_for):
    print()
    print("=" * 78)
    print("GROUND TRUTH -- YOLO on EVERY recorded frame (640 / 0.15, unchanged)")
    print("=" * 78)
    for cid in sorted(results):
        rows = results[cid]
        withp = [r for r in rows if r["n"] > 0]
        confs = [c for r in rows for c in r["confs"]]
        print()
        print(f"CAMERA {cid}")
        print(f"  frames analysed        {len(rows)}")
        print(f"  frames with a person   {len(withp)}"
              f"  ({100 * len(withp) / len(rows) if rows else 0:.0f}%)")
        print(f"  detections             {sum(r['n'] for r in rows)}")
        print(f"  detections per frame   "
              f"{sum(r['n'] for r in rows) / len(rows) if rows else 0:.2f}")
        if rows:
            h, w = rows[0]["shape"]
            print(f"  frame size             {w}x{h}")
        if confs:
            print(f"  confidence             {_stats(confs)}")
        else:
            print("  confidence             NO DETECTIONS AT ALL")

        boxes = [b for r in rows for b in r["boxes"]]
        if boxes and rows:
            h, w = rows[0]["shape"]
            widths = [b[2] - b[0] for b in boxes]
            heights = [b[3] - b[1] for b in boxes]
            areas = [100.0 * (b[2] - b[0]) * (b[3] - b[1]) / (w * h) for b in boxes]
            print(f"  bbox width px          {_stats(widths, '{:.0f}')}")
            print(f"  bbox height px         {_stats(heights, '{:.0f}')}")
            print(f"  bbox area % of frame   {_stats(areas, '{:.1f}')}")

        gaps = []
        prev = None
        for r in withp:
            if prev is not None:
                gaps.append(r["t"] - prev)
            prev = r["t"]
        if gaps:
            print(f"  gap between person-frames  {_stats(gaps)}s")

        trs = _transits(rows)
        print(f"  transits observed      {len(trs)}")
        for i, tr in enumerate(trs, 1):
            dur = tr["end"] - tr["start"]
            mc = statistics.mean(tr["confs"]) if tr["confs"] else 0
            print(f"    {i:>2}. t={tr['start']:6.1f}s  duration {dur:4.1f}s  "
                  f"{tr['frames']:>2} frames  peak {tr['peak']} people  "
                  f"mean conf {mc:.2f}")
        if trs:
            durs = [t["end"] - t["start"] for t in trs]
            print(f"  transit duration       {_stats(durs)}s")

            print()
            print(f"  IF THE SCHEDULER SAMPLES EVERY N SECONDS, "
                  f"WHAT FRACTION OF THESE {len(trs)} TRANSITS IS SEEN?")
            print(f"    {'interval':>10}{'caught':>9}{'missed':>9}")
            for iv in (1.0, 2.0, 3.0, 5.0, 9.4, 15.0):
                caught, missed = _missed(trs, iv)
                flag = "   <-- measured Step 6 doorway interval" if iv == 9.4 else ""
                print(f"    {iv:>9.1f}s{100 * caught:>8.0f}%"
                      f"{100 * missed:>8.0f}%{flag}")

    print()
    print("=" * 78)
    print("VERDICT INPUTS")
    print("=" * 78)
    for cid in sorted(results):
        rows = results[cid]
        withp = sum(1 for r in rows if r["n"] > 0)
        if not rows:
            print(f"  camera {cid}: no frames")
        elif withp == 0:
            print(f"  camera {cid}: YOLO found NOBODY in {len(rows)} frames. "
                  f"Either nobody walked through, or C (config failing).")
        else:
            print(f"  camera {cid}: YOLO found people in {withp}/{len(rows)} "
                  f"frames -> detection WORKS at 640/0.15. "
                  f"Any live miss is sampling rate (B), not config (C).")


# ---------------------------------------------------------------------------
# cpubench
# ---------------------------------------------------------------------------
def cmd_cpubench(args) -> int:
    import cv2
    import psutil

    from app.cctv_v2.capture.grabber import FrameSnapshot
    from app.cctv_v2.pipeline.detect import PersonDetector

    manifest_path = CLIP_DIR / "manifest.json"
    frames = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for m in manifest[:args.passes * 2]:
            img = cv2.imread(str(CLIP_DIR / m["file"]))
            if img is not None:
                frames.append((m["camera_id"], img))
    if not frames:
        print(f"no recorded frames in {CLIP_DIR}; run `record` first")
        return 1

    det = PersonDetector()
    if not det.available:
        print("YOLO could not be loaded")
        return 1

    proc = psutil.Process()
    logical = psutil.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) or 1

    print(f"machine: {physical} physical / {logical} logical cores")
    print(f"warming up...")
    cid, img = frames[0]
    det.detect(cid, FrameSnapshot(cid, img, time.time(), 0))

    # ---- serial ---------------------------------------------------------
    n = min(args.passes, len(frames))
    print(f"\nrunning {n} SERIAL passes...")
    cpu0 = proc.cpu_times()
    t0 = time.time()
    lat = []
    for i in range(n):
        cid, img = frames[i]
        s = time.time()
        det.detect(cid, FrameSnapshot(cid, img, s, i))
        lat.append(time.time() - s)
    serial_wall = time.time() - t0
    cpu1 = proc.cpu_times()
    serial_cpu = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)

    print(f"  wall            {serial_wall:.1f}s")
    print(f"  latency         {_stats(lat)}s")
    print(f"  throughput      {n / serial_wall:.3f} passes/s")
    print(f"  CPU-seconds     {serial_cpu:.1f}s total, "
          f"{serial_cpu / n:.2f}s per pass")
    print(f"  cores busy      {serial_cpu / serial_wall:.2f} "
          f"of {logical} logical ({100 * serial_cpu / serial_wall / logical:.0f}%)")

    # ---- two concurrent -------------------------------------------------
    print(f"\nrunning {n} passes with 2 CONCURRENT inferences...")
    work = list(range(n))
    lock = threading.Lock()
    lat2 = []

    def worker():
        while True:
            with lock:
                if not work:
                    return
                i = work.pop()
            cid, img = frames[i]
            s = time.time()
            det.detect(cid, FrameSnapshot(cid, img, s, i))
            with lock:
                lat2.append(time.time() - s)

    cpu0 = proc.cpu_times()
    t0 = time.time()
    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    par_wall = time.time() - t0
    cpu1 = proc.cpu_times()
    par_cpu = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)

    print(f"  wall            {par_wall:.1f}s")
    print(f"  latency         {_stats(lat2)}s")
    print(f"  throughput      {n / par_wall:.3f} passes/s")
    print(f"  CPU-seconds     {par_cpu:.1f}s total, {par_cpu / n:.2f}s per pass")
    print(f"  cores busy      {par_cpu / par_wall:.2f} "
          f"of {logical} logical ({100 * par_cpu / par_wall / logical:.0f}%)")

    print()
    print("=" * 78)
    print("1 vs 2 CONCURRENT")
    print("=" * 78)
    gain = (n / par_wall) / (n / serial_wall) if serial_wall else 0
    print(f"  throughput   {n / serial_wall:.3f}/s serial "
          f"-> {n / par_wall:.3f}/s with 2   ({gain:.2f}x)")
    print(f"  latency      {statistics.mean(lat):.2f}s -> "
          f"{statistics.mean(lat2):.2f}s per pass "
          f"({statistics.mean(lat2) / statistics.mean(lat):.2f}x)")
    print()
    if gain >= 1.25:
        print(f"  A second worker buys {100 * (gain - 1):.0f}% more throughput.")
        print("  Worth considering for Step 7 -- but per-pass latency rises, and")
        print("  a doorway cares about latency, not throughput.")
    else:
        print(f"  A second worker buys only {100 * (gain - 1):.0f}% -- confirms the")
        print("  earlier finding. One YOLO pass already uses the cores it needs;")
        print("  the fix for doorway coverage is not more workers.")
    return 0


# ---------------------------------------------------------------------------
# schedsim
# ---------------------------------------------------------------------------
def cmd_schedsim(args) -> int:
    """What the REAL measured costs do to the scheduler, deterministically.

    The fairness that was validated live used a stub sleeping 3.4s / 2.2s. The
    real costs turned out to be 3.406s / 2.632s -- the doorway estimate was
    almost exact, the room one was 20% low. This replays the scheduler with the
    real numbers, and with alternative priorities, WITHOUT changing production.
    """
    from app.cctv_v2.capture.grabber import CameraGrabber
    from app.cctv_v2.config.cameras import role_for
    from app.cctv_v2.scheduler.loop import InferenceScheduler

    class Clock:
        def __init__(self):
            self.t = 1000.0

        def __call__(self):
            return self.t

        def advance(self, dt):
            self.t += dt

    def run(priorities, horizon=1800.0):
        import app.cctv_v2.config.profiles as P
        clock = Clock()
        grabbers = {cid: CameraGrabber(cid, f"sim://{cid}") for cid in (57, 58, 59, 60)}
        served = defaultdict(int)
        gaps = defaultdict(list)
        last = {}

        def process(cid, snap):
            role = role_for(cid)
            if cid in last:
                gaps[cid].append(clock.t - last[cid])
            last[cid] = clock.t
            served[cid] += 1
            clock.advance(REAL_COST[role])

        sched = InferenceScheduler(grabbers.values(), process=process, clock=clock)

        orig = {"doorway": P.DOORWAY.priority, "room": P.ROOM.priority}
        object.__setattr__(P.DOORWAY, "priority", priorities["doorway"])
        object.__setattr__(P.ROOM, "priority", priorities["room"])
        try:
            start = clock.t
            while clock.t - start < horizon:
                for cid in grabbers:
                    grabbers[cid].publish("f", timestamp=clock.t)
                if sched.run_once() is None:
                    clock.advance(0.05)
        finally:
            object.__setattr__(P.DOORWAY, "priority", orig["doorway"])
            object.__setattr__(P.ROOM, "priority", orig["room"])
        return served, gaps, horizon

    print("=" * 78)
    print("SCHEDULER WITH REAL MEASURED COSTS")
    print(f"  doorway pass {REAL_COST['doorway']:.3f}s   "
          f"room pass {REAL_COST['room']:.3f}s   (measured Step 6)")
    print("  simulation only -- production priorities are NOT changed")
    print("=" * 78)

    for label, pri in (("current  3:1", {"doorway": 3, "room": 1}),
                       ("          4:1", {"doorway": 4, "room": 1}),
                       ("          6:1", {"doorway": 6, "room": 1}),
                       ("          2:1", {"doorway": 2, "room": 1})):
        served, gaps, horizon = run(pri)
        total = sum(served.values()) or 1
        print()
        print(f"{label}")
        print(f"  {'cam':>5}{'role':>9}{'served':>8}{'share':>8}"
              f"{'mean interval':>15}{'p95':>9}{'max':>9}")
        for cid in sorted(served):
            g = gaps[cid]
            print(f"  {cid:>5}{role_for(cid):>9}{served[cid]:>8}"
                  f"{100 * served[cid] / total:>7.1f}%"
                  f"{(statistics.mean(g) if g else 0):>15.2f}s"
                  f"{_pct(g, 0.95):>8.2f}s{(max(g) if g else 0):>8.2f}s")
        dw = [statistics.mean(gaps[c]) for c in (57, 58) if gaps[c]]
        if dw:
            print(f"  a 2s transit is caught roughly "
                  f"{100 * min(1.0, 2.0 / statistics.mean(dw)):.0f}% of the time")
    return 0




# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def cmd_report(args) -> int:
    """Re-report from a saved analysis, optionally against a human transit count.

    WHY THIS IS SEPARATE FROM `analyse`
    -----------------------------------
    `analyse` finds transits BY DETECTING THEM. That makes its transit count a
    detected-transit count, not a real one -- a transit YOLO missed in every
    single frame leaves no trace in the data at all, so dividing by it would
    silently assume a 100% detector and report the sampling rate as if it were
    the capture rate.

    So two DIFFERENT numbers are reported and never merged:

      DETECTION capture   detected transits / REAL transits.
                          Needs a human count (--real). Without it the
                          denominator is unknown and no rate is claimed.

      SAMPLING capture    of the transits YOLO can see when it looks at every
                          frame, what fraction would a scheduler sampling every
                          N seconds land inside? Simulated over many phases.
                          This is a THEORETICAL PROBABILITY, not a measurement.

    The end-to-end rate is the product, and it is only knowable when a person
    says how many times they actually walked.
    """
    global CLIP_DIR
    if getattr(args, "dir", None):
        CLIP_DIR = REPO / "data" / args.dir
    path = CLIP_DIR / "analysis.json"
    if not path.exists():
        print(f"no analysis at {path}; run `analyse` first")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))

    real = {}
    if args.real:
        for part in args.real.split(","):
            cid, _, n = part.partition("=")
            real[int(cid.strip())] = int(n)

    print("=" * 78)
    print("DOORWAY TRANSIT REPORT")
    print("=" * 78)

    all_detected = 0
    all_real = 0
    for cid_s in sorted(data, key=int):
        cid = int(cid_s)
        rows = data[cid_s]
        trs = _transits(rows)
        withp = [r for r in rows if r["n"] > 0]
        confs = [c for r in rows for c in r["confs"]]
        durations = [t["end"] - t["start"] for t in trs]

        print()
        print(f"CAMERA {cid}")
        print(f"  frames analysed         {len(rows)}  (every recorded frame)")
        print(f"  frames with a person    {len(withp)}"
              f"  ({100 * len(withp) / len(rows) if rows else 0:.0f}%)")
        print(f"  DETECTED transits       {len(trs)}")
        all_detected += len(trs)

        if cid in real:
            n = real[cid]
            all_real += n
            missed = max(0, n - len(trs))
            print(f"  REAL transits (stated)  {n}")
            print(f"  completely missed       {missed}")
            print(f"  DETECTION capture rate  {100 * min(len(trs), n) / n:.0f}%"
                  f"   <-- MEASURED")
        else:
            print("  REAL transits           UNKNOWN -- pass --real "
                  f"{cid}=N to get a measured capture rate")

        if confs:
            print(f"  confidence              {_stats(confs)}")
        if durations:
            print(f"  transit duration        {_stats(durations)}s")
            per = [t["frames"] for t in trs]
            print(f"  detections per transit  {_stats(per, '{:.1f}')}")
        gaps, prev = [], None
        for r in withp:
            if prev is not None:
                gaps.append(r["t"] - prev)
            prev = r["t"]
        if gaps:
            print(f"  detection-to-detection  {_stats(gaps)}s")

        if trs:
            print()
            print("  THEORETICAL sampling probability over these detected")
            print("  transits (NOT a measured capture rate):")
            print(f"    {'interval':>10}{'caught':>9}{'missed':>9}")
            for iv in (2.0, 5.0, 6.9, 9.4):
                caught, miss = _missed(trs, iv)
                flag = "  <-- current 3-worker doorway" if iv == 6.9 else ""
                print(f"    {iv:>9.1f}s{100 * caught:>8.0f}%{100 * miss:>8.0f}%{flag}")

            short = [t for t in trs if (t["end"] - t["start"]) < 6.9]
            print(f"  transits shorter than the 6.9s interval: "
                  f"{len(short)}/{len(trs)}"
                  f"  (these are the ones sampling can miss)")
            lowc = [t for t in trs
                    if t["confs"] and statistics.mean(t["confs"]) < 0.4]
            print(f"  transits with mean confidence below 0.40: {len(lowc)}/{len(trs)}")

    print()
    print("=" * 78)
    print("COMBINED 57 + 58")
    print("=" * 78)
    print(f"  detected transits   {all_detected}")
    if all_real:
        print(f"  real transits       {all_real}")
        print(f"  DETECTION capture   "
              f"{100 * min(all_detected, all_real) / all_real:.0f}%   <-- MEASURED")
    else:
        print("  real transits       UNKNOWN")
        print("  No capture rate is claimed from an unlabelled recording.")
        print("  Re-run with:  report --real 57=N,58=M")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="capture doorway frames while somebody walks")
    r.add_argument("--seconds", type=float, default=90.0)
    r.add_argument("--fps", type=float, default=3.0)
    r.add_argument("--out", type=str, default=None,
                   help="output dir (default data/cctv_v2_step65)")
    r.set_defaults(func=cmd_record)

    a = sub.add_parser("analyse", help="run YOLO on every recorded frame")
    a.add_argument("--dir", type=str, default=None, help="clip dir")
    a.add_argument("--workers", type=int, default=3,
                   help="independent YOLO instances (3 measured best here)")
    a.set_defaults(func=cmd_analyse)

    c = sub.add_parser("cpubench", help="CPU per pass, and 1 vs 2 concurrent")
    c.add_argument("--passes", type=int, default=12)
    c.set_defaults(func=cmd_cpubench)

    rp = sub.add_parser("report", help="re-report a saved analysis, with ground truth")
    rp.add_argument("--dir", type=str, default=None, help="clip dir")
    rp.add_argument("--real", type=str, default=None,
                    help="human transit count, e.g. 57=12,58=11")
    rp.set_defaults(func=cmd_report)

    s = sub.add_parser("schedsim", help="scheduler behaviour at real costs")
    s.set_defaults(func=cmd_schedsim)

    args = ap.parse_args()
    import logging
    logging.disable(logging.INFO)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
