"""Replay recorded frames through track+crossing at different sampling rates.

MANUAL / INTEGRATION ONLY.

THE QUESTION THIS ANSWERS
-------------------------
The live pipeline has produced zero transits. That has two very different
explanations and no live run can separate them, because the live run only ever
offers one sampling rate:

    A. the crossing logic does not work
    B. the logic is fine and 6.5s sampling never sees the same person twice

So: take REAL recorded footage of people actually walking through, run YOLO over
every frame ONCE, then feed the identical detections through the identical
tracker and crossing detector at several sampling intervals. Everything except
the interval is held constant, so any difference in transit count is caused by
the interval and nothing else.

If dense replay produces transits and sparse replay does not, the logic works
and the sampling is the bottleneck. If dense replay produces none either, the
logic is broken and no amount of throughput will help.

Inference is cached to disk, so the sweep costs one YOLO pass per frame no
matter how many intervals are compared.

Usage:
    python scripts/cctv_v2_replay.py --dir cctv_v2_lunch --camera 58 \
        --start 500 --seconds 200
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Attendance Management" / "backend"
for p in (str(BACKEND), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="cctv_v2_lunch")
    ap.add_argument("--camera", type=int, default=58)
    ap.add_argument("--start", type=float, default=0.0, help="t_rel seconds")
    ap.add_argument("--seconds", type=float, default=200.0)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    import logging
    logging.disable(logging.INFO)
    import cv2

    from app.cctv_v2.capture.grabber import FrameSnapshot
    from app.cctv_v2.pipeline.crossing import CrossingDetector
    from app.cctv_v2.pipeline.detect import PersonDetector
    from app.cctv_v2.pipeline.track import CameraTracker

    clip = REPO / "data" / args.dir
    manifest = json.loads((clip / "manifest.json").read_text(encoding="utf-8"))
    frames = sorted(
        (m for m in manifest
         if m["camera_id"] == args.camera
         and args.start <= m["t_rel"] < args.start + args.seconds),
        key=lambda m: m["t_rel"],
    )
    if not frames:
        print(f"no frames for camera {args.camera} in that window")
        return 1

    cache_path = clip / f"replay_{args.camera}_{int(args.start)}_{int(args.seconds)}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"using cached inference for {len(cached)} frames")
    else:
        print(f"running YOLO on {len(frames)} frames "
              f"({args.workers} independent models)...")
        pool = [PersonDetector() for _ in range(args.workers)]
        out = [None] * len(frames)
        nxt = [0]
        lock = threading.Lock()

        def worker(det):
            while True:
                with lock:
                    if nxt[0] >= len(frames):
                        return
                    i = nxt[0]
                    nxt[0] += 1
                m = frames[i]
                img = cv2.imread(str(clip / m["file"]))
                if img is None:
                    continue
                r = det.detect(args.camera,
                               FrameSnapshot(args.camera, img,
                                             m["timestamp"], m["sequence"]))
                out[i] = {
                    "t": m["t_rel"], "seq": m["sequence"],
                    "boxes": [list(d.bbox) for d in r.detections],
                    "confs": [d.confidence for d in r.detections],
                }
                with lock:
                    if (i + 1) % 25 == 0:
                        print(f"  ...{i + 1}/{len(frames)}")

        ts = [threading.Thread(target=worker, args=(d,)) for d in pool]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        cached = [c for c in out if c is not None]
        cache_path.write_text(json.dumps(cached), encoding="utf-8")
        print(f"cached to {cache_path.name}")

    have = [c for c in cached if c["boxes"]]
    print()
    print("=" * 78)
    print(f"REPLAY  camera {args.camera}  window t+{args.start:.0f}s "
          f"for {args.seconds:.0f}s")
    print("=" * 78)
    print(f"  frames analysed          {len(cached)}")
    print(f"  frames with a person     {len(have)} "
          f"({100 * len(have) / max(1, len(cached)):.0f}%)")
    print(f"  recorded frame spacing   "
          f"{statistics.median([b['t'] - a['t'] for a, b in zip(cached, cached[1:])]):.2f}s")

    from app.cctv_v2.pipeline.detect import DetectionResult, PersonDetection
    from app.cctv_v2.pipeline.track import TrackState

    INTERVALS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

    def replay(interval):
        """One sampling schedule over the SAME cached detections.

        Only which timestamps reach the tracker changes. Same footage, same
        model, same boxes, same tracker, same line, same rules -- so any
        difference in the result is caused by the interval and nothing else.
        """
        tracker = CameraTracker(args.camera)
        crossing = CrossingDetector(args.camera)
        last_t, passes, seq, dets_total = None, 0, 0, 0
        for row in cached:
            if last_t is not None and (row["t"] - last_t) < interval - 1e-9:
                continue
            last_t = row["t"]
            passes += 1
            seq += 1
            dets_total += len(row["boxes"])
            dets = tuple(
                PersonDetection(camera_id=args.camera, frame_timestamp=row["t"],
                                frame_sequence=seq, bbox=tuple(b), confidence=c)
                for b, c in zip(row["boxes"], row["confs"])
            )
            seen = tracker.update(DetectionResult(
                camera_id=args.camera, role="doorway", frame_sequence=seq,
                frame_timestamp=row["t"], detections=dets,
                inference_ms=0.0, frame_age_ms=0.0, imgsz=640, conf=0.15))
            crossing.update(seen, row["t"], seq)
        return tracker, crossing, passes, dets_total

    print()
    print("=" * 78)
    print("INTERVAL COMPARISON  (same footage, same detections, same logic)")
    print("=" * 78)
    print(f"  {'interval':>9}{'passes':>8}{'person det':>12}{'tracks':>8}"
          f"{'confirmed':>11}{'crossings':>11}{'IN':>4}{'OUT':>5}")
    runs = {}
    for iv in INTERVALS:
        tracker, crossing, passes, dets_total = replay(iv)
        runs[iv] = (tracker, crossing, passes, dets_total)
        hits = tracker.hit_histogram()
        confirmed = sum(1 for h in hits if h >= tracker.min_hits)
        s = crossing.summary()
        print(f"  {iv:>8.1f}s{passes:>8}{dets_total:>12}{len(hits):>8}"
              f"{confirmed:>11}{s['events']:>11}{s['people_in']:>4}{s['people_out']:>5}")

    print()
    print("=" * 78)
    print("TRACKING BEHAVIOUR")
    print("=" * 78)
    print(f"  {'interval':>9}{'tracks':>8}{'mean hits':>11}{'median':>8}{'max':>6}"
          f"{'>=2 obs':>9}{'confirmed':>11}{'rej unconf':>12}")
    for iv in INTERVALS:
        tracker, crossing, _, _ = runs[iv]
        hits = tracker.hit_histogram()
        if not hits:
            print(f"  {iv:>8.1f}s{0:>8}          -       -     -        -          -"
                  f"{crossing.rejected_unconfirmed:>12}")
            continue
        multi = sum(1 for h in hits if h >= 2)
        confirmed = sum(1 for h in hits if h >= tracker.min_hits)
        print(f"  {iv:>8.1f}s{len(hits):>8}{statistics.mean(hits):>11.1f}"
              f"{statistics.median(hits):>8.1f}{max(hits):>6}"
              f"{100 * multi / len(hits):>8.0f}%{100 * confirmed / len(hits):>10.0f}%"
              f"{crossing.rejected_unconfirmed:>12}")

    print()
    print("=" * 78)
    print("CROSSING BEHAVIOUR  (does each event make physical sense?)")
    print("=" * 78)
    for iv in INTERVALS:
        tracker, crossing, _, _ = runs[iv]
        evs = crossing.events
        if not evs:
            print(f"  {iv:>4.1f}s: no events")
            continue
        by_track = defaultdict(list)
        for e in evs:
            by_track[e.track_id].append(e)
        dupes = {k: v for k, v in by_track.items() if len(v) > 1}
        thin = [e for e in evs if e.track_hits < 2]
        print(f"  {iv:>4.1f}s: {len(evs)} events, {len(by_track)} distinct tracks, "
              f"{len(dupes)} track(s) firing twice+, "
              f"{len(thin)} on a 1-hit track"
              f"{'  <-- SHOULD BE 0' if thin else ''}")
        for e in evs[:6]:
            print(f"        track {e.track_id:>3}  {e.direction.value:<10} "
                  f"t={e.timestamp:7.1f}s  hits={e.track_hits}  "
                  f"travel={e.travel:.3f}  identity={e.identity}")
        if len(evs) > 6:
            print(f"        ... and {len(evs) - 6} more")

    print()
    print("=" * 78)
    print("COST OF EACH INTERVAL  (this camera alone)")
    print("=" * 78)
    print("  Measured: a doorway YOLO pass costs ~4.1s of wall time and the")
    print("  3-worker pool sustains ~0.68-0.78 passes/s across ALL FOUR cameras.")
    print(f"  {'interval':>9}{'passes/hour':>13}{'passes/s':>11}"
          f"{'share of pool':>15}")
    for iv in INTERVALS:
        per_s = 1.0 / iv
        print(f"  {iv:>8.1f}s{3600 / iv:>13.0f}{per_s:>11.2f}"
              f"{100 * per_s / 0.73:>14.0f}%")
    print("  >100% means this ONE camera would need the entire pool and still")
    print("  not keep up, leaving nothing for the other three.")

    print()
    dense = runs[0.5][1].summary()["events"]
    live_like = runs[6.0][1].summary()["events"]
    print(f"  dense (0.5s): {dense} crossings     live-like (6.0s): {live_like}")
    if dense > 0 and live_like == 0:
        print("  -> Logic works; the sampling interval is the measured bottleneck.")
    elif dense == 0:
        print("  -> Nothing even at 0.5s. The cause is NOT sampling -- look at the")
        print("     line position, inside_side, the detector, or this footage.")
    else:
        print("  -> Both produce crossings; compare the counts above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
