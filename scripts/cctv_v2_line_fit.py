"""Choose a crossing line from measured foot points, not from a screenshot.

MANUAL / ANALYSIS ONLY. Changes nothing; prints a recommendation.

WHY THIS EXISTS
---------------
Camera 57's line was carried over from V1 at 0.35 without checking it against
V2's reference point. V1 measured a different point on the body; V2 uses the
FOOT point, and on camera 57 every foot sits low in frame. The result was 24
tracks and zero side changes over 50 minutes -- not a near miss, a line nobody
could ever cross.

Reading a replacement off a still frame would repeat the mistake with a
different number. So this reads the actual distribution of foot points that the
detector produced on that camera and picks a line from it.

HOW THE LINE IS CHOSEN
----------------------
A usable line has to sit INSIDE the range people actually traverse, with margin
on both sides -- not at the edge of it. A line at the extreme is crossed by
nobody; a line in the middle of where people merely stand is crossed constantly
by noise.

So: take the observed foot range, and prefer the position that maximises the
number of TRACKS that have points on both sides of it. That is the definition of
a crossable line -- it is the quantity the crossing detector actually needs -- and
it is measured per track rather than per detection, so one person loitering
cannot outvote ten people walking through.

The margin check then rejects a winner that only just works.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "Attendance Management" / "backend"
for p in (str(BACKEND), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="replay_*.json produced by cctv_v2_replay")
    ap.add_argument("--camera", type=int, required=True)
    ap.add_argument("--interval", type=float, default=1.0,
                    help="sampling interval to build tracks at")
    ap.add_argument("--height", type=float, default=1080.0)
    ap.add_argument("--keep-clipped", action="store_true",
                    help="include boxes touching the frame bottom (default: drop)")
    args = ap.parse_args()

    import logging
    logging.disable(logging.INFO)

    from app.cctv_v2.config.geometry import crossing_line
    from app.cctv_v2.pipeline.detect import DetectionResult, PersonDetection
    from app.cctv_v2.pipeline.track import CameraTracker

    path = Path(args.cache)
    if not path.is_absolute():
        path = REPO / args.cache
    cached = json.loads(path.read_text(encoding="utf-8"))

    # Rebuild tracks so foot points can be grouped BY PERSON. A per-detection
    # histogram would let one stationary person dominate the whole distribution.
    tracker = CameraTracker(args.camera)
    per_track: dict[int, list[float]] = defaultdict(list)
    clipped_dropped = [0]
    last_t, seq = None, 0
    for row in cached:
        if last_t is not None and (row["t"] - last_t) < args.interval - 1e-9:
            continue
        last_t = row["t"]
        seq += 1
        dets = tuple(
            PersonDetection(camera_id=args.camera, frame_timestamp=row["t"],
                            frame_sequence=seq, bbox=tuple(b), confidence=c)
            for b, c in zip(row["boxes"], row["confs"])
        )
        seen = tracker.update(DetectionResult(
            camera_id=args.camera, role="doorway", frame_sequence=seq,
            frame_timestamp=row["t"], detections=dets,
            inference_ms=0.0, frame_age_ms=0.0, imgsz=640, conf=0.15))
        for t in seen:
            foot = t.bbox[3]
            # A box touching the frame bottom is CLIPPED: the person's feet are
            # out of shot and `foot` is the frame edge, not a body part.
            # Measured on camera 57, 30% of boxes are clipped and they pile up
            # into a false peak that makes a line near the edge look ideal.
            if not args.keep_clipped and foot >= args.height - 2.0:
                clipped_dropped[0] += 1
                continue
            per_track[t.track_id].append(foot / args.height)

    feet = [f for v in per_track.values() for f in v]
    if not feet:
        print("no foot points -- nothing detected in this window")
        return 1

    print("=" * 78)
    print(f"FOOT-POINT DISTRIBUTION  camera {args.camera}  "
          f"({len(feet)} points, {len(per_track)} tracks)")
    print("=" * 78)
    print(f"  min {min(feet):.3f}   median {statistics.median(feet):.3f}   "
          f"max {max(feet):.3f}   mean {statistics.mean(feet):.3f}")
    if clipped_dropped[0]:
        print(f"  dropped {clipped_dropped[0]} CLIPPED boxes (foot at the frame "
              f"edge -- feet out of shot, not a real position)")

    # Histogram
    bins = 20
    counts = [0] * bins
    for f in feet:
        counts[min(bins - 1, int(f * bins))] += 1
    peak = max(counts) or 1
    current = crossing_line(args.camera)
    cur_pos = current.position if current else None

    print()
    print(f"  {'y':>6}  {'n':>5}")
    for i, c in enumerate(counts):
        lo = i / bins
        bar = "#" * int(40 * c / peak)
        mark = ""
        if cur_pos is not None and lo <= cur_pos < lo + 1 / bins:
            mark = "   <== CURRENT LINE " + f"({cur_pos:.2f})"
        print(f"  {lo:>6.2f}  {c:>5}  {bar}{mark}")

    # Which line position separates the most TRACKS?
    print()
    print("=" * 78)
    print("CROSSABILITY  (tracks with foot points on BOTH sides of a line)")
    print("=" * 78)
    print(f"  {'line':>6}{'tracks split':>14}{'min margin':>12}")
    best = []
    for i in range(1, 40):
        pos = i / 40.0
        split, margins = 0, []
        for pts in per_track.values():
            above = [p for p in pts if p < pos]
            below = [p for p in pts if p >= pos]
            if above and below:
                split += 1
                margins.append(min(pos - max(above), min(below) - pos))
        if split:
            best.append((split, pos, min(margins) if margins else 0.0))
    if not best:
        print("  NO line position is crossed by ANY track in this window.")
        print("  Every track stays entirely on one side -- people are not")
        print("  traversing this camera's view, or the window has no transits.")
        return 0

    best.sort(key=lambda x: (-x[0], -x[2]))
    for split, pos, margin in best[:12]:
        flag = "  <== best" if (split, pos, margin) == best[0] else ""
        print(f"  {pos:>6.3f}{split:>14}{margin:>12.3f}{flag}")

    top_split, top_pos, top_margin = best[0]
    if cur_pos is not None:
        cur_split = sum(
            1 for pts in per_track.values()
            if any(p < cur_pos for p in pts) and any(p >= cur_pos for p in pts)
        )
    else:
        cur_split = 0

    print()
    print("=" * 78)
    print("RECOMMENDATION")
    print("=" * 78)
    print(f"  current line {cur_pos}: crossed by {cur_split}/{len(per_track)} tracks")
    print(f"  best line    {top_pos:.3f}: crossed by {top_split}/{len(per_track)} tracks")
    print(f"  margin at best line: {top_margin:.3f} "
          f"({'adequate' if top_margin > 0.02 else 'THIN -- crossings sit near the line'})")
    if top_split == 0:
        print("  -> No line works on this footage. Do not change the config from this.")
    elif cur_split == 0:
        print(f"  -> The current line is UNCROSSABLE here. {top_pos:.3f} is supported")
        print(f"     by {top_split} tracks in this window.")
    else:
        print(f"  -> Current line already works for {cur_split} tracks; "
              f"change only if {top_split} is materially better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
