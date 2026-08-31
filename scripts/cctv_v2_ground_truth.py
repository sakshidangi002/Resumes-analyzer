"""Establish how many people REALLY walked through a doorway, without asking YOLO.

WHY THIS CANNOT USE YOLO
------------------------
The question is "what fraction of real transits did YOLO catch?". Counting the
real ones with YOLO would make the answer 100% by construction: a transit the
detector missed in every frame leaves no trace, so it would never enter the
denominator. The ground truth has to come from somewhere the detector cannot
influence.

So this uses frame differencing -- pure pixel arithmetic, no model. A person
crossing a doorway moves a large, connected block of pixels; that is visible to
subtraction whether or not a person detector fires on it. Motion events are then
rendered as contact sheets for a human (or a vision model) to look at and count.

WHAT MOTION CAN AND CANNOT ESTABLISH
------------------------------------
Motion is a CANDIDATE FINDER, not the ground truth itself. It over-triggers on
doors swinging, lights changing and camera auto-exposure, and it could in
principle miss someone moving very slowly at the far edge of frame. The count
that matters is the one a person makes from the contact sheets; motion only
decides which frames are worth looking at, which turns 15,000 frames into a
few dozen sheets.

The threshold is deliberately LOW. Over-triggering costs a few extra sheets to
look through; under-triggering silently deflates the real transit count and
would inflate the capture rate -- the exact error this whole exercise exists to
avoid.

    motion    find candidate motion events   -> events.json
    sheets    render them as contact sheets  -> sheets/*.jpg for visual counting
    compare   match a stated real count against live pipeline detections

Usage:
    python scripts/cctv_v2_ground_truth.py motion --dir cctv_v2_lunch
    python scripts/cctv_v2_ground_truth.py sheets --dir cctv_v2_lunch --camera 57
    python scripts/cctv_v2_ground_truth.py compare --dir cctv_v2_lunch \\
        --detections lunch_detections.jsonl --real 57=31,58=28
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

# Fraction of pixels that must change for a frame to count as "something moved".
# Low on purpose -- see the module docstring.
MOTION_PIXEL_FRAC = 0.004
# Per-pixel intensity change that counts as changed.
MOTION_DELTA = 22
# Frames this far apart in time belong to different motion events.
EVENT_GAP_SEC = 2.0
# Ignore events shorter than this; a single flagged frame is usually noise.
MIN_EVENT_FRAMES = 2


def _clip(name):
    return REPO / "data" / name


def _pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p))]


def _stats(v, fmt="{:.2f}"):
    if not v:
        return "no samples"
    return (f"min {fmt.format(min(v))}  mean {fmt.format(statistics.mean(v))}"
            f"  median {fmt.format(statistics.median(v))}"
            f"  max {fmt.format(max(v))}")


# ---------------------------------------------------------------------------
# motion
# ---------------------------------------------------------------------------
def cmd_motion(args) -> int:
    import cv2
    import numpy as np

    clip = _clip(args.dir)
    manifest = json.loads((clip / "manifest.json").read_text(encoding="utf-8"))
    by_cam = defaultdict(list)
    for m in manifest:
        by_cam[m["camera_id"]].append(m)

    all_events = {}
    for cid in sorted(by_cam):
        frames = sorted(by_cam[cid], key=lambda x: x["t_rel"])
        print(f"camera {cid}: differencing {len(frames)} frames...")

        prev = None
        flagged = []
        scores = []
        for m in frames:
            img = cv2.imread(str(clip / m["file"]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            small = cv2.resize(img, (240, 270))
            small = cv2.GaussianBlur(small, (5, 5), 0)
            if prev is not None:
                diff = cv2.absdiff(small, prev)
                frac = float(np.count_nonzero(diff > MOTION_DELTA)) / diff.size
                scores.append(frac)
                if frac >= MOTION_PIXEL_FRAC:
                    flagged.append((m, frac))
            prev = small

        # Group flagged frames into events.
        events = []
        cur = None
        for m, frac in flagged:
            if cur and m["t_rel"] - cur["end"] <= EVENT_GAP_SEC:
                cur["end"] = m["t_rel"]
                cur["frames"].append(m["file"])
                cur["peak"] = max(cur["peak"], frac)
            else:
                if cur and len(cur["frames"]) >= MIN_EVENT_FRAMES:
                    events.append(cur)
                cur = {"start": m["t_rel"], "end": m["t_rel"],
                       "frames": [m["file"]], "peak": frac}
        if cur and len(cur["frames"]) >= MIN_EVENT_FRAMES:
            events.append(cur)

        all_events[str(cid)] = events
        durs = [e["end"] - e["start"] for e in events]
        print(f"  motion score      {_stats(scores, '{:.4f}')}")
        print(f"  frames flagged    {len(flagged)}/{len(frames)}"
              f"  ({100 * len(flagged) / max(1, len(frames)):.0f}%)")
        print(f"  motion events     {len(events)}")
        if durs:
            print(f"  event duration    {_stats(durs)}s")

    (clip / "events.json").write_text(json.dumps(all_events), encoding="utf-8")
    print()
    print(f"wrote {clip / 'events.json'}")
    print("NOTE: these are motion CANDIDATES, not a transit count. Render sheets")
    print("      and count people visually before quoting any number.")
    return 0


# ---------------------------------------------------------------------------
# sheets
# ---------------------------------------------------------------------------
def cmd_sheets(args) -> int:
    """Render motion events as labelled contact sheets for visual counting."""
    import cv2
    import numpy as np

    clip = _clip(args.dir)
    events = json.loads((clip / "events.json").read_text(encoding="utf-8"))
    cam = str(args.camera)
    if cam not in events:
        print(f"no events for camera {cam}; have {sorted(events)}")
        return 1

    out = clip / "sheets"
    out.mkdir(exist_ok=True)
    for old in out.glob(f"{cam}_*.jpg"):
        old.unlink()

    tw, th = 200, 225
    cols, rows = args.cols, args.rows
    per_sheet = cols * rows

    # One representative strip per event, so every event is visible and a long
    # event does not crowd out ten short ones.
    tiles = []
    for idx, ev in enumerate(events[cam], 1):
        picks = ev["frames"]
        if len(picks) > args.per_event:
            step = len(picks) / args.per_event
            picks = [picks[int(i * step)] for i in range(args.per_event)]
        for j, f in enumerate(picks):
            tiles.append((idx, ev["start"], f, j == 0))

    print(f"camera {cam}: {len(events[cam])} events -> {len(tiles)} tiles")
    sheets = 0
    for s in range(0, len(tiles), per_sheet):
        batch = tiles[s:s + per_sheet]
        canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
        for k, (evid, t0, fname, first) in enumerate(batch):
            img = cv2.imread(str(clip / fname))
            if img is None:
                continue
            tile = cv2.resize(img, (tw, th))
            # Solid strip behind the label. Without it the text lands on
            # whatever the corridor happens to look like and is unreadable in
            # exactly the frames that matter.
            colour = (0, 255, 255) if first else (230, 230, 230)
            cv2.rectangle(tile, (0, 0), (tw, 20), (0, 0, 0), -1)
            cv2.putText(tile, f"E{evid}  t={t0:.0f}s", (4, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
            if first:
                cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), (0, 255, 255), 2)
            r, c = divmod(k, cols)
            canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
        path = out / f"{cam}_sheet{sheets:02d}.jpg"
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        sheets += 1

    print(f"wrote {sheets} sheets to {out}")
    print("Each yellow-bordered tile starts a new motion event (E<n>).")
    print("Count DISTINCT PEOPLE crossing, not events -- one event may hold two")
    print("people, and one person may span two events if they paused.")
    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def cmd_compare(args) -> int:
    """Set a stated real transit count against what the LIVE pipeline detected.

    The two numbers answer different questions and are never merged:

      DETECTION capture   detected transits / REAL transits.  MEASURED, and only
                          computable because the real count came from outside
                          the detector.

      SAMPLING capture    of transits the detector CAN see, what fraction would
                          a sampler at interval N land inside.  THEORETICAL.
    """
    clip = _clip(args.dir)
    real = {}
    if args.real:
        for part in args.real.split(","):
            cid, _, n = part.partition("=")
            real[int(cid.strip())] = int(n)

    path = Path(args.detections)
    if not path.is_absolute():
        path = REPO / args.detections
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    by_cam = defaultdict(list)
    for r in rows:
        by_cam[r["camera_id"]].append(r)

    print("=" * 78)
    print("LIVE PIPELINE vs REAL TRANSITS")
    print("=" * 78)

    tot_real = tot_det = 0
    for cid in sorted(by_cam):
        rs = sorted(by_cam[cid], key=lambda x: x["t"])
        passes = len(rs)
        hits = [r for r in rs if r["n"] > 0]
        confs = [c for r in rs for c in r["confs"]]
        gaps = [b["t"] - a["t"] for a, b in zip(rs, rs[1:])]

        # Consecutive passes with people, within one sampling gap, are one
        # transit as far as the pipeline can tell.
        groups, cur = [], None
        for r in hits:
            if cur and r["t"] - cur["end"] <= args.link_sec:
                cur["end"] = r["t"]
                cur["passes"] += 1
                cur["peak"] = max(cur["peak"], r["n"])
                cur["confs"].extend(r["confs"])
            else:
                if cur:
                    groups.append(cur)
                cur = {"start": r["t"], "end": r["t"], "passes": 1,
                       "peak": r["n"], "confs": list(r["confs"])}
        if cur:
            groups.append(cur)

        print()
        print(f"CAMERA {cid}")
        print(f"  YOLO passes             {passes}")
        print(f"  passes with a person    {len(hits)}"
              f"  ({100 * len(hits) / passes if passes else 0:.0f}%)")
        print(f"  sampling interval       "
              f"{statistics.mean(gaps) if gaps else 0:.2f}s mean, "
              f"{_pct(gaps, 0.95):.2f}s p95")
        print(f"  DETECTED transits       {len(groups)}")
        tot_det += len(groups)
        if confs:
            print(f"  detection confidence    {_stats(confs, '{:.3f}')}")
        if groups:
            print(f"  detections per transit  "
                  f"{_stats([g['passes'] for g in groups], '{:.1f}')}")
            print(f"  people seen at once     max {max(g['peak'] for g in groups)}")

        if cid in real:
            n = real[cid]
            tot_real += n
            det = min(len(groups), n)
            print(f"  REAL transits (counted) {n}")
            print(f"  completely missed       {max(0, n - len(groups))}")
            print(f"  CAPTURE RATE            {100 * det / n:.0f}%   <-- MEASURED")
            if len(groups) > n:
                print(f"  NOTE: more detected groups ({len(groups)}) than real "
                      f"transits ({n}); one transit probably split across "
                      f"sampling gaps. Capping at {n}.")
        else:
            print("  REAL transits           NOT SUPPLIED -- no capture rate claimed")

    print()
    print("=" * 78)
    print("COMBINED")
    print("=" * 78)
    print(f"  detected transits  {tot_det}")
    if tot_real:
        print(f"  real transits      {tot_real}")
        print(f"  CAPTURE RATE       "
              f"{100 * min(tot_det, tot_real) / tot_real:.0f}%   <-- MEASURED")
    else:
        print("  real transits      NOT SUPPLIED")
        print("  No capture rate is claimed from an unlabelled recording.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("motion", help="find motion candidates (no YOLO)")
    m.add_argument("--dir", default="cctv_v2_lunch")
    m.set_defaults(func=cmd_motion)

    s = sub.add_parser("sheets", help="render contact sheets for visual counting")
    s.add_argument("--dir", default="cctv_v2_lunch")
    s.add_argument("--camera", type=int, required=True)
    s.add_argument("--cols", type=int, default=6)
    s.add_argument("--rows", type=int, default=5)
    s.add_argument("--per-event", type=int, default=3)
    s.set_defaults(func=cmd_sheets)

    c = sub.add_parser("compare", help="real count vs live pipeline detections")
    c.add_argument("--dir", default="cctv_v2_lunch")
    c.add_argument("--detections", required=True)
    c.add_argument("--real", default=None, help="e.g. 57=31,58=28")
    c.add_argument("--link-sec", type=float, default=9.0,
                   help="passes closer than this are one transit")
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
