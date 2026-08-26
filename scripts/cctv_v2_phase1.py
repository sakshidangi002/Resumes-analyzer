"""Phase 1: did the pipeline see the people who actually walked through?

MANUAL / INTEGRATION ONLY.

THE TWO NUMBERS THIS TOOL REFUSES TO CONFLATE
---------------------------------------------
    REAL transits      people who actually crossed. Comes from outside the
                       pipeline -- a human count, or motion differencing as an
                       AID. Never from YOLO.

    DETECTED transits  crossings the pipeline emitted.

    capture = detected / real

Deriving the denominator from the detector makes the answer 100% by
construction: a person the detector never saw leaves no trace and never enters
the count. That is the single error this whole exercise exists to avoid, so
`--real` must be supplied before any capture rate is printed. Without it the
tool reports INSUFFICIENT DATA and stops.

MOTION IS AN AID, NOT GROUND TRUTH
----------------------------------
Motion differencing finds candidate windows cheaply and independently of YOLO,
which makes it useful for deciding where to look. It is NOT a person count:
measured on the lunch recording, only 6 of 31 motion events on camera 58
contained a visible person -- the rest were compression and lighting noise --
while on camera 57 nearly all did, because it watches an enclosed area where
anything that moves is a person. One threshold cannot serve both, so motion
output is always labelled MOTION-DERIVED and never printed as a transit count.

Usage:
    python scripts/cctv_v2_phase1.py --dump pm_transits.jsonl --dir cctv_v2_pm
    python scripts/cctv_v2_phase1.py --dump ... --dir ... --real 57=12,58=9
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

DOORWAYS = (57, 58)


def _pct(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p))]


def _sampling_probability(duration: float, interval: float) -> float:
    """Chance a periodic sampler with random phase lands inside a window.

    P = min(1, duration / interval). THEORETICAL, and an optimistic ceiling: it
    assumes the detector sees the person on whichever pass lands in the window,
    which the measured per-pass detection rate says is not guaranteed.
    """
    if interval <= 0:
        return 1.0
    return min(1.0, duration / interval)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="transit JSONL from the pipeline")
    ap.add_argument("--dir", default=None, help="frame recording dir under data/")
    ap.add_argument("--real", default=None, help="human transit count, e.g. 57=12,58=9")
    args = ap.parse_args()

    path = Path(args.dump)
    if not path.is_absolute():
        path = REPO / args.dump
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    passes = [r for r in rows if r.get("kind") == "pass"]
    transits = [r for r in rows if r.get("kind") == "transit"]

    real = {}
    if args.real:
        for part in args.real.split(","):
            cid, _, n = part.partition("=")
            real[int(cid.strip())] = int(n)

    print("=" * 78)
    print("PHASE 1 -- DOORWAY CAPTURE")
    print("=" * 78)

    for cid in DOORWAYS:
        ps = sorted((p for p in passes if p["camera_id"] == cid), key=lambda x: x["t"])
        tr = [t for t in transits if t["camera_id"] == cid]
        if not ps:
            print(f"\nCAMERA {cid}: no passes recorded")
            continue

        ts = [p["t"] for p in ps]
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        withp = [p for p in ps if p["n"] > 0]
        dets = sum(p["n"] for p in ps)

        print()
        print(f"CAMERA {cid}")
        print(f"  YOLO passes                {len(ps)}")
        print(f"  passes seeing a person     {len(withp)} "
              f"({100 * len(withp) / len(ps):.1f}%)")
        print(f"  person detections          {dets}")
        print(f"  detections per pass        {dets / len(ps):.3f}")
        if gaps:
            print(f"  sampling interval          mean {statistics.mean(gaps):.2f}s  "
                  f"min {min(gaps):.2f}s  max {max(gaps):.2f}s  "
                  f"p95 {_pct(gaps, 0.95):.2f}s")
        print(f"  tracks seen on a pass      "
              f"{sum(p.get('tracks', 0) for p in ps)}")
        print(f"  DETECTED transits          {len(tr)}")

        if tr:
            by_track = defaultdict(list)
            for t in tr:
                by_track[t["track_id"]].append(t)
            dupes = {k: v for k, v in by_track.items() if len(v) > 1}
            print(f"    distinct tracks          {len(by_track)}")
            print(f"    tracks firing twice+     {len(dupes)}")
            print(f"    IN / OUT                 "
                  f"{sum(1 for t in tr if t['direction'] == 'PERSON_IN')} / "
                  f"{sum(1 for t in tr if t['direction'] == 'PERSON_OUT')}")
            print(f"    hits behind each event   "
                  f"{sorted(t['hits'] for t in tr)}")
            print(f"    unattributed             "
                  f"{sum(1 for t in tr if t['identity'] is None)}/{len(tr)}")
            single = [t for t in tr if t["hits"] < 2]
            print(f"    fired on a 1-hit track   {len(single)}"
                  f"{'  <-- SHOULD BE 0' if single else '  (correct)'}")
        else:
            print("    no crossing events to inspect")

        # -- capture rate -----------------------------------------------------
        if cid in real:
            n = real[cid]
            det = len(tr)
            print(f"  REAL transits (supplied)   {n}")
            print(f"  missed                     {max(0, n - det)}")
            if n > 0:
                print(f"  CAPTURE RATE               "
                      f"{100 * min(det, n) / n:.1f}%   <-- MEASURED")
            if det > n:
                print(f"  NOTE: more detected ({det}) than real ({n}) -- "
                      f"duplicates or a merged track; inspect above.")
        else:
            print("  REAL transits              NOT SUPPLIED")
            print("  CAPTURE RATE               INSUFFICIENT DATA")

    # -- motion aid ----------------------------------------------------------
    if args.dir:
        ev_path = REPO / "data" / args.dir / "events.json"
        if ev_path.exists():
            ev = json.loads(ev_path.read_text(encoding="utf-8"))
            print()
            print("=" * 78)
            print("MOTION-DERIVED CANDIDATES  (an AID, not a person count)")
            print("=" * 78)
            for cid in DOORWAYS:
                evs = ev.get(str(cid), [])
                durs = [e["end"] - e["start"] for e in evs]
                print(f"  camera {cid}: {len(evs)} motion events"
                      + (f", duration median {statistics.median(durs):.1f}s "
                         f"p95 {_pct(durs, 0.95):.1f}s" if durs else ""))
            print("  On the lunch recording only 6 of 31 camera-58 motion events")
            print("  held a visible person; on camera 57 nearly all did. These")
            print("  counts are NOT transits and must be reviewed visually.")

    # -- sampling analysis ---------------------------------------------------
    print()
    print("=" * 78)
    print("SAMPLING ANALYSIS  (THEORETICAL -- not a measured capture rate)")
    print("=" * 78)
    intervals = {}
    for cid in DOORWAYS:
        ts = sorted(p["t"] for p in passes if p["camera_id"] == cid)
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        if gaps:
            intervals[f"camera {cid} measured"] = statistics.mean(gaps)
    intervals["earlier measurement"] = 5.34

    print(f"  {'transit lasts':>15}" + "".join(f"{k:>26}" for k in intervals))
    for d in (1.0, 2.0, 3.0, 5.0, 7.0, 10.0):
        row = f"  {d:>13.0f}s"
        for k, iv in intervals.items():
            row += f"{100 * _sampling_probability(d, iv):>25.0f}%"
        print(row)
    print()
    print("  P = min(1, duration / interval), for a periodic sampler with random")
    print("  phase. It is an optimistic CEILING: it assumes the detector fires on")
    print("  whichever pass lands inside the window. The measured per-pass")
    print("  detection rate above says that is not guaranteed, so real capture")
    print("  cannot exceed these numbers and will usually be below them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
