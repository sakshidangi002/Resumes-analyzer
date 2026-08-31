"""Find the chairs in a room camera by clustering COCO detections over time.

MANUAL / SETUP ONLY. Run once per camera; writes nothing unless asked.

WHY CLUSTERING RATHER THAN A SINGLE FRAME
-----------------------------------------
A single-frame measurement of camera 59 returned between 1 and 9 chairs
depending on input size and confidence, against roughly 10 visible. That looked
like "COCO cannot see these chairs", but it conflates two different things: a
detector that is WRONG, and a detector that is NOISY.

A chair does not move. So if the same physical seat is found in even a third of
frames, its location is knowable to far better precision than any one frame
suggests -- the noise averages out and the false positives, which land in
different places each time, do not survive.

This runs the detector over many frames spread across time (so people move and
occlude different seats), clusters detections that refer to the same physical
place, and reports how PERSISTENT each cluster is. Persistence is the signal
that separates a real chair from a one-frame hallucination.

WHAT IT DOES NOT DO
-------------------
It does not write the chair map on its own. It prints what it found and how
stable each seat was, and only emits config when asked with --emit. A chair map
generated from unreliable detections would look authoritative and be wrong, and
every occupancy number downstream would inherit that silently.
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

OUT = REPO / "data" / "cctv_v2_geometry"


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + bb - inter)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=59)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--spread", type=float, default=120.0,
                    help="seconds to spread the sample over")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min-persistence", type=float, default=0.30,
                    help="fraction of frames a cluster must appear in")
    ap.add_argument("--emit", action="store_true",
                    help="print a ROOM_GEOMETRY block for config/geometry.py")
    args = ap.parse_args()

    import logging
    logging.disable(logging.INFO)
    import cv2

    from app.cctv_v2.capture.grabber import CameraGrabber
    from app.cctv_v2.pipeline.detect import detect_chairs
    from app.db.session import SessionLocal
    from app.models.camera import CameraConfig
    from ultralytics import YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db:
        row = db.query(CameraConfig).filter(CameraConfig.id == args.camera).first()
    if row is None or not row.source_url:
        print(f"camera {args.camera}: no source_url")
        return 1

    # -- collect frames spread over time -------------------------------------
    g = CameraGrabber(args.camera, row.source_url)
    g.start()
    print(f"collecting {args.frames} frames over {args.spread:.0f}s "
          f"(spread so people occlude different seats)...")
    time.sleep(6.0)
    frames, last_seq = [], -1
    interval = args.spread / max(1, args.frames)
    deadline = time.time() + args.spread + 20
    while len(frames) < args.frames and time.time() < deadline:
        snap = g.get_latest()
        if snap is not None and snap.sequence != last_seq:
            last_seq = snap.sequence
            frames.append(snap.frame.copy())
        time.sleep(interval)
    g.stop()
    if not frames:
        print("no frames captured")
        return 1
    h, w = frames[0].shape[:2]
    print(f"  {len(frames)} frames at {w}x{h}")

    # -- detect chairs on each -----------------------------------------------
    models = [YOLO(str(BACKEND / "models" / "yolo11m.pt")) for _ in range(args.workers)]
    per_frame = [None] * len(frames)
    nxt, lock = [0], threading.Lock()

    def worker(m):
        while True:
            with lock:
                if nxt[0] >= len(frames):
                    return
                i = nxt[0]
                nxt[0] += 1
            per_frame[i] = detect_chairs(m, frames[i], args.imgsz, args.conf,
                                         args.camera)

    ts = [threading.Thread(target=worker, args=(m,)) for m in models]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    per_frame = [p for p in per_frame if p is not None]
    counts = [len(p) for p in per_frame]
    print(f"  chair detections per frame: min {min(counts)}  "
          f"median {statistics.median(counts):.0f}  max {max(counts)}")

    # -- cluster across frames -----------------------------------------------
    # A chair does not move, so detections of one seat overlap heavily across
    # frames. Clusters are grown greedily by IoU against the running mean box.
    clusters: list[dict] = []
    for fi, boxes in enumerate(per_frame):
        for b in boxes:
            best, best_iou = None, 0.0
            for c in clusters:
                o = _iou(b, c["box"])
                if o > best_iou:
                    best_iou, best = o, c
            if best is not None and best_iou >= 0.45:
                n = best["n"]
                best["box"] = tuple((best["box"][k] * n + b[k]) / (n + 1)
                                    for k in range(4))
                best["n"] += 1
                best["frames"].add(fi)
            else:
                clusters.append({"box": tuple(b), "n": 1, "frames": {fi}})

    nf = len(per_frame)
    for c in clusters:
        c["persistence"] = len(c["frames"]) / nf
    clusters.sort(key=lambda c: -c["persistence"])

    print()
    print("=" * 78)
    print(f"CHAIR CLUSTERS  camera {args.camera}  "
          f"({len(clusters)} candidates from {nf} frames)")
    print("=" * 78)
    print(f"  {'#':>3}{'persistence':>13}{'frames':>8}{'w x h px':>14}"
          f"{'area %':>9}   normalised box")
    stable = []
    for i, c in enumerate(clusters, 1):
        x1, y1, x2, y2 = c["box"]
        bw, bh = x2 - x1, y2 - y1
        area = 100.0 * bw * bh / (w * h)
        keep = c["persistence"] >= args.min_persistence
        if keep:
            stable.append(c)
        flag = "" if keep else "   (rejected: too intermittent)"
        print(f"  {i:>3}{100 * c['persistence']:>12.0f}%{len(c['frames']):>8}"
              f"{bw:>7.0f}x{bh:<6.0f}{area:>9.1f}   "
              f"({x1 / w:.3f},{y1 / h:.3f},{x2 / w:.3f},{y2 / h:.3f}){flag}")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  candidate clusters      {len(clusters)}")
    print(f"  stable (>={100 * args.min_persistence:.0f}% of frames)  {len(stable)}")
    print(f"  intermittent (rejected) {len(clusters) - len(stable)}")
    if stable:
        ps = [c["persistence"] for c in stable]
        print(f"  persistence of stable   min {100 * min(ps):.0f}%  "
              f"median {100 * statistics.median(ps):.0f}%  max {100 * max(ps):.0f}%")

    # -- overlay -------------------------------------------------------------
    canvas = frames[len(frames) // 2].copy()
    for i, c in enumerate(stable, 1):
        x1, y1, x2, y2 = (int(v) for v in c["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(canvas, (x1, y1 - 20), (x1 + 108, y1), (0, 0, 0), -1)
        cv2.putText(canvas, f"C{i} {100 * c['persistence']:.0f}%", (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    for c in clusters:
        if c in stable:
            continue
        x1, y1, x2, y2 = (int(v) for v in c["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 120, 255), 1)
    path = OUT / f"chairs{args.camera}_discovered.jpg"
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  overlay -> {path.name}  (GREEN = stable, ORANGE = rejected)")

    if args.emit and stable:
        print()
        print("  Paste into ROOM_GEOMETRY in config/geometry.py after checking")
        print("  the overlay against the real room:")
        print()
        print(f"    {args.camera}: RoomGeometry(chairs=(")
        for i, c in enumerate(stable, 1):
            x1, y1, x2, y2 = c["box"]
            print(f'        ChairZone("C{i}", ({x1 / w:.3f}, {y1 / h:.3f}, '
                  f"{x2 / w:.3f}, {y2 / h:.3f})),")
        print("    )),")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
