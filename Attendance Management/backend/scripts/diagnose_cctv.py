"""One-shot diagnostic for the CCTV person / face / Re-ID pipeline.

Tests every stage against a REAL camera frame and prints OK / FAIL per stage, so
you can see exactly where it breaks instead of guessing.

Run from the backend folder, with the repo venv:

    python scripts/diagnose_cctv.py                 # DVR channel 1 (dev room)
    python scripts/diagnose_cctv.py --channel 3     # other dev-room camera
    python scripts/diagnose_cctv.py --rtsp rtsp://user:pass@ip:554/...

It does NOT write anything to the database (Re-ID enrolment is stubbed out).
"""
import argparse
import os
import sys
from datetime import date
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

OK = "[ OK ]"
BAD = "[FAIL]"
WARN = "[WARN]"


def rtsp_from_env(channel: int) -> str:
    """DVR creds live in .env, which pydantic loads into Settings (not os.environ)."""
    from app.core.config import get_settings

    s = get_settings()
    ip = getattr(s, "dvr_ip", "") or os.getenv("DVR_IP", "")
    user = getattr(s, "dvr_username", "") or os.getenv("DVR_USERNAME", "")
    pwd = getattr(s, "dvr_password", "") or os.getenv("DVR_PASSWORD", "")
    if not (ip and user):
        return ""
    return f"rtsp://{quote(str(user))}:{quote(str(pwd))}@{ip}:554/Streaming/Channels/{channel}01"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", type=int, default=1)
    ap.add_argument("--rtsp", default="")
    args = ap.parse_args()

    # .env is loaded by app.core.config; import it first.
    from app.core.config import get_settings

    s = get_settings()

    print("=" * 62)
    print(" CCTV PIPELINE DIAGNOSTIC")
    print("=" * 62)

    # ---------- 1. Models on disk ----------
    print("\n1) MODELS")
    for label, path in (
        ("YOLO11 (person)", s.yolo_person_model_path),
        ("ByteTrack cfg  ", s.bytetrack_config_path),
        ("OSNet (Re-ID)  ", s.reid_model_path),
    ):
        p = path if os.path.exists(path) else os.path.join(os.path.dirname(__file__), "..", path)
        print(f"   {OK if os.path.exists(p) else BAD} {label}: {path}")

    # ---------- 2. Engines load ----------
    print("\n2) ENGINES")
    from app.services import bytetrack_engine, reid_service

    yolo_ok = bytetrack_engine.is_available()
    reid_ok = reid_service.is_available()
    print(f"   {OK if yolo_ok else BAD} YOLO11 + ByteTrack available")
    print(f"   {OK if reid_ok else BAD} OSNet Re-ID available")
    print(f"   ... imgsz={s.yolo_person_imgsz}  reid_threshold={s.reid_threshold}")

    # ---------- 3. Camera config ----------
    print("\n3) CAMERA CONFIG (database)")
    try:
        from app.db.session import SessionLocal
        from app.models import CameraConfig

        with SessionLocal() as db:
            for c in db.query(CameraConfig).all():
                purpose = getattr(c, "camera_purpose", None)
                tag = OK if purpose else WARN
                body = "body-tracking" if purpose == "MONITOR" else "face-only"
                print(f"   {tag} id={c.id:<3} {str(c.name):<12} purpose={purpose:<8} -> {body}")
    except Exception as exc:
        print(f"   {BAD} could not read cameras: {exc}")

    # ---------- 4. Grab a live frame ----------
    print(f"\n4) LIVE FRAME (channel {args.channel})")
    url = args.rtsp or rtsp_from_env(args.channel)
    if not url:
        print(f"   {BAD} no RTSP url (set DVR_IP/DVR_USERNAME/DVR_PASSWORD or pass --rtsp)")
        return
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    frame = None
    for _ in range(15):
        got, f = cap.read()
        if got and f is not None:
            frame = f
    cap.release()
    if frame is None:
        print(f"   {BAD} could not read a frame from {url}")
        return
    print(f"   {OK} frame {frame.shape[1]}x{frame.shape[0]}")
    cv2.imwrite("diagnose_frame.jpg", frame)
    print("   ... saved to diagnose_frame.jpg (open it to see what the AI sees)")

    # ---------- 5. YOLO + ByteTrack ----------
    print("\n5) YOLO11 -> ByteTrack")
    engine = bytetrack_engine.ByteTrackEngine(
        conf=float(os.getenv("CCTV_PERSON_CONF", "0.05")), camera_id=f"diag{args.channel}"
    )
    tracks = engine.update(frame)
    tracks = engine.update(frame)  # 2nd pass: tracker needs a step to confirm ids
    if tracks:
        print(f"   {OK} PEOPLE DETECTED: {len(tracks)}   track ids={[t.track_id for t in tracks]}")
    else:
        print(f"   {BAD} 0 people detected  <-- YOLO cannot see anyone in this frame")
        print("        try: lower CCTV_PERSON_CONF, raise yolo_person_imgsz,")
        print("             or use a bigger model (models/yolo11m.pt)")

    # ---------- 6. SCRFD faces ----------
    print("\n6) SCRFD (face detection)")
    from app.services.recognition import extract_faces_from_rgb

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = extract_faces_from_rgb(rgb)
    if faces:
        print(f"   {OK} faces found: {len(faces)}")
    else:
        print(f"   {WARN} 0 faces  (EXPECTED if everyone faces away — Re-ID is what saves you)")

    # ---------- 7. ArcFace identity ----------
    print("\n7) ArcFace (who is it?)")
    if not faces:
        print("   ...  skipped (no faces in frame)")
    else:
        from app.services.recognition import recognize_face

        for i, fc in enumerate(faces):
            r = recognize_face(fc, threshold=s.default_threshold, source="diag", mark_attendance=False)
            fd = (r.get("faces") or [{}])[0]
            if fd.get("matched"):
                print(f"   {OK} face {i}: {fd.get('employee_name')} (score={fd.get('score'):.3f})")
            else:
                print(f"   {WARN} face {i}: no match (best={fd.get('score', 0):.3f}) — is this person enrolled?")

    # ---------- 8. Re-ID ----------
    print("\n8) OSNet Re-ID (body embeddings)")
    if not tracks:
        print("   ...  skipped (no person tracks)")
    elif not reid_ok:
        print(f"   {BAD} Re-ID model unavailable")
    else:
        embs = reid_service.extract_body_embeddings(frame, [t.box for t in tracks])
        good = sum(1 for e in embs if e is not None)
        print(f"   {OK if good else BAD} body embeddings: {good}/{len(tracks)}")

        from app.services.identity_manager import identity_manager as IM

        IM._persist = lambda *a, **k: None  # never write during a diagnostic
        # Same-person sanity check: embed twice, must be ~1.0
        if good:
            e2 = reid_service.extract_body_embeddings(frame, [t.box for t in tracks])
            sim = float(np.dot(embs[0], e2[0]))
            print(f"   {OK if sim > 0.95 else BAD} self-similarity = {sim:.3f} (should be ~1.0)")

    # ---------- 9. Today's gallery ----------
    print("\n9) IDENTITY GALLERY (today)")
    try:
        from app.db.session import SessionLocal
        from app.models import BodyEmbedding, Employee

        with SessionLocal() as db:
            rows = (
                db.query(BodyEmbedding, Employee)
                .join(Employee, Employee.id == BodyEmbedding.employee_id)
                .filter(BodyEmbedding.day == date.today())
                .all()
            )
        if rows:
            seen: dict = {}
            for r, emp in rows:
                seen.setdefault((emp.full_name, r.camera_id), 0)
                seen[(emp.full_name, r.camera_id)] += 1
            print(f"   {OK} {len(rows)} embeddings learned today:")
            for (nm, cam), n in seen.items():
                print(f"        {nm:<22} camera {cam}: {n} embedding(s)")
        else:
            print(f"   {WARN} gallery EMPTY — nobody's face has been recognised on a monitor")
            print("        camera yet today, so Re-ID has nothing to match against.")
            print("        FIX: have the person LOOK AT the dev-room camera once.")
    except Exception as exc:
        print(f"   {BAD} gallery read failed: {exc}")

    print("\n" + "=" * 62)


if __name__ == "__main__":
    main()
