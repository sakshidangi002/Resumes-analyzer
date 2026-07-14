"""
camera_service.py
=================
Production-ready CCTV camera manager for Hikvision DVR.

Supports both RTSP streams (via OpenCV) and HCNetSDK (direct DVR connection).

Architecture
------------
  CameraManager            – singleton, manages N cameras
    └── CameraWorker       – one per camera; runs two daemon threads:
          ├── StreamThread – opens RTSP / USB, reads frames continuously
          └── RecognitionThread – picks latest frame, runs face recognition

    OR (for HCNetSDK):
    └── HCNetSDKCameraWorker – uses HCNetSDK for direct DVR connection
          ├── DVR Login
          ├── Live Preview
          ├── Stream Callback
          └── PlayCtrl Decoder

Design decisions
----------------
* Persistent VideoCapture – one capture per camera, NOT per request.
* Auto-reconnect with exponential back-off (2 s → 30 s max).
* Stale-frame watchdog – if no frame for 15 s, forces reconnect.
* Frame buffer protected by threading.Lock – recognition thread always
  gets the latest JPEG without blocking the stream reader.
* camera_purpose ("IN"/"OUT") is forwarded to the recognition service
  so the correct attendance event type is forced, regardless of any
  previous event for that employee.
* Environment variable CCTV_FFMPEG_OPTS allows passing extra FFmpeg
  options for network tuning without code changes.
* HCNetSDK support for direct DVR connection when source_type="hcnetsdk"
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.services.face_tracker import FaceTracker
from app.services.person_tracker import PersonTracker, check_line_crossing
from app.services import person_detector
from app.services import bytetrack_engine

logger = logging.getLogger(__name__)

# Import HCNetSDK components (will be used when source_type="hcnetsdk")
try:
    from app.services.hcnetsdk_camera import HCNetSDKCameraWorker
    HCNETSDK_AVAILABLE = True
except ImportError:
    HCNETSDK_AVAILABLE = False
    logger.warning("HCNetSDK camera module not available")

# ---------------------------------------------------------------------------
# Tuning constants (all overridable via env vars)
# ---------------------------------------------------------------------------
_RECONNECT_INIT_DELAY = float(os.getenv("CCTV_RECONNECT_INIT", "2.0"))   # seconds
_RECONNECT_MAX_DELAY  = float(os.getenv("CCTV_RECONNECT_MAX",  "30.0"))
_STALE_TIMEOUT        = float(os.getenv("CCTV_STALE_TIMEOUT",  "15.0"))  # force reconnect
_OPEN_TIMEOUT_MS      = int(os.getenv("CCTV_OPEN_TIMEOUT_MS",  "10000")) # 10 s
_READ_TIMEOUT_MS      = int(os.getenv("CCTV_READ_TIMEOUT_MS",  "5000"))  # 5 s
_JPEG_QUALITY         = int(os.getenv("CCTV_JPEG_QUALITY",     "80"))
_FPS_WINDOW           = 30  # frames used to compute rolling FPS
# Number of consecutive detected+matched frames before attendance is recorded.
# Default 1: exit cameras detect faces slowly and intermittently, so requiring 2
# consecutive frames caused recognitions to never confirm (name showed but no
# event). A single match is safe because the margin gate (min_match_margin)
# already rejects ambiguous/lookalike matches, and the per-camera cooldown
# blocks duplicates. Raise via CCTV_CONFIRM_FRAMES for stricter confirmation.
_CONFIRM_FRAMES       = int(os.getenv("CCTV_CONFIRM_FRAMES",   "1"))
# Target FPS for the display/encode thread. This is decoupled from recognition
# so the live feed stays smooth even while face analysis runs in the background.
_DISPLAY_FPS          = float(os.getenv("CCTV_DISPLAY_FPS",    "25"))
# Attendance cooldown PER CAMERA PER EMPLOYEE. Once an employee is marked on a
# camera, further marks are ignored until this many seconds pass — this covers
# the "employee lingers in view / leaves and comes back" case. IN and OUT are
# separate cameras (separate workers) so they never block each other.
_ATTENDANCE_COOLDOWN  = float(os.getenv("CCTV_ATTENDANCE_COOLDOWN", "20"))
# Optional: downscale the longest frame side to this many px BEFORE detection to
# speed up analysis on high-res streams (0 = disabled, detect at full res).
_DETECT_MAXSIDE       = int(os.getenv("CCTV_DETECT_MAXSIDE",   "0"))
# How often the analysis (detect+track+identify) loop runs. Kept small so face
# boxes follow people smoothly; the actual rate is bounded by detector speed.
_ANALYSIS_INTERVAL    = float(os.getenv("CCTV_ANALYSIS_INTERVAL", "0.12"))
# MONITOR cameras are display-only (never mark attendance) and often have people
# permanently in view (e.g. a seating area), so at the fast interval they run
# detection every cycle and MONOPOLISE the single global inference lock —
# starving the IN/OUT attendance cameras (a person at the entrance then waits
# 20-30s for a free inference slot). Monitor cameras therefore analyse at a much
# slower rate, reserving inference throughput for the attendance cameras. Their
# on-screen boxes/names simply refresh a little less often (no attendance impact).
_MONITOR_ANALYSIS_INTERVAL = float(os.getenv("CCTV_MONITOR_ANALYSIS_INTERVAL", "1.5"))
# How long (seconds) a RECOGNISED person keeps their name after their face is no
# longer visible — the box coasts along their motion until they leave the frame.
_IDENTITY_HOLD_SEC    = float(os.getenv("CCTV_IDENTITY_HOLD_SEC", "2.5"))
# Motion gate: skip the expensive face/person detection when the frame barely
# changed (empty doorway). Mean abs-diff below this = "no motion". This keeps
# CPU free with many cameras and makes detection instant when someone appears.
_MOTION_THRESHOLD     = float(os.getenv("CCTV_MOTION_THRESHOLD", "3.0"))
# If nobody has viewed a camera's stream for this long, stop encoding preview
# JPEGs (recognition/attendance keep running). Saves CPU for background work.
_DISPLAY_IDLE_SEC     = float(os.getenv("CCTV_DISPLAY_IDLE_SEC", "8.0"))
# Body/person tracking. When active, a recognised face binds to the person's body
# track so the name stays on them even when the face turns away, until they leave
# the frame. Engine: YOLO11+ByteTrack (models/yolo11n.pt) → MobileNet-SSD.
# Global switch: turns body tracking on for EVERY camera (incl. IN/OUT). Leave off
# — it adds body-detector inference to the attendance cameras and slows them.
_PERSON_TRACKING      = os.getenv("CCTV_PERSON_TRACKING", "").lower() in {"1", "true", "yes"}
# MONITOR-only switch (default ON). Working-area/room cameras see people from
# behind, from the side and seated — where the face-only pipeline detects nobody.
# Body tracking lets them box EVERY person and keep a name bound to that person
# once their face is seen even briefly. Scoped to MONITOR so the IN/OUT attendance
# cameras stay on the fast face-only path.
_MONITOR_PERSON_TRACKING = os.getenv("CCTV_MONITOR_PERSON_TRACKING", "true").lower() in {"1", "true", "yes"}
_PERSON_REVERIFY_SEC  = float(os.getenv("CCTV_PERSON_REVERIFY_SEC", "5.0"))
# Detection floor handed to YOLO. Deliberately LOW: ByteTrack's second-stage
# association needs low-score boxes to re-attach a seated/occluded person to an
# existing track. Track CREATION precision is guarded by `new_track_thresh` in
# models/bytetrack_person.yaml, not by this value. (PERSON_CONF kept for
# backwards compatibility with existing .env files.)
_PERSON_CONF          = float(os.getenv("CCTV_PERSON_CONF", os.getenv("PERSON_CONF", "0.10")))
# Person Re-ID: keeps an employee's name on their body track when their face is
# not visible. MONITOR cameras only — a body match must never mark attendance.
# Body Re-ID naming — DISABLED. It cannot tell these people apart.
#
# Measured on the live dev-room cameras (cosine, OSNet):
#     correct Adarsh match      0.71 - 0.82   (fluctuates frame to frame)
#     WRONG match (man->Saloni) 0.77
#     a different person        up to 0.75
# The true and false ranges OVERLAP COMPLETELY, so no threshold can separate them:
# set it low and people get the wrong name, set it high and nobody is named. From a
# tiny top-down seated crop OSNet mostly encodes clothing colour, which several
# employees share.
#
# A wrong name is worse than no name, so identity now comes ONLY from a real FACE
# match (which is correct when the face is visible: 73-79%). Once a face names a
# person, the name stays on their track; if the track is lost they revert to
# "Person #N" until their face is seen again.
_REID_ENABLED         = os.getenv("CCTV_REID_ENABLED", "false").lower() in {"1", "true", "yes"}
# Per-person face zoom. A seated person's face is ~20-30 px in the full frame —
# too small for SCRFD. Cropping their head region and upscaling it makes the face
# 100+ px, so it can be detected and recognised. Without this, a ceiling-mounted
# room camera can essentially never identify anybody.
# Per-person face ZOOM — REQUIRED on these cameras.
#
# It crops each person's head and upscales it 3x before running SCRFD. Measured on
# the live feed: the FULL-FRAME face of a seated person is only 18-20px and its
# ArcFace embedding scores 0.09 — i.e. recognition is IMPOSSIBLE without the zoom.
# With it, real people match at 73-79%.
#
# The catch: an upscaled face is interpolated, so its embedding is not fully
# trustworthy and can occasionally score high against the WRONG employee (a man was
# once labelled "Saloni Pathania" at 77%). Score and face size CANNOT separate the
# good from the bad — the correct matches sat in the same 73-79% range.
#
# The defence is therefore NOT to disable this, but _IDENT_CONFIRM below: a false
# match is random and won't repeat, a real one will.
_FACE_CROP_ENABLED    = os.getenv("CCTV_FACE_CROP", "true").lower() in {"1", "true", "yes"}
# Cameras mounted at a STEEP top-down angle. Their people score only ~0.11 (a
# well-aimed camera scores 0.36-0.67), so the normal 0.30 track threshold throws
# every real person away and the room shows "People: 0". These cameras get a
# permissive tracker + lower detection floor so people at least get DETECTED and
# boxed; well-aimed cameras keep the strict config that stops empty chairs being
# boxed and named. Comma-separated camera ids, e.g. CCTV_STEEP_CAMERAS=55
_STEEP_CAMERAS = {
    c.strip() for c in os.getenv("CCTV_STEEP_CAMERAS", "").split(",") if c.strip()
}
_STEEP_CONF           = float(os.getenv("CCTV_STEEP_PERSON_CONF", "0.08"))
_STEEP_TRACKER_CFG    = os.getenv("CCTV_STEEP_TRACKER", "models/bytetrack_person_lowconf.yaml")
_FACE_CROP_SCALE      = int(os.getenv("CCTV_FACE_CROP_SCALE", "3"))     # upscale factor
_FACE_CROP_HEAD_RATIO = float(os.getenv("CCTV_FACE_CROP_HEAD", "0.55"))  # top N of the body box
# Minimum REAL face width (original-frame pixels) before we even try to recognise.
# ArcFace needs genuine facial detail; upscaling a 25px face makes a BIGGER BLURRY
# face, not a more detailed one, and its embedding is meaningless — it can score
# high against the WRONG person (a man was labelled "Saloni Pathania"). Below this
# size we leave the person as "Person #N" instead of risking a wrong name.
#
# 45 was too strict: the faces these cameras genuinely resolve are ~30-40px, and
# they were producing CORRECT names (73-79%). Blocking them left everyone Unknown.
# 24 keeps those working while still rejecting the truly tiny faces whose
# embeddings are meaningless.
_MIN_FACE_PX          = int(os.getenv("CCTV_MIN_FACE_PX", "16"))
# How many times IN A ROW the same employee must be recognised on a track before
# their name is shown.
#
# The real defence against wrong names. Face size and score CANNOT separate good
# from bad here: the correct names scored 73-79% and the WRONG one ("Saloni
# Pathania" on a man) scored 77% — the same range, from faces of the same size.
# But a false match is RANDOM: it does not repeat. A genuine one does. Requiring
# two consecutive agreeing reads therefore filters the impostor while every real
# person still gets named (a few seconds later).
# How many times IN A ROW the same employee must be recognised before their name
# is shown. 1 = name them on the first good face match.
#
# It was briefly 2, to guard against the wrong name ("Saloni Pathania" on a man).
# But the evidence shows that wrong name came from BODY Re-ID (which scored 0.77
# for the wrong person and is now disabled), NOT from a face. Meanwhile requiring
# 2 consecutive face matches made naming almost impossible: a face here is only
# visible for a MOMENT — a person glancing at the camera is caught once, never
# twice — so nobody ever got named.
#
# Face matches on these cameras are correct when they happen (73-79%), so 1 is
# right. Raise this to 2 only if a wrong name ever appears from a FACE match.
_IDENT_CONFIRM        = int(os.getenv("CCTV_IDENT_CONFIRM", "1"))


def _face_px_width(face: dict, scale: int = 1) -> float:
    """Face width in ORIGINAL frame pixels (a zoomed crop is `scale`x enlarged)."""
    box = face.get("box") or []
    if len(box) < 4:
        return 0.0
    return abs(float(box[2]) - float(box[0])) / max(1, scale)
# Safety floor for a camera's recognition threshold. Older camera rows may still
# hold the legacy 0.05 value, which accepts near-random faces as a match. We
# never let a worker run below this floor regardless of the stored DB value.
_MIN_THRESHOLD        = float(os.getenv("CCTV_MIN_THRESHOLD", "0.35"))


# ---------------------------------------------------------------------------
# Background attendance writer
# ---------------------------------------------------------------------------
# The attendance DB write (insert event + recalc daily summary + commit) is a
# chain of Postgres round-trips that can take 50–200 ms. Running it inline on a
# recognition thread FREEZES recognition for every OTHER person in view during
# that write — the classic "second person lags" delay. We offload it to a small
# pool instead. Safety: the per-camera cooldown (`note_attendance_marked`) is
# already recorded synchronously BEFORE we submit, so a duplicate can never be
# queued even though the write itself happens off-thread.
_attendance_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("CCTV_ATTENDANCE_WRITERS", "2")),
    thread_name_prefix="attn-write",
)


def _submit_attendance(employee_id: int, camera_id: str, camera_purpose: str) -> None:
    """Queue an attendance write off the recognition thread and never raise."""
    from app.services.recognition import mark_cctv_attendance

    def _run() -> None:
        t0 = time.time()
        try:
            mark_cctv_attendance(
                employee_id, camera_id=camera_id, camera_purpose=camera_purpose,
            )
            logger.info(
                "ATTN-WRITE done emp=%s camera=%s purpose=%s took=%.0fms",
                employee_id, camera_id, camera_purpose, (time.time() - t0) * 1000,
            )
        except Exception:
            logger.exception(
                "ATTN-WRITE FAILED emp=%s camera=%s purpose=%s",
                employee_id, camera_id, camera_purpose,
            )

    _attendance_executor.submit(_run)


# ---------------------------------------------------------------------------
# HCNetSDK Configuration Parser
# ---------------------------------------------------------------------------
def parse_hcnetsdk_config(source_url: str) -> dict:
    """Parse HCNetSDK configuration from source_url.
    
    Expected format: hcnetsdk://ip:port@username:password?channel=X
    Example: hcnetsdk://192.168.1.100:8000@admin:password123?channel=1
    
    Returns dict with keys: dvr_ip, dvr_port, dvr_username, dvr_password, dvr_channel
    """
    try:
        # Pattern: hcnetsdk://ip:port@username:password?channel=X
        pattern = r'hcnetsdk://([^:]+):(\d+)@([^:]+):([^?]+)\?channel=(\d+)'
        match = re.match(pattern, source_url)
        
        if match:
            return {
                "dvr_ip": match.group(1),
                "dvr_port": int(match.group(2)),
                "dvr_username": match.group(3),
                "dvr_password": match.group(4),
                "dvr_channel": int(match.group(5)),
            }
        else:
            logger.error(f"Invalid HCNetSDK URL format: {source_url}")
            return None
    except Exception as e:
        logger.error(f"Error parsing HCNetSDK config: {e}")
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CameraRuntimeState:
    """Live runtime metrics for a single camera."""
    status: str = "stopped"             # stopped | connecting | running | reconnecting | error
    last_error: Optional[str] = None
    total_frames: int = 0
    reconnect_count: int = 0
    last_frame_time: float = 0.0        # time.time() of last successful frame
    updated_at: float = 0.0
    fps: float = 0.0
    latest_jpeg: Optional[bytes] = None
    latest_result: dict = field(default_factory=dict)
    _fps_ts: deque = field(default_factory=lambda: deque(maxlen=_FPS_WINDOW))
    active_tracks: int = 0  # Number of active face tracks
    display_fps: float = 0.0  # rendered (encoded) FPS shown to the viewer
    recognition_status: str = "idle"  # idle | analyzing | recognized
    crossing_count: int = 0  # people who crossed the doorway line (this camera)
    _disp_ts: deque = field(default_factory=lambda: deque(maxlen=_FPS_WINDOW))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _draw_enhanced_overlay(
    frame: np.ndarray,
    tracks: list,
    camera_name: str,
    fps: float,
    line: Optional[dict] = None,
    crossing_count: int = 0,
    track_label: str = "Faces",
) -> np.ndarray:
    """Enhanced overlay with green/red boxes, labels, confidence, and metadata."""
    annotated = frame.copy()

    # Doorway crossing line (cyan) if configured.
    if line:
        h, w = annotated.shape[:2]
        if line.get("orientation") == "vertical":
            x = int(line.get("position", 0.5) * w)
            cv2.line(annotated, (x, 0), (x, h), (255, 255, 0), 2)
        else:
            y = int(line.get("position", 0.5) * h)
            cv2.line(annotated, (0, y), (w, y), (255, 255, 0), 2)
    
    # Draw face overlays
    for track in tracks:
        display_info = track.get_display_info()
        box = display_info["box"]
        if len(box) < 4:
            continue
        
        x1, y1, x2, y2 = box
        matched = display_info["matched"]
        employee_name = display_info["employee_name"]
        confidence = display_info["confidence"]
        employee_id = display_info["employee_id"]
        employee_code = display_info["employee_code"]
        
        logger.debug(
            "STAGE-overlay track=%s matched=%s employee=%s confidence=%.4f",
            display_info.get("track_id", "unknown"),
            matched,
            employee_name,
            confidence,
        )

        # Color based on recognition status
        if matched:
            color = (34, 197, 94)  # Green for known employees
        else:
            color = (239, 68, 68)  # Red for unknown
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Build label text — always show the Track ID (office monitoring needs it)
        track_id = display_info.get("track_id", "?")
        label_lines = [f"{employee_name} #{track_id}"]

        if matched:
            # Add confidence percentage
            confidence_pct = int(confidence * 100)
            label_lines.append(f"Confidence: {confidence_pct}%")

            # Add employee ID if available
            if employee_id:
                id_display = employee_code or str(employee_id)
                label_lines.append(f"ID: {id_display}")
        
        # Draw label background and text
        label_height = 24 * len(label_lines)
        text_bg_x2 = x1 + max(180, max(len(line) for line in label_lines) * 9)
        
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - label_height - 4)),
            (text_bg_x2, y1 - 2),
            color,
            -1,
        )
        
        for i, line in enumerate(label_lines):
            y_pos = max(16, y1 - label_height + 4 + i * 24)
            cv2.putText(
                annotated,
                line,
                (x1 + 4, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    
    # Draw camera info overlay (top-left)
    overlay_lines = [
        f"Camera: {camera_name}",
        f"FPS: {fps:.1f}",
        # In body-tracking mode these are PERSON tracks, not faces — the caller
        # passes the correct label so the counter never misreports.
        f"{track_label}: {len(tracks)}",
    ]
    if line:
        overlay_lines.append(f"Crossings: {crossing_count}")
    
    # Add timestamp
    from datetime import datetime
    overlay_lines.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Draw overlay background
    overlay_height = 24 * len(overlay_lines) + 8
    overlay_width = 220
    cv2.rectangle(
        annotated,
        (10, 10),
        (10 + overlay_width, 10 + overlay_height),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(
        annotated,
        (10, 10),
        (10 + overlay_width, 10 + overlay_height),
        (255, 255, 255),
        1,
    )
    
    # Draw overlay text
    for i, line in enumerate(overlay_lines):
        y_pos = 30 + i * 24
        cv2.putText(
            annotated,
            line,
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    
    return annotated


def _open_capture(stream_url: str, source_type: str, camera_id: int) -> cv2.VideoCapture:
    """Open a VideoCapture with appropriate backend and timeouts."""
    source = stream_url.strip()
    logger.info("Camera %s: Opening stream: %s", camera_id, source)

    if source_type == "usb" or source.isdigit():
        logger.info("Camera %s: USB/webcam mode, index=%s", camera_id, source)
        return cv2.VideoCapture(int(source))

    # RTSP / HTTP – use FFmpeg backend with Hikvision-compatible options
    # These options help handle non-standard H.264 encoding from older DVRs
    # Append FFmpeg options to the URL for older OpenCV versions
    ffmpeg_options = {
        'rtsp_transport': 'tcp',  # Use TCP instead of UDP for reliability
        'fflags': 'nobuffer',     # Disable buffering
        'flags': 'low_delay',     # Low latency mode
        'rtsp_flags': 'prefer_tcp',  # Prefer TCP for RTSP
        'analyzeduration': '5000000',  # Analyze 5 seconds of stream for better SPS/PPS detection
        'probesize': '5000000',   # Probe 5 MB of stream
        'max_delay': '0',         # No delay
    }
    
    # Build FFmpeg options string and append to URL
    options_str = '&'.join([f'{k}={v}' for k, v in ffmpeg_options.items()])
    source_with_options = f"{source}?{options_str}"
    
    cap = cv2.VideoCapture(source_with_options, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, _OPEN_TIMEOUT_MS)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, _READ_TIMEOUT_MS)
    
    logger.info(f"Camera {camera_id}: Using FFmpeg options: {options_str}")
    return cap


def _check_ffmpeg() -> bool:
    """Return True if OpenCV was built with FFmpeg support.

    OpenCV's build info reports this as "FFMPEG: YES" (uppercase). The check
    must be case-insensitive — matching the exact string "FFmpeg" wrongly
    reported no FFmpeg on standard opencv-python wheels and skipped ALL camera
    startup (so no attendance cameras ran).
    """
    build = cv2.getBuildInformation()
    ok = "ffmpeg" in build.lower()
    if not ok:
        logger.error(
            "CRITICAL: OpenCV is built WITHOUT FFmpeg – RTSP streams will NOT work! "
            "Fix: run fix_opencv.bat to reinstall opencv-python (full build)."
        )
    return ok


def _is_blurry(gray: np.ndarray, threshold: float = 80.0) -> bool:
    """Return True when the frame is too blurry for reliable recognition.

    Takes a precomputed grayscale image so the recognition loop can reuse the
    same gray for the motion gate instead of converting to gray twice per tick.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold


def _face_in_person_crop(rgb: np.ndarray, box) -> Optional[dict]:
    """Find a face by zooming INTO one person, instead of scanning the whole frame.

    On a ceiling-mounted room camera a seated person's face is only ~20-30 px wide
    in the full frame — below what SCRFD can detect, so the face pass returns
    nothing and the person can never be identified. Cropping that person's head
    region and upscaling it makes the same face 100+ px, which SCRFD finds easily.
    (Measured on the live dev-room camera: full frame missed faces that the
    upscaled crop detected at 0.61 confidence.)

    The returned face dict carries its own ArcFace embedding, computed on the
    aligned crop, so it can be passed straight to recognize_face().
    """
    from app.services.recognition import extract_faces_from_rgb

    x1, y1, x2, y2 = (int(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    # Head region: the top slice of the body box, with a little padding.
    pad = int(bw * 0.15)
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2 = min(rgb.shape[1], x2 + pad)
    cy2 = min(rgb.shape[0], y1 + int(bh * _FACE_CROP_HEAD_RATIO))
    if cx2 - cx1 < 12 or cy2 - cy1 < 12:
        return None

    crop = rgb[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    up = cv2.resize(
        crop,
        (crop.shape[1] * _FACE_CROP_SCALE, crop.shape[0] * _FACE_CROP_SCALE),
        interpolation=cv2.INTER_CUBIC,
    )
    faces = extract_faces_from_rgb(up)
    if not faces:
        return None
    best = max(faces, key=lambda f: float(f.get("confidence", 0.0)))
    # The crop was enlarged _FACE_CROP_SCALE times, so divide back out to get the
    # REAL number of face pixels the camera actually captured. Too few = the
    # embedding is guesswork and could match the wrong employee.
    if _face_px_width(best, _FACE_CROP_SCALE) < _MIN_FACE_PX:
        return None
    return best


def _face_in_box(faces: list, box) -> Optional[dict]:
    """Return the highest-confidence detected face whose centre lies inside box."""
    x1, y1, x2, y2 = box
    best, best_score = None, -1.0
    for f in faces:
        fb = f.get("box") or []
        if len(fb) < 4:
            continue
        cx, cy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            score = float(f.get("confidence", 0.0))
            if score > best_score:
                best, best_score = f, score
    return best


# ---------------------------------------------------------------------------
# StreamThread – reads frames from RTSP continuously
# ---------------------------------------------------------------------------
class _StreamThread(threading.Thread):
    def __init__(self, worker: "CameraWorker") -> None:
        super().__init__(daemon=True, name=f"stream-{worker.camera_id}")
        self._w = worker
        self._stop_evt = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        w = self._w
        cap: Optional[cv2.VideoCapture] = None
        reconnect_delay = _RECONNECT_INIT_DELAY
        consecutive_failures = 0

        logger.info("Camera %s [%s]: Stream thread started", w.camera_id, w.name)

        while not self._stop_evt.is_set():
            # ── connect ────────────────────────────────────────────────────
            if cap is None or not cap.isOpened():
                w.state.status = "connecting"
                logger.warning(
                    "Camera %s [%s]: Connecting (attempt %d) ...",
                    w.camera_id, w.name, w.state.reconnect_count + 1,
                )
                try:
                    cap = _open_capture(w.stream_url, w.source_type, w.camera_id)
                except Exception as exc:
                    logger.error("Camera %s: Open exception: %s", w.camera_id, exc)
                    cap = None

                if cap is None or not cap.isOpened():
                    consecutive_failures += 1
                    w.state.reconnect_count += 1
                    w.state.status = "error"
                    w.state.last_error = (
                        f"Cannot open stream (attempt {w.state.reconnect_count}). "
                        "Check DVR IP, RTSP port, credentials, and H.264 codec."
                    )
                    logger.error("Camera %s: %s", w.camera_id, w.state.last_error)
                    reconnect_delay = min(reconnect_delay * 1.5, _RECONNECT_MAX_DELAY)
                    logger.info(
                        "Camera %s: Retrying in %.1fs", w.camera_id, reconnect_delay
                    )
                    self._stop_evt.wait(reconnect_delay)
                    continue

                consecutive_failures = 0
                reconnect_delay = _RECONNECT_INIT_DELAY
                w.state.status = "running"
                w.state.last_error = None
                logger.info("Camera %s [%s]: Connected successfully", w.camera_id, w.name)

            # ── stale watchdog ─────────────────────────────────────────────
            if (
                w.state.last_frame_time > 0
                and time.time() - w.state.last_frame_time > _STALE_TIMEOUT
            ):
                logger.warning(
                    "Camera %s: No frame for %.0fs – forcing reconnect",
                    w.camera_id, _STALE_TIMEOUT,
                )
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                cap = None
                w.state.status = "reconnecting"
                continue

            # ── read frame ─────────────────────────────────────────────────
            try:
                ok, frame = cap.read()
            except Exception as exc:
                logger.warning("Camera %s: Read exception: %s", w.camera_id, exc)
                ok, frame = False, None

            if not ok or frame is None:
                consecutive_failures += 1
                logger.warning(
                    "Camera %s: Frame read failed (%d consecutive)",
                    w.camera_id, consecutive_failures,
                )
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                cap = None
                w.state.status = "reconnecting"
                if consecutive_failures > 3:
                    reconnect_delay = min(reconnect_delay * 2, _RECONNECT_MAX_DELAY)
                self._stop_evt.wait(reconnect_delay)
                continue

            # ── frame received ─────────────────────────────────────────────
            consecutive_failures = 0
            now = time.time()
            w.state.total_frames += 1
            w.state.last_frame_time = now
            w.state.status = "running"

            # Rolling FPS calculation
            w.state._fps_ts.append(now)
            if len(w.state._fps_ts) >= 2:
                span = w.state._fps_ts[-1] - w.state._fps_ts[0]
                w.state.fps = round((len(w.state._fps_ts) - 1) / span, 1) if span > 0 else 0.0

            # Store latest frame for RecognitionThread
            with w._frame_lock:
                w._latest_frame = frame.copy()

            if w.state.total_frames % 200 == 0:
                logger.info(
                    "Camera %s: %d frames, %.1f FPS, reconnects=%d",
                    w.camera_id, w.state.total_frames, w.state.fps,
                    w.state.reconnect_count,
                )

        # cleanup
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        logger.info(
            "Camera %s: Stream thread stopped. frames=%d reconnects=%d",
            w.camera_id, w.state.total_frames, w.state.reconnect_count,
        )


# ---------------------------------------------------------------------------
# RecognitionThread – face detection on latest frame
# ---------------------------------------------------------------------------
class _RecognitionThread(threading.Thread):
    def __init__(self, worker: "CameraWorker") -> None:
        super().__init__(daemon=True, name=f"recog-{worker.camera_id}")
        self._w = worker
        self._stop_evt = threading.Event()
        self._prev_gray: Optional[np.ndarray] = None  # for motion gating
        # Edge-triggered log flags: log a skip ONCE when it starts, not every tick.
        self._blur_logged = False
        self._motion_logged = False
        self._stage_sig: Optional[tuple] = None  # last (persons, faces) logged
        # track_id -> [employee_id, consecutive_agreeing_reads]. A name is only
        # shown once the SAME employee is recognised _IDENT_CONFIRM times in a row,
        # which filters out random false matches (see _IDENT_CONFIRM).
        self._pending_ident: dict = {}

    def stop(self) -> None:
        self._stop_evt.set()

    def _apply_reid(self, w: "CameraWorker", frame: np.ndarray, ptracks: list, face_confirmed: list) -> None:
        """Keep a name on people whose face isn't visible (MONITOR cameras only).

        1. ENROL  — every track whose identity was just confirmed by ArcFace teaches
           this camera what that employee looks like *from this angle* (body
           embedding + seat position).
        2. MATCH  — every still-unknown track is matched against today's gallery for
           this camera (body Re-ID), then against their usual seat.

        This only ever LABELS a box. Attendance is untouched: monitor cameras are
        hard-blocked from marking, and this method never runs on IN/OUT cameras.
        """
        from app.services import reid_service
        from app.services.identity_manager import identity_manager

        if not reid_service.is_available():
            return

        unknown = [pt for pt in ptracks if pt.employee_id is None and pt.consecutive_misses == 0]
        confirmed_tracks = [pt for pt, _ in face_confirmed]
        need = confirmed_tracks + unknown
        if not need:
            return

        embeddings = reid_service.extract_body_embeddings(frame, [pt.box for pt in need])
        emb_of = {id(pt): e for pt, e in zip(need, embeddings)}

        # 1. Teach the gallery from the face-confirmed tracks.
        for pt, fd in face_confirmed:
            emb = emb_of.get(id(pt))
            if emb is None:
                continue
            identity_manager.enroll(
                employee_id=int(fd["employee_id"]),
                camera_id=str(w.camera_id),
                embedding=emb,
                centroid=pt.centroid(),   # PersonTrack.centroid is a method, not a property
                score=float(fd.get("score") or 0.0),
                name=fd.get("employee_name"),
                code=fd.get("employee_code"),
            )

        # 2. Put a name on the unknown tracks. An employee already bound to another
        #    live track on this camera is excluded — one person, one place.
        taken = {pt.employee_id for pt in ptracks if pt.employee_id is not None}
        for pt in unknown:
            emb = emb_of.get(id(pt))
            emp_id, score, source = identity_manager.match(str(w.camera_id), emb, taken)
            if emp_id is None:
                emp_id = identity_manager.seat_match(str(w.camera_id), pt.centroid(), taken)
                score, source = 0.0, "seat"
            if emp_id is None:
                self._pending_ident.pop(pt.track_id, None)
                continue

            # Same stable-confirmation rule as the face path: a body match must
            # agree with itself twice before it is allowed to name anyone. A false
            # Re-ID hit is random and won't repeat (that is how a man once got
            # labelled "Saloni Pathania" from a single 0.77 body match).
            _p = self._pending_ident.get(pt.track_id)
            if _p and _p[0] == int(emp_id):
                _p[1] += 1
            else:
                self._pending_ident[pt.track_id] = [int(emp_id), 1]

            name, code = identity_manager.label(emp_id)
            if self._pending_ident[pt.track_id][1] < _IDENT_CONFIRM:
                logger.info(
                    "IDENTITY camera=%s track=%d candidate=%s score=%.3f via=%s "
                    "(%d/%d confirmations — still Person #%d)",
                    w.camera_id, pt.track_id, name, float(score), source,
                    self._pending_ident[pt.track_id][1], _IDENT_CONFIRM, pt.track_id,
                )
                continue

            pt.bind_identity(int(emp_id), name, code, True, float(score))
            taken.add(int(emp_id))
            logger.info(
                "IDENTITY camera=%s track=%d employee=%s (id=%s) score=%.3f via=%s",
                w.camera_id, pt.track_id, name, emp_id, float(score), source,
            )

    def _analyze_person(
        self, w: "CameraWorker", frame: np.ndarray, rgb: np.ndarray, skip_faces: bool = False
    ) -> None:
        """Body-tracking pipeline: detect people, bind recognised faces to their
        body track, and keep the name on them until they leave the frame.

        `skip_faces` (set on a blurry frame) skips ONLY the face/recognition stage —
        body detection and tracking still run, so people stay boxed and already-bound
        identities keep riding their track.
        """
        from app.services.recognition import recognize_face
        from app.services.recognition import extract_faces_from_rgb

        # 1. Detect & track whole bodies. YOLO11+ByteTrack when available (best
        #    for crossing paths); else MobileNet-SSD detections + IoU tracker.
        #    (The engine logs detections / track ids itself, on change only.)
        if w.bytetrack_engine is not None:
            ptracks = w.bytetrack_engine.update(frame)
        else:
            persons = person_detector.detect_persons(frame)
            ptracks = w.person_tracker.update(persons)

        # 2. Detect faces (with embeddings) once on the full frame. Skipped when the
        #    frame is too blurry to recognise anyone reliably.
        faces = [] if skip_faces else extract_faces_from_rgb(rgb)

        # Log the stage counts only when they CHANGE (a static room would otherwise
        # log identical lines every analysis tick).
        sig = (len(ptracks), len(faces), skip_faces)
        if sig != self._stage_sig:
            self._stage_sig = sig
            logger.info(
                "PIPELINE camera=%s monitor=%s persons=%d faces=%d%s",
                w.camera_id, w.is_monitor, len(ptracks), len(faces),
                " (faces skipped: blurry frame)" if skip_faces else "",
            )

        # Precompute the doorway line position in pixels (if crossing enabled).
        line_px = None
        if w.crossing_enabled:
            h, wpx = frame.shape[:2]
            line_px = w.line_position * (wpx if w.line_orientation == "vertical" else h)

        def _mark(emp_id: int) -> None:
            if emp_id is not None and w.can_mark_attendance(int(emp_id)):
                w.note_attendance_marked(int(emp_id))
                w.state.recognition_status = "recognized"
                # Off-thread: never block body tracking on the DB round-trip.
                _submit_attendance(
                    int(emp_id), camera_id=str(w.camera_id), camera_purpose=w.camera_purpose,
                )

        any_match = False
        face_confirmed: list = []   # tracks whose identity came from a FACE this tick
        for pt in ptracks:
            fresh = pt.consecutive_misses == 0

            # (a) Bind identity: recognise a face inside this body when the track
            #     is still unknown or a periodic re-verify is due.
            if fresh and pt.needs_recognition(_PERSON_REVERIFY_SEC):
                face = _face_in_box(faces, pt.box)
                # Too few real face pixels -> the embedding is unreliable and can
                # match the WRONG employee. Better "Person #N" than a wrong name.
                if face is not None and _face_px_width(face) < _MIN_FACE_PX:
                    face = None
                # Zoom into this person when the full-frame pass found no face on
                # them. On a ceiling camera the face is far too small to detect at
                # frame scale — this is what makes identification possible at all.
                if face is None and _FACE_CROP_ENABLED and not skip_faces:
                    face = _face_in_person_crop(rgb, pt.box)
                if face is not None:
                    result = recognize_face(
                        face, threshold=w.threshold, source="cctv",
                        camera_id=str(w.camera_id), camera_purpose=w.camera_purpose,
                        mark_attendance=False,
                    )
                    fd = (result.get("faces") or [{}])[0]
                    _prev_emp = pt.employee_id

                    # ── Stable confirmation ─────────────────────────────────
                    # Never name someone on a single read. A false match is random
                    # and won't repeat; a real one will. Only bind after the SAME
                    # employee is recognised _IDENT_CONFIRM times consecutively.
                    _emp = fd.get("employee_id") if fd.get("matched") else None
                    if _emp is None:
                        self._pending_ident.pop(pt.track_id, None)
                    else:
                        _p = self._pending_ident.get(pt.track_id)
                        if _p and _p[0] == _emp:
                            _p[1] += 1
                        else:
                            self._pending_ident[pt.track_id] = [_emp, 1]

                        if self._pending_ident[pt.track_id][1] >= _IDENT_CONFIRM:
                            pt.bind_identity(
                                _emp, fd.get("employee_name") or "Person",
                                fd.get("employee_code"), True, fd.get("score", 0.0),
                            )
                        else:
                            logger.info(
                                "IDENTITY camera=%s track=%d candidate=%s score=%.3f "
                                "(%d/%d confirmations — still Person #%d)",
                                w.camera_id, pt.track_id, fd.get("employee_name"),
                                float(fd.get("score") or 0.0),
                                self._pending_ident[pt.track_id][1], _IDENT_CONFIRM,
                                pt.track_id,
                            )
                    # Log only when the identity on this track actually changes —
                    # a periodic re-verify of the same person stays silent.
                    if pt.employee_id != _prev_emp and pt.employee_id is not None:
                        logger.info(
                            "IDENTITY camera=%s track=%d employee=%s (id=%s) score=%.3f",
                            w.camera_id, pt.track_id, pt.employee_name,
                            pt.employee_id, float(fd.get("score") or 0.0),
                        )
                    if fd.get("matched") and fd.get("employee_id"):
                        face_confirmed.append((pt, fd))
            if pt.matched:
                any_match = True

            # A MONITOR camera NEVER records attendance — it only detects, tracks
            # and labels people. Skip all attendance logic for it.
            if w.is_monitor:
                continue

            # (b) Attendance. A recognised person is marked ONCE per track when
            #     EITHER they cross the doorway line OR their identity is stably
            #     confirmed — whichever happens first. This makes marking robust
            #     to a mis-set line and to recognition landing a frame after the
            #     crossing. Duplicates are still blocked by the per-camera
            #     cooldown and the IN/OUT presence state machine.
            if fresh:
                if w.crossing_enabled and check_line_crossing(
                    pt, w.line_orientation, line_px, w.entry_direction
                ):
                    pt.crossed = True
                    w.state.crossing_count += 1
                    logger.info(
                        "Camera %s [%s]: LINE-CROSS #%d track=%d person=%s",
                        w.camera_id, w.camera_purpose, w.state.crossing_count,
                        pt.track_id, pt.employee_name or "Person",
                    )

                if pt.matched and not pt.attendance_marked:
                    # Count consecutive recognised frames of the same employee.
                    if pt.pending_employee_id == pt.employee_id:
                        pt.confirm_count += 1
                    else:
                        pt.pending_employee_id = pt.employee_id
                        pt.confirm_count = 1

                    confirmed = pt.confirm_count >= _CONFIRM_FRAMES
                    # If a line is configured, a crossing is required OR a longer
                    # confirmation as fallback; without a line, confirmation alone.
                    trigger = pt.crossed or confirmed
                    if trigger:
                        pt.attendance_marked = True
                        logger.info(
                            "Camera %s [%s]: ATTENDANCE track=%d emp=%s via=%s",
                            w.camera_id, w.camera_purpose, pt.track_id,
                            pt.employee_name, "cross" if pt.crossed else "confirm",
                        )
                        _mark(pt.employee_id)

        # (a2) Re-ID / seat anchoring — MONITOR cameras only, and only AFTER the
        #      attendance loop above has run. Gating on `is_monitor` guarantees a
        #      BODY match can never reach attendance: only ArcFace on an IN/OUT
        #      camera may mark. This purely puts a name on a box.
        if w.is_monitor and _REID_ENABLED and ptracks:
            self._apply_reid(w, frame, ptracks, face_confirmed)
            any_match = any_match or any(pt.matched for pt in ptracks)

        # 3. Publish person tracks for the display thread.
        w.state.active_tracks = len(ptracks)
        with w._frame_lock:
            w._latest_tracks = list(ptracks)
            w.state.updated_at = time.time()
        if w.state.recognition_status == "analyzing":
            w.state.recognition_status = "recognized" if any_match else ("idle" if not ptracks else "analyzing")

    def run(self) -> None:
        w = self._w
        logger.info("Camera %s: Recognition thread started", w.camera_id)

        while not self._stop_evt.is_set():
            self._stop_evt.wait(max(0.02, w.analysis_interval))
            if self._stop_evt.is_set():
                break

            with w._frame_lock:
                frame = w._latest_frame

            if frame is None:
                continue

            # Grayscale is computed ONCE here and reused by both the blur check
            # and the motion gate below (previously two full-frame conversions).
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Blur gate. A blurry frame is useless for FACE recognition, but a
            # person's BODY is still perfectly detectable — so when body tracking
            # is on we keep the frame and only skip the face stage. Face-only
            # cameras behave exactly as before (frame dropped).
            blurry = _is_blurry(gray)
            if blurry and not w.use_person_tracking:
                if not self._blur_logged:
                    self._blur_logged = True
                    logger.info("Camera %s: skipping blurry frames (face-only mode)", w.camera_id)
                continue
            if not blurry:
                self._blur_logged = False

            try:
                w.state.recognition_status = "analyzing"

                # ── Motion gate ──────────────────────────────────────────────
                # Skip expensive detection ONLY on a truly empty, static scene
                # (an idle doorway with nobody tracked) to save CPU. It must NOT
                # skip when people are present: MONITOR cameras watch people who
                # sit still, and an entrance camera must keep tracking a person
                # who has stopped moving. So: never skip on a monitor camera, and
                # never skip while any track is active.
                static = False
                if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
                    motion = float(np.mean(cv2.absdiff(gray, self._prev_gray)))
                    static = motion < _MOTION_THRESHOLD
                self._prev_gray = gray

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if static and not w.is_monitor and w.state.active_tracks == 0:
                    # Empty, static scene → nothing to do; skip detection.
                    if not self._motion_logged:
                        self._motion_logged = True
                        logger.info("Camera %s: idle scene — detection paused (no motion)", w.camera_id)
                    with w._frame_lock:
                        w._latest_tracks = []
                    w.state.recognition_status = "idle"
                    continue
                if self._motion_logged:
                    self._motion_logged = False
                    logger.info("Camera %s: motion resumed — detection active", w.camera_id)

                # Body-tracking mode: track people, bind a recognised face to the
                # person so the name persists while they are in view.
                if w.use_person_tracking:
                    # `blurry` only disables the FACE stage — YOLO still runs.
                    self._analyze_person(w, frame, rgb, skip_faces=blurry)
                    continue

                # Import here to avoid circular imports at module load
                from app.services.recognition import recognize_face
                from app.services.recognition import extract_faces_from_rgb

                # Step 1: Detect all faces ONCE. Each face dict already carries its
                # embedding, so recognition never needs a second detection pass.
                # Optionally downscale first to speed up detection on hi-res feeds.
                det_scale = 1.0
                det_rgb = rgb
                if _DETECT_MAXSIDE > 0:
                    h, w_px = rgb.shape[:2]
                    longest = max(h, w_px)
                    if longest > _DETECT_MAXSIDE:
                        det_scale = _DETECT_MAXSIDE / float(longest)
                        det_rgb = cv2.resize(
                            rgb, (int(w_px * det_scale), int(h * det_scale))
                        )

                _t_detect0 = time.time()
                faces = extract_faces_from_rgb(det_rgb)
                _detect_ms = (time.time() - _t_detect0) * 1000
                if det_scale != 1.0:
                    inv = 1.0 / det_scale
                    for f in faces:
                        f["box"] = [c * inv for c in f["box"]]
                logger.debug(
                    "STAGE-detect camera=%s faces_detected=%d took=%.0fms",
                    w.camera_id, len(faces), _detect_ms,
                )

                # Carry the full face dict (with embedding) into the tracker so
                # each track can be recognised directly from its embedding.
                detections = [{"box": face.get("box"), "face": face} for face in faces]

                # Step 2: Update face tracker (always update for smooth tracking).
                # frame_shape lets coasted (recognised-but-face-hidden) tracks
                # expire once they drift out of the frame.
                tracks = w.face_tracker.update(detections, frame_shape=rgb.shape[:2])
                logger.debug(
                    "STAGE-track camera=%s active_tracks=%d", w.camera_id, len(tracks)
                )

                # Frame skipping for recognition only (not tracking)
                with w._frame_lock:
                    w._frame_counter += 1
                    skip_recognition = w.frame_skip > 0 and w._frame_counter % (w.frame_skip + 1) != 0

                # Step 3: Identify tracks seen in THIS frame (fresh embedding).
                # Identification is side-effect free — attendance is only marked
                # after the same employee is CONFIRMED across _CONFIRM_FRAMES.
                _match_ms = 0.0
                if not skip_recognition:
                    for track in tracks:
                        if track.consecutive_misses != 0 or track.face is None:
                            continue

                        # Reuse identity: once a track is confirmed & recorded,
                        # keep displaying it and stop re-matching (Issue 3). This
                        # both saves work and prevents identity flicker.
                        if track.attendance_marked and track.matched:
                            continue

                        logger.debug(
                            "STAGE-identify camera=%s track=%d box=%s",
                            w.camera_id, track.track_id, track.box,
                        )

                        # Match from the already-computed embedding (no re-detect,
                        # no attendance side effect).
                        _t_match0 = time.time()
                        result = recognize_face(
                            track.face,
                            threshold=w.threshold,
                            source="cctv",
                            camera_id=str(w.camera_id),
                            camera_purpose=w.camera_purpose,
                            mark_attendance=False,
                        )
                        _match_ms += (time.time() - _t_match0) * 1000
                        faces_data = result.get("faces", [])
                        face_data = faces_data[0] if faces_data else {}
                        emp_id = face_data.get("employee_id")
                        matched = face_data.get("matched", False)

                        track.update_recognition(
                            employee_id=emp_id,
                            employee_name=face_data.get("employee_name") or "Unknown Person",
                            employee_code=face_data.get("employee_code"),
                            matched=matched,
                            confidence=face_data.get("score", 0.0),
                        )

                        # Stable confirmation → mark attendance exactly once.
                        # (A MONITOR camera in face-mode fallback never marks.)
                        should_mark = track.register_identification(
                            emp_id, matched, _CONFIRM_FRAMES
                        )
                        if should_mark and not w.is_monitor:
                            # Per-camera cooldown: covers "lingering in view" and
                            # "left and came back quickly" (Issue 5). IN and OUT
                            # are separate workers so they never block each other.
                            if not w.can_mark_attendance(int(emp_id)):
                                logger.info(
                                    "Camera %s [%s]: track=%d %s within cooldown "
                                    "(%.0fs) -> attendance NOT re-marked",
                                    w.camera_id, w.camera_purpose, track.track_id,
                                    track.employee_name, _ATTENDANCE_COOLDOWN,
                                )
                            else:
                                logger.info(
                                    "Camera %s [%s]: track=%d CONFIRMED %s (id=%s, conf=%.1f%%) "
                                    "after %d frames -> marking attendance",
                                    w.camera_id, w.camera_purpose, track.track_id,
                                    track.employee_name, emp_id,
                                    track.confidence * 100, track.confirm_count,
                                )
                                w.note_attendance_marked(int(emp_id))
                                w.state.recognition_status = "recognized"
                                # Off-thread: never block recognition of the next
                                # person on this employee's DB write.
                                _submit_attendance(
                                    int(emp_id),
                                    camera_id=str(w.camera_id),
                                    camera_purpose=w.camera_purpose,
                                )
                        elif matched:
                            logger.debug(
                                "Camera %s: track=%d identified %s confirm=%d/%d",
                                w.camera_id, track.track_id, track.employee_name,
                                track.confirm_count, _CONFIRM_FRAMES,
                            )
                        else:
                            logger.debug(
                                "Camera %s: track=%d REJECTED score=%.4f reason=%s",
                                w.camera_id, track.track_id,
                                face_data.get("score", 0.0),
                                face_data.get("state", "no_match_or_below_threshold"),
                            )

                    # Multi-face summary — proves EVERY detected face was
                    # evaluated independently this frame (not just the first).
                    if tracks:
                        recognized = [
                            (t.employee_id, t.employee_name) for t in tracks if t.matched
                        ]
                        logger.info(
                            "MULTI-FACE camera=%s [%s] faces=%d tracks=%d recognized=%d ids=%s "
                            "detect=%.0fms match=%.0fms",
                            w.camera_id, w.camera_purpose, len(faces), len(tracks),
                            len(recognized), [r[0] for r in recognized],
                            _detect_ms, _match_ms,
                        )

                # Step 4: Publish tracks for the display thread to render.
                # NOTE: this thread NO LONGER encodes the display JPEG. The
                # dedicated _DisplayThread draws these persistent tracks onto the
                # latest raw frame at a high FPS, so the video stays smooth and
                # boxes never flicker even though analysis runs slower.
                w.state.active_tracks = len(tracks)
                with w._frame_lock:
                    w._latest_tracks = list(tracks)
                    w.state.updated_at = time.time()
                if w.state.recognition_status == "analyzing":
                    w.state.recognition_status = "idle" if not tracks else "recognized"

            except Exception as exc:
                logger.error(
                    "Camera %s: Recognition error: %s", w.camera_id, exc, exc_info=True
                )

        logger.info("Camera %s: Recognition thread stopped", w.camera_id)


# ---------------------------------------------------------------------------
# DisplayThread – renders the latest frame + latest tracks at a high FPS
# ---------------------------------------------------------------------------
class _DisplayThread(threading.Thread):
    """Encodes the preview JPEG independently of recognition.

    This is the key to a smooth feed: it reuses the most recent tracks (drawn as
    persistent overlays) and re-encodes the newest raw frame at ~_DISPLAY_FPS,
    so viewers see near-real-time video regardless of how long analysis takes.
    """

    def __init__(self, worker: "CameraWorker") -> None:
        super().__init__(daemon=True, name=f"display-{worker.camera_id}")
        self._w = worker
        self._stop_evt = threading.Event()
        self._drawn_ids: set = set()  # track ids whose box is already on screen

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        w = self._w
        period = 1.0 / _DISPLAY_FPS if _DISPLAY_FPS > 0 else 0.04
        logger.info("Camera %s: Display thread started (target %.0f FPS)", w.camera_id, _DISPLAY_FPS)

        while not self._stop_evt.is_set():
            self._stop_evt.wait(period)
            if self._stop_evt.is_set():
                break

            # Nobody watching → don't waste CPU encoding JPEGs. Recognition and
            # attendance keep running in the background regardless.
            if time.time() - w._last_view_ts > _DISPLAY_IDLE_SEC:
                w.state.display_fps = 0.0
                continue

            with w._frame_lock:
                frame = w._latest_frame
                tracks = list(w._latest_tracks)

            if frame is None:
                continue

            # STAGE: bounding box drawn — log the first frame each track's box is
            # actually rendered on screen (with timestamp, for latency measuring).
            cur_ids = {t.track_id for t in tracks if getattr(t, "track_id", None) is not None}
            for tid in (cur_ids - self._drawn_ids):
                logger.info("STAGE-box_drawn camera=%s track=%s", w.camera_id, tid)
            self._drawn_ids = cur_ids

            try:
                line_info = (
                    {"orientation": w.line_orientation, "position": w.line_position}
                    if (w.crossing_enabled and w.use_person_tracking) else None
                )
                annotated = _draw_enhanced_overlay(
                    frame, tracks, w.name, w.state.fps,
                    line=line_info, crossing_count=w.state.crossing_count,
                    track_label="People" if w.use_person_tracking else "Faces",
                )
                ok_enc, jpeg_buf = cv2.imencode(
                    ".jpg", annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY],
                )
                now = time.time()
                w.state._disp_ts.append(now)
                if len(w.state._disp_ts) >= 2:
                    span = w.state._disp_ts[-1] - w.state._disp_ts[0]
                    w.state.display_fps = round((len(w.state._disp_ts) - 1) / span, 1) if span > 0 else 0.0
                with w._frame_lock:
                    if ok_enc:
                        w.state.latest_jpeg = jpeg_buf.tobytes()
            except Exception as exc:
                logger.error("Camera %s: Display error: %s", w.camera_id, exc)

        logger.info("Camera %s: Display thread stopped", w.camera_id)


# ---------------------------------------------------------------------------
# CameraWorker – owns one StreamThread + one RecognitionThread
# ---------------------------------------------------------------------------
class CameraWorker:
    """Manages a single camera's stream + recognition lifecycle."""

    def __init__(
        self,
        *,
        camera_id: int,
        name: str,
        source_url: str,  # Database uses source_url
        source_type: str,
        camera_purpose: str,
        threshold: float,
        interval_sec: float,
        frame_skip: int = 0,  # Skip N frames between recognition (0 = no skip)
        crossing_enabled: bool = False,
        line_orientation: str = "horizontal",
        line_position: float = 0.5,
        entry_direction: str = "down",
    ) -> None:
        self.camera_id = camera_id
        self.name = name
        self.stream_url = source_url  # Keep as stream_url internally for consistency
        self.source_url = source_url  # Database field
        self.source_type = source_type
        self.camera_purpose = camera_purpose.upper()   # "IN" | "OUT"
        # Clamp to the safety floor so a stale/misconfigured DB row (e.g. the
        # legacy 0.05) can never make this camera accept near-random matches.
        self.threshold = max(float(threshold), _MIN_THRESHOLD)
        if float(threshold) < _MIN_THRESHOLD:
            logger.warning(
                "Camera %s: configured threshold %.3f below floor %.3f — using %.3f",
                camera_id, float(threshold), _MIN_THRESHOLD, self.threshold,
            )
        self.interval_sec = interval_sec
        self.frame_skip = frame_skip

        # Doorway line-crossing config
        self.crossing_enabled = bool(crossing_enabled)
        self.line_orientation = (line_orientation or "horizontal").lower()
        self.line_position = float(line_position)
        self.entry_direction = (entry_direction or "down").lower()

        self.state = CameraRuntimeState()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_tracks: list = []       # published by recog, drawn by display
        self._frame_counter = 0  # For frame skipping
        self._last_view_ts = 0.0  # last time the preview JPEG was requested

        # Per-employee last-marked timestamp for the camera-level cooldown.
        self._last_marked: dict[int, float] = {}

        # Attendance cameras (IN/OUT) analyse fast; display-only MONITOR cameras
        # analyse slowly so they don't starve the shared inference lock.
        self.is_monitor = self.camera_purpose == "MONITOR"
        self.analysis_interval = _MONITOR_ANALYSIS_INTERVAL if self.is_monitor else _ANALYSIS_INTERVAL

        # Track-expiry is counted in ANALYSIS FRAMES, so it MUST be derived from
        # THIS worker's analysis_interval. Otherwise a slow monitor camera (1.5s
        # per frame) keeps a recognised box on screen ~30s after the person has
        # left. Convert the desired hold TIME to frames for this interval.
        _hold_frames = max(2, round(_IDENTITY_HOLD_SEC / max(0.02, self.analysis_interval)))
        _unknown_frames = max(2, round(1.5 / max(0.02, self.analysis_interval)))

        # Face tracking for multi-face recognition. A recognised person keeps
        # their identity (name follows them) for ~_IDENTITY_HOLD_SEC after the
        # face turns away, then the track expires when they leave the frame.
        self.face_tracker = FaceTracker(
            max_distance=100.0,
            recognition_cooldown=3.0,
            max_misses=_unknown_frames,
            identity_max_misses=_hold_frames,
        )

        # Body/person tracking.
        #   * MONITOR (working-area) cameras use it by DEFAULT — a room camera sees
        #     people from behind/side and seated, where the face-only pipeline finds
        #     nobody. Body tracking boxes EVERY person and, once their face is seen
        #     even briefly, binds their name to that body track so the label sticks
        #     while they stay in frame. Disable with CCTV_MONITOR_PERSON_TRACKING=false.
        #   * IN/OUT attendance cameras stay on the pure face-detection pipeline
        #     (fast — no body-detector inference) unless CCTV_PERSON_TRACKING=true.
        # MONITOR cameras still NEVER mark attendance (enforced in _mark_attendance).
        # Engine preference: YOLO11+ByteTrack (models/yolo11n.pt) → MobileNet-SSD.
        _body_misses = max(3, _hold_frames)
        self.bytetrack_engine = None
        self.person_tracker: Optional[PersonTracker] = None
        self.use_person_tracking = False
        if _PERSON_TRACKING or (self.is_monitor and _MONITOR_PERSON_TRACKING):
            if bytetrack_engine.is_available():
                # A steep top-down camera needs a permissive tracker (its people
                # score ~0.11); a well-aimed one keeps the strict config so empty
                # chairs are never boxed.
                steep = str(camera_id) in _STEEP_CAMERAS
                # One engine (and therefore one YOLO model + one ByteTrack state)
                # PER CAMERA — tracker state must never be shared between feeds.
                self.bytetrack_engine = bytetrack_engine.ByteTrackEngine(
                    conf=(_STEEP_CONF if steep else _PERSON_CONF),
                    max_misses=_body_misses,
                    camera_id=str(camera_id),
                    tracker_cfg=(_STEEP_TRACKER_CFG if steep else None),
                )
                self.use_person_tracking = True
                logger.info(
                    "Camera %s: body tracking = YOLO11+ByteTrack (%s)",
                    camera_id, "STEEP/permissive" if steep else "standard",
                )
            elif person_detector.is_available():
                self.person_tracker = PersonTracker(max_misses=_body_misses)
                self.use_person_tracking = True
                logger.info("Camera %s: body tracking = MobileNet-SSD + IoU", camera_id)
            else:
                logger.warning(
                    "Camera %s: body tracking requested but NO model installed "
                    "(need ultralytics + models/yolo11n.pt, or models/mobilenet_ssd/). "
                    "%s", camera_id,
                    "MONITOR camera will only track visible faces."
                    if self.is_monitor else "Falling back to face tracking.",
                )

        self._stream_thread: Optional[_StreamThread] = None
        self._recog_thread: Optional[_RecognitionThread] = None
        self._display_thread: Optional[_DisplayThread] = None

    # ── attendance cooldown ─────────────────────────────────────────────────
    def can_mark_attendance(self, employee_id: int) -> bool:
        """True if this employee is outside the per-camera cooldown window."""
        last = self._last_marked.get(employee_id)
        return last is None or (time.time() - last) >= _ATTENDANCE_COOLDOWN

    def note_attendance_marked(self, employee_id: int) -> None:
        self._last_marked[employee_id] = time.time()

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._stream_thread and self._stream_thread.is_alive():
            return
        self._stream_thread = _StreamThread(self)
        self._recog_thread  = _RecognitionThread(self)
        self._display_thread = _DisplayThread(self)
        self._stream_thread.start()
        self._recog_thread.start()
        self._display_thread.start()
        logger.info(
            "Camera %s [%s]: Worker started (purpose=%s url=%s)",
            self.camera_id, self.name, self.camera_purpose, self.stream_url,
        )

    def stop(self) -> None:
        if self._stream_thread:
            self._stream_thread.stop()
        if self._recog_thread:
            self._recog_thread.stop()
        if self._display_thread:
            self._display_thread.stop()
        if self._stream_thread:
            self._stream_thread.join(timeout=6)
        if self._recog_thread:
            self._recog_thread.join(timeout=6)
        if self._display_thread:
            self._display_thread.join(timeout=6)
        self.state.status = "stopped"
        logger.info("Camera %s [%s]: Worker stopped", self.camera_id, self.name)

    def restart(self) -> None:
        logger.info("Camera %s: Restarting...", self.camera_id)
        self.stop()
        self.state = CameraRuntimeState()
        with self._frame_lock:
            self._latest_frame = None
            self._latest_tracks = []
        self.face_tracker.reset()
        if self.person_tracker is not None:
            self.person_tracker.reset()
        if self.bytetrack_engine is not None:
            self.bytetrack_engine.reset()
        self._last_marked.clear()
        self.start()
    
    def is_alive(self) -> bool:
        """Check if the worker is alive."""
        return self._stream_thread is not None and self._stream_thread.is_alive()

    # ── query helpers ───────────────────────────────────────────────────────
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame as numpy array."""
        with self._frame_lock:
            return self._latest_frame
    
    def get_latest_jpeg(self) -> Optional[bytes]:
        self._last_view_ts = time.time()  # someone is watching → keep encoding
        with self._frame_lock:
            return self.state.latest_jpeg

    def serialize_state(self) -> dict:
        s = self.state
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "location": getattr(self, "location", None),
            "stream_url": self.stream_url,
            "source_type": self.source_type,
            "camera_purpose": self.camera_purpose,
            "threshold": self.threshold,
            "interval_sec": self.interval_sec,
            "status": s.status,
            "last_error": s.last_error,
            "fps": s.fps,
            "capture_fps": s.fps,
            "display_fps": s.display_fps,
            "recognition_status": s.recognition_status,
            "active_tracks": s.active_tracks,
            "crossing_enabled": self.crossing_enabled,
            "crossing_count": s.crossing_count,
            "person_tracking": self.use_person_tracking,
            "total_frames": s.total_frames,
            "reconnect_count": s.reconnect_count,
            "last_frame_time": s.last_frame_time,
            "updated_at": s.updated_at,
            "last_result_faces": len(s.latest_result.get("faces", [])),
        }


# ---------------------------------------------------------------------------
# CameraManager – singleton
# ---------------------------------------------------------------------------
class CameraManager:
    """Thread-safe manager for all active camera workers."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._workers: dict[int, CameraWorker] = {}
        self._ffmpeg_ok: Optional[bool] = None

    # ── startup / shutdown ──────────────────────────────────────────────────
    def start_all_from_db(self) -> None:
        """Load enabled cameras from PostgreSQL and start workers.
        Called once from the FastAPI lifespan on application start.
        """
        self._ffmpeg_ok = _check_ffmpeg()
        if not self._ffmpeg_ok:
            logger.error(
                "Camera startup skipped: OpenCV lacks FFmpeg support. "
                "Run fix_opencv.bat then restart the server."
            )
            return

        try:
            from app.db.session import SessionLocal
            from app.models.camera import CameraConfig

            with SessionLocal() as db:
                rows = db.query(CameraConfig).filter(CameraConfig.enabled.is_(True)).all()

            logger.info("Camera startup: %d enabled camera(s) found", len(rows))
            for row in rows:
                self._start_worker_from_model(row)
        except Exception:
            logger.exception("Camera startup: failed to load cameras from DB")

    def stop_all(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for w in workers:
            w.stop()
        logger.info("CameraManager: all cameras stopped")

    def _start_worker_from_model(self, model) -> None:
        source_type = (model.source_type or "rtsp").lower()
        
        # Use HCNetSDK if source_type is "hcnetsdk"
        if source_type == "hcnetsdk":
            if not HCNETSDK_AVAILABLE:
                logger.error(
                    f"Camera {model.id}: HCNetSDK requested but not available, "
                    "falling back to RTSP"
                )
                source_type = "rtsp"
            else:
                config = parse_hcnetsdk_config(model.source_url)
                if not config:
                    logger.error(
                        f"Camera {model.id}: Failed to parse HCNetSDK config from {model.source_url}"
                    )
                    return
                
                worker = HCNetSDKCameraWorker(
                    camera_id=model.id,
                    name=model.name,
                    dvr_ip=config["dvr_ip"],
                    dvr_port=config["dvr_port"],
                    dvr_username=config["dvr_username"],
                    dvr_password=config["dvr_password"],
                    dvr_channel=config["dvr_channel"],
                    camera_purpose=model.camera_purpose or "IN",
                    threshold=float(model.threshold),
                    interval_sec=float(model.interval_sec),
                    frame_skip=getattr(model, 'frame_skip', 0),
                )
                worker.location = getattr(model, "location", None)
                
                # Update tracker config from model (if HCNetSDKCameraWorker has face_tracker)
                if hasattr(worker, 'face_tracker'):
                    tracking_max_distance = getattr(model, 'tracking_max_distance', 100.0)
                    tracking_cooldown = getattr(model, 'tracking_cooldown', 3.0)
                    worker.face_tracker.max_distance = tracking_max_distance
                    worker.face_tracker.recognition_cooldown = tracking_cooldown
                with self._lock:
                    old = self._workers.pop(model.id, None)
                    self._workers[model.id] = worker
                # Stop the previous worker OUTSIDE the lock: stop() joins its
                # threads (up to several seconds) and must not block every other
                # camera status/preview call that needs this lock.
                if old:
                    old.stop()
                worker.start()
                return
        
        # Use RTSP/USB for other source types
        worker = CameraWorker(
            camera_id=model.id,
            name=model.name,
            source_url=model.source_url,  # Database uses source_url
            source_type=source_type,
            camera_purpose=model.camera_purpose or "IN",
            threshold=float(model.threshold),
            interval_sec=float(model.interval_sec),
            frame_skip=getattr(model, 'frame_skip', 0),
            crossing_enabled=bool(getattr(model, 'crossing_enabled', False)),
            line_orientation=getattr(model, 'line_orientation', 'horizontal'),
            line_position=float(getattr(model, 'line_position', 0.5)),
            entry_direction=getattr(model, 'entry_direction', 'down'),
        )
        worker.location = getattr(model, "location", None)
        
        # Update tracker config from model
        tracking_max_distance = getattr(model, 'tracking_max_distance', 100.0)
        tracking_cooldown = getattr(model, 'tracking_cooldown', 3.0)
        worker.face_tracker.max_distance = tracking_max_distance
        worker.face_tracker.recognition_cooldown = tracking_cooldown
        with self._lock:
            old = self._workers.pop(model.id, None)
            self._workers[model.id] = worker
        # Stop the previous worker OUTSIDE the lock (see note above): joining its
        # threads must not block other camera status/preview calls.
        if old:
            old.stop()
        worker.start()

    # ── CRUD operations ─────────────────────────────────────────────────────
    def add_camera(self, camera_id: int) -> None:
        """Start a camera that was just added/enabled in the DB."""
        try:
            from app.db.session import SessionLocal
            from app.models.camera import CameraConfig

            with SessionLocal() as db:
                model = db.query(CameraConfig).filter(CameraConfig.id == camera_id).first()

            if not model:
                logger.warning("add_camera: camera_id=%d not found in DB", camera_id)
                return
            self._start_worker_from_model(model)
        except Exception:
            logger.exception("add_camera failed for camera_id=%d", camera_id)

    def remove_camera(self, camera_id: int) -> None:
        with self._lock:
            worker = self._workers.pop(camera_id, None)
        if worker:
            worker.stop()
            logger.info("CameraManager: camera %d removed", camera_id)

    def restart_camera(self, camera_id: int) -> bool:
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker:
            worker.restart()
            return True
        # Not running yet – try to start from DB
        self.add_camera(camera_id)
        return True

    # ── queries ─────────────────────────────────────────────────────────────
    def get_latest_jpeg(self, camera_id: int) -> Optional[bytes]:
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker:
            # Handle both CameraWorker and HCNetSDKCameraWorker
            if hasattr(worker, 'get_latest_jpeg'):
                return worker.get_latest_jpeg()
            elif hasattr(worker, 'state') and hasattr(worker.state, 'latest_jpeg'):
                return worker.state.latest_jpeg
        return None

    def get_status(self, camera_id: int) -> Optional[dict]:
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker:
            # Handle both CameraWorker and HCNetSDKCameraWorker
            if hasattr(worker, 'serialize_state'):
                return worker.serialize_state()
            elif hasattr(worker, 'state'):
                # Fallback for HCNetSDKCameraWorker
                return {
                    "camera_id": worker.camera_id,
                    "name": worker.name,
                    "status": worker.state.status,
                    "last_error": worker.state.last_error,
                    "fps": worker.state.fps,
                    "total_frames": worker.state.total_frames,
                    "reconnect_count": worker.state.reconnect_count,
                    "last_frame_time": worker.state.last_frame_time,
                }
        return None

    def list_statuses(self) -> list[dict]:
        with self._lock:
            workers = list(self._workers.values())
        statuses = []
        for w in workers:
            if hasattr(w, 'serialize_state'):
                statuses.append(w.serialize_state())
            elif hasattr(w, 'state'):
                # Fallback for HCNetSDKCameraWorker
                statuses.append({
                    "camera_id": w.camera_id,
                    "name": w.name,
                    "status": w.state.status,
                    "last_error": w.state.last_error,
                    "fps": w.state.fps,
                    "total_frames": w.state.total_frames,
                    "reconnect_count": w.state.reconnect_count,
                    "last_frame_time": w.state.last_frame_time,
                })
        return statuses

    def get_stats(self) -> dict:
        with self._lock:
            workers = list(self._workers.values())
        total    = len(workers)
        running  = sum(1 for w in workers if w.state.status == "running")
        error    = sum(1 for w in workers if w.state.status == "error")
        frames   = sum(w.state.total_frames for w in workers)
        reconnects = sum(w.state.reconnect_count for w in workers)
        return {
            "ffmpeg_ok": self._ffmpeg_ok,
            "total_cameras": total,
            "running_cameras": running,
            "error_cameras": error,
            "total_frames_processed": frames,
            "total_reconnects": reconnects,
        }

    def is_ffmpeg_ok(self) -> bool:
        if self._ffmpeg_ok is None:
            self._ffmpeg_ok = _check_ffmpeg()
        return bool(self._ffmpeg_ok)


# ---------------------------------------------------------------------------
# Module-level singleton used by all API routes and lifespan hooks
# ---------------------------------------------------------------------------
camera_manager = CameraManager()
