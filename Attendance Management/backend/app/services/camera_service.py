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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.services.face_tracker import FaceTracker
from app.services.person_tracker import PersonTracker, check_line_crossing
from app.services import attendance_gate
from app.services import bytetrack_engine
from app.services import camera_profile
from app.services import face_quality
from app.services import person_detector
from app.services import unknown_faces
from app.services import unknown_attendance

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
# Warn once when the displayed picture falls this far behind the live scene.
# 1.5s is well past normal (DVR encode + decode + encode is ~200-400ms) but
# below the point where an operator would call the feed broken.
_FRAME_AGE_WARN_MS    = float(os.getenv("CCTV_FRAME_AGE_WARN_MS", "1500"))
# After this many CONSECUTIVE failed opens, stop retrying every 30s and drop to
# a long interval.
#
# A camera that has failed 15 times in a row is misconfigured, unplugged, or the
# credentials are refused — none of which fixes itself in the next 30 seconds.
# Worse, retrying that fast is actively harmful: a Hikvision DVR locks out a
# source IP after ~5 failed logins, so a fast retry loop with rejected
# credentials REFRESHES that lockout forever and the camera can never recover
# on its own, even once the underlying problem is fixed.
_PERSISTENT_FAILURES  = int(os.getenv("CCTV_PERSISTENT_FAILURES", "15"))
_PERSISTENT_RETRY_SEC = float(os.getenv("CCTV_PERSISTENT_RETRY_SEC", "300"))  # 5 min
# Consecutive decode failures (grab() succeeded, retrieve() did not) before the
# connection is torn down. Guards the case where packets keep arriving but no
# picture can be decoded — an unsupported codec, typically H.265.
_MAX_RETRIEVE_FAILURES = int(os.getenv("CCTV_MAX_DECODE_FAILURES", "30"))
# Match on a quality-weighted average of a track's embeddings instead of on the
# single latest frame. See FaceTrack.add_observation for the reasoning. Set
# CCTV_EMBEDDING_FUSION=false to go back to single-frame matching.
_EMBEDDING_FUSION = os.getenv("CCTV_EMBEDDING_FUSION", "true").lower() in {"1", "true", "yes"}
# Seat anchoring: remember WHERE a person was when their face was confirmed, and
# reuse that to name them later when no face is visible. Independent of the
# body Re-ID flag above — appearance matching was unreliable here, position is
# not. MONITOR cameras only; can never mark attendance.
_SEAT_ANCHOR = os.getenv("CCTV_SEAT_ANCHOR", "false").lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# FFmpeg capture options  (C2)
# ---------------------------------------------------------------------------
def _ffmpeg_capture_options() -> str:
    """FFmpeg options for cv2.CAP_FFMPEG, in the format OpenCV actually reads.

    OpenCV takes these from the OPENCV_FFMPEG_CAPTURE_OPTIONS env var as
    `key;value` pairs joined by `|`. They CANNOT be passed as a URL query
    string — this module used to append `?rtsp_transport=tcp&...` to the RTSP
    URL, which merely sent that text to the DVR as part of the request URI. So
    TCP transport, `nobuffer` and `low_delay` were never actually enabled and
    every stream ran on the DVR's default (usually UDP), where packet loss
    produced the stalls that then wedged the reconnect loop (see C1).

    NOTE on the connect timeout key: FFmpeg renamed `stimeout` -> `timeout` for
    the RTSP demuxer in 5.0. Both are emitted; the demuxer ignores the one it
    does not recognise. Verify against your build with `ffmpeg -h demuxer=rtsp`.
    """
    micros = _OPEN_TIMEOUT_MS * 1000
    default = "|".join([
        "rtsp_transport;tcp",      # TCP — no packet loss. The whole point.
        "rtsp_flags;prefer_tcp",
        "fflags;nobuffer",         # do not accumulate a decode buffer
        "flags;low_delay",
        "reorder_queue_size;0",    # do not wait to reorder late RTP packets
        f"stimeout;{micros}",      # FFmpeg < 5 connect/read timeout (microseconds)
        f"timeout;{micros}",       # FFmpeg >= 5 equivalent
        "analyzeduration;2000000",
        "probesize;2000000",
    ])
    return os.getenv("CCTV_FFMPEG_OPTS", default)


# Must be set BEFORE the first VideoCapture is created — the FFmpeg backend
# reads it at capture-construction time. (CCTV_FFMPEG_OPTS was documented in
# this module's docstring but read nowhere; it now works as advertised.)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _ffmpeg_capture_options()
logger.info("FFmpeg capture options: %s", os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"])


# ---------------------------------------------------------------------------
# Credential redaction  (C4)
# ---------------------------------------------------------------------------
_URL_SCHEME_RE = re.compile(r"^(?P<scheme>[a-zA-Z][\w+.\-]*://)(?P<rest>.*)$", re.DOTALL)


def _redact_url(url: str) -> str:
    """Strip the password from a stream URL so it never reaches a log or an API
    response.

    Keeps the username — useful for diagnosis and not itself a secret. This
    module used to log the full source URL at INFO on every connect attempt,
    so a reconnect loop wrote the DVR password to disk thousands of times.

    Deliberately does NOT use urlparse for the split. Two of the three URL
    shapes here break it:

      rtsp://user:pass@host:554/path      standard  (creds LEFT of '@')
      hcnetsdk://ip:port@user:pass?ch=1   inverted  (creds RIGHT of '@')

    On the hcnetsdk form urlparse treats `user:pass` as host:port, so reading
    `.port` raises ValueError trying to int() the password. That would make
    this helper throw from inside the error handlers that call it, masking the
    original failure — so the parsing is done by hand and every path is
    total.
    """
    if not url:
        return ""

    match = _URL_SCHEME_RE.match(url)
    if not match:
        # No scheme (USB index, bare path). Nothing credential-shaped unless
        # there is an '@', in which case be conservative.
        return url if "@" not in url else "<redacted>"

    scheme, rest = match.group("scheme"), match.group("rest")
    if "@" not in rest:
        return url                       # no credentials present

    try:
        if scheme.lower().startswith("hcnetsdk"):
            # hcnetsdk://ip:port@username:password?channel=N
            location, _, credentials = rest.partition("@")
            user, sep, password = credentials.partition(":")
            if not sep:
                return url               # no password component
            # The `?channel=N` suffix sits on the END of the password, not the
            # username. Keep it — which channel failed is exactly what the
            # error logs calling this need to report.
            tail_at = password.find("?")
            query = password[tail_at:] if tail_at >= 0 else ""
            return f"{scheme}{location}@{user}:***{query}"

        # Standard: scheme://user:pass@host[:port]/path
        credentials, _, location = rest.partition("@")
        user, sep, _password = credentials.partition(":")
        if not sep:
            return url                   # userinfo with no password
        return f"{scheme}{user}:***@{location}"
    except Exception:
        # Never let redaction raise — a log call must not become an exception.
        return "<redacted>"
# NOTE ON REMOVED CONSTANTS
# --------------------------
# CCTV_CONFIRM_FRAMES, CCTV_IDENT_CONFIRM, CCTV_MIN_FACE_PX, CCTV_ATTENDANCE_
# COOLDOWN, CCTV_ANALYSIS_INTERVAL, CCTV_MIN_THRESHOLD and CCTV_FACE_CROP_SCALE
# used to live here as process-wide values. They are now per-camera settings in
# the database — see services/camera_profile.py and CameraWorker.profile().
#
# The reason is not tidiness. One global value had to serve both the check-in
# camera, whose matches become payroll rows, and the ceiling-mounted room
# cameras that can never mark attendance. Every time the room cameras stopped
# seeing anyone, the shared value was loosened — and that loosening silently
# applied to the payroll cameras too. CCTV_MIN_FACE_PX walking 45 -> 24 -> 16 is
# that process recorded in the git history.
#
# Existing env vars are no longer read. Set the equivalent per-camera columns, or
# leave them NULL to inherit the purpose defaults in camera_profile.
#
# Target FPS for the display/encode thread. This is decoupled from recognition
# so the live feed stays smooth even while face analysis runs in the background.
_DISPLAY_FPS          = float(os.getenv("CCTV_DISPLAY_FPS",    "25"))
# Fallback attendance cooldown, used only for sizing the bounded `_last_marked`
# dict and the warm-from-DB window at worker start. The value that actually
# gates a write is per-camera — CameraWorker.attendance_cooldown.
_COOLDOWN_HINT        = 20.0
# Optional: downscale the longest frame side to this many px BEFORE detection to
# speed up analysis on high-res streams (0 = disabled, detect at full res).
_DETECT_MAXSIDE       = int(os.getenv("CCTV_DETECT_MAXSIDE",   "0"))
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
# The defence is NOT to disable this. It is the evidence gate in
# services/attendance_gate: a false match is random and does not survive being
# combined with the rest of the track's observations, while a real one does. The
# zoom stays; what changed is that its output must now clear a quality gate
# (pose included) and contribute to a consistent fused template before it can
# name anybody, and it can never mark attendance on its own.
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
_FACE_CROP_HEAD_RATIO = float(os.getenv("CCTV_FACE_CROP_HEAD", "0.55"))  # top N of the body box
# How many times a SEAT/Re-ID guess must repeat before it may put a name on a
# box. MONITOR cameras only — this path can never mark attendance, so the cost
# of being wrong is a briefly-wrong overlay label. It is NOT the attendance
# confirmation, which lives in services/attendance_gate.
_ANCHOR_CONFIRM       = int(os.getenv("CCTV_ANCHOR_CONFIRM", "2"))


# Face size, threshold and margin floors are per-camera now — see
# services/camera_profile.py. `face_px_width` moved to services/face_quality so
# that the pixel measurement and the gate that uses it cannot drift apart.
_face_px_width = face_quality.face_px_width


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


# Outcomes meaning "no event was written, and retrying cannot help" — the state
# machine or business rules refused it. Anything else that failed is transient
# (a DB blip) and is worth retrying.
_RETRYABLE_ACTIONS = {"attendance_failed", "validation_failed"}

# Cap the backlog. The executor's queue is unbounded by default, so a database
# stall would grow it until the process ran out of memory.
_ATTENDANCE_QUEUE_MAX = int(os.getenv("CCTV_ATTENDANCE_QUEUE_MAX", "500"))


def _submit_attendance(
    employee_id: int,
    camera_id: str,
    camera_purpose: str,
    evidence: dict | None = None,
    event_time=None,
    attempt: int = 1,
    max_attempts: int = 3,
) -> None:
    """Queue an attendance write off the recognition thread and never raise.

    The result used to be DISCARDED. Because the caller has already set
    `attendance_marked` on the track and recorded the cooldown before getting
    here, a failed write meant the event was lost permanently — nothing retried
    it and nothing surfaced it. Now transient failures are retried with backoff
    and an exhausted one is logged at ERROR so it can be alerted on and keyed
    in by hand.
    """
    from app.services.recognition import mark_cctv_attendance

    queued = _attendance_executor._work_queue.qsize()
    if queued > _ATTENDANCE_QUEUE_MAX:
        logger.error(
            "ATTN-WRITE queue overflow (%d > %d) — dropping write for emp=%s "
            "camera=%s. The database is not keeping up.",
            queued, _ATTENDANCE_QUEUE_MAX, employee_id, camera_id,
        )
        return

    def _run() -> None:
        t0 = time.time()
        try:
            _payload, action = mark_cctv_attendance(
                employee_id,
                camera_id=camera_id,
                camera_purpose=camera_purpose,
                evidence=evidence,
                event_time=event_time,
            )
        except Exception:
            logger.exception(
                "ATTN-WRITE crashed emp=%s camera=%s purpose=%s",
                employee_id, camera_id, camera_purpose,
            )
            action = "attendance_failed"

        took_ms = (time.time() - t0) * 1000

        if action in _RETRYABLE_ACTIONS and attempt < max_attempts:
            delay = 2 ** attempt          # 2s, 4s
            logger.warning(
                "ATTN-WRITE retry %d/%d in %ds emp=%s camera=%s action=%s",
                attempt, max_attempts, delay, employee_id, camera_id, action,
            )
            timer = threading.Timer(
                delay,
                _submit_attendance,
                args=(employee_id, camera_id, camera_purpose, evidence, event_time,
                      attempt + 1, max_attempts),
            )
            timer.daemon = True
            timer.start()
            return

        if action in _RETRYABLE_ACTIONS:
            # Exhausted. Loud and countable: this is a lost attendance event and
            # somebody has to enter it by hand.
            logger.error(
                "ATTN-WRITE LOST emp=%s camera=%s purpose=%s action=%s after %d "
                "attempts — attendance NOT recorded, manual entry required",
                employee_id, camera_id, camera_purpose, action, max_attempts,
            )
            return

        logger.info(
            "ATTN-WRITE done emp=%s camera=%s purpose=%s action=%s took=%.0fms",
            employee_id, camera_id, camera_purpose, action, took_ms,
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
            # Redacted: an hcnetsdk:// URL carries the DVR password in-line.
            logger.error("Invalid HCNetSDK URL format: %s", _redact_url(source_url))
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
    # Age of the frame at the moment it was encoded for display, in ms — i.e.
    # how far behind real life the operator's picture is. THE number to watch:
    # a flat value means the pipeline keeps up, a steadily climbing one means
    # the reader is losing to the source and latency is accumulating without
    # bound (which ends in a stall). Cannot be inferred from FPS, which stays
    # healthy-looking while the backlog grows.
    frame_age_ms: float = 0.0
    retrieve_fps: float = 0.0  # frames actually DECODED per second (see _StreamThread)
    _disp_ts: deque = field(default_factory=lambda: deque(maxlen=_FPS_WINDOW))
    _retr_ts: deque = field(default_factory=lambda: deque(maxlen=_FPS_WINDOW))


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
    """Enhanced overlay with green/red boxes, labels, confidence, and metadata.

    Every dimension here is derived from the FRAME SIZE. Previously the box
    width (220 px), line height (24 px), font scale (0.5) and padding were
    hard-coded, having been tuned against a 1080p main stream. On a smaller
    feed — a D1/CIF sub-stream, say — those absolute sizes swallowed most of
    the picture: the info panel alone covered a third of the frame and the
    per-person labels ran off the right edge.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Scale factor against a 720p reference, clamped so a tiny feed stays
    # legible and a 4K one does not get a comically thin hairline overlay.
    scale = max(0.30, min(1.5, h / 720.0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55 * scale
    text_thick = max(1, int(round(1.2 * scale)))
    box_thick = max(1, int(round(2 * scale)))
    line_h = max(9, int(round(24 * scale)))
    pad = max(2, int(round(5 * scale)))

    def _text_w(text: str) -> int:
        return cv2.getTextSize(text, font, font_scale, text_thick)[0][0]

    # Doorway crossing line (cyan) if configured.
    if line:
        if line.get("orientation") == "vertical":
            x = int(line.get("position", 0.5) * w)
            cv2.line(annotated, (x, 0), (x, h), (255, 255, 0), box_thick)
        else:
            y = int(line.get("position", 0.5) * h)
            cv2.line(annotated, (0, y), (w, y), (255, 255, 0), box_thick)


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

        # Colour by HOW the identity was established. NOTE: OpenCV is BGR, not
        # RGB — the old "red" value (239, 68, 68) is B=239 and rendered BLUE, so
        # unknown people were boxed blue while every comment said red.
        #
        #   green  — a real face was matched
        #   amber  — nobody's face was visible; this is who normally sits here
        #   red    — unidentified
        #
        # The amber case must be visually distinct: presenting a positional
        # guess as a confident name is exactly how a man once ended up labelled
        # with a colleague's name.
        identity_source = display_info.get("identity_source")
        if matched and identity_source == "seat":
            color = (0, 170, 255)     # amber  (B, G, R)
        elif matched:
            color = (94, 197, 34)     # green  (B, G, R)
        else:
            color = (68, 68, 239)     # red    (B, G, R)
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thick)


        # Build label text — always show the Track ID (office monitoring needs it).
        # A body we cannot put a name to is labelled "Unknown", not "Person": the
        # box means "somebody is here but we do not know who", which is the honest
        # statement and what an operator wants to see. The track id still
        # distinguishes one unknown body from another.
        track_id = display_info.get("track_id", "?")
        _display_name = employee_name if matched else "Unknown"
        if not _display_name or _display_name in ("Person", "Unknown Person"):
            _display_name = "Unknown"
        label_lines = [f"{_display_name} - Track {track_id}"]

        if matched and identity_source == "seat":
            # Say plainly that this is a positional inference, not a face
            # identification. An operator must be able to tell the difference at
            # a glance — the name may be wrong if somebody swapped desks.
            label_lines = [f"{_display_name}? - Track {track_id}"]
            label_lines.append("by seat - face not seen")
        elif matched:
            # Add confidence percentage
            confidence_pct = int(confidence * 100)
            label_lines.append(f"Confidence: {confidence_pct}%")

            # Add employee ID if available
            if employee_id:
                id_display = employee_code or str(employee_id)
                label_lines.append(f"ID: {id_display}")
        
        # Label background sized to the text that is ACTUALLY rendered.
        # The old width was `max(180, longest_line * 9)` — a guess at 9 px per
        # character plus a 180 px floor, neither of which tracked the font
        # scale. It over-drew on short labels and ran off the right edge on
        # long ones, which is why "Unknown #4" was clipped mid-word.
        label_height = line_h * len(label_lines)
        label_width = max(_text_w(t) for t in label_lines) + pad * 2

        # Keep the label inside the frame: shift left if it would overflow the
        # right edge, and drop it BELOW the box if there is no room above.
        lx1 = max(0, min(x1, w - label_width))
        lx2 = min(w, lx1 + label_width)
        if y1 - label_height - pad >= 0:
            ly1 = y1 - label_height - pad
        else:
            ly1 = min(h - label_height - pad, y2)      # below the box instead
        ly1 = max(0, ly1)
        ly2 = min(h, ly1 + label_height + pad)

        cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), color, -1)

        for i, text in enumerate(label_lines):
            baseline_y = ly1 + line_h * (i + 1) - max(2, int(round(6 * scale)))
            cv2.putText(
                annotated,
                text,
                (lx1 + pad, baseline_y),
                font,
                font_scale,
                (255, 255, 255),
                text_thick,
                cv2.LINE_AA,
            )


    # Camera info panel (top-right).
    #
    # On a small feed the panel is compacted: labels are dropped to their
    # initials and the date is dropped from the timestamp. The date is the
    # longest string on the panel by some margin, and on a CIF sub-stream
    # spelling it out costs more of the picture than it is worth — an operator
    # watching a live feed already knows today's date.
    from datetime import datetime

    compact = scale < 0.6          # roughly: frame shorter than 432px
    now_text = datetime.now().strftime("%H:%M:%S" if compact else "%Y-%m-%d %H:%M:%S")

    if compact:
        overlay_lines = [
            str(camera_name),
            f"{fps:.0f}fps  {track_label[0]}:{len(tracks)}",
        ]
        if line:
            overlay_lines.append(f"X:{crossing_count}")
    else:
        overlay_lines = [
            f"Camera: {camera_name}",
            f"FPS: {fps:.1f}",
            # In body-tracking mode these are PERSON tracks, not faces — the
            # caller passes the correct label so the counter never misreports.
            f"{track_label}: {len(tracks)}",
        ]
        if line:
            overlay_lines.append(f"Crossings: {crossing_count}")
    overlay_lines.append(now_text)


    # Info panel, sized to its own content and capped at a third of the frame
    # width. The old version was a fixed 220x(24n+8) px block regardless of
    # resolution, which on a small sub-stream covered most of the scene.
    margin = max(4, int(round(10 * scale)))
    panel_h = line_h * len(overlay_lines) + pad * 2
    panel_w = min(
        max(_text_w(t) for t in overlay_lines) + pad * 2,
        max(80, int(w * 0.34)),
    )
    px1 = max(0, w - margin - panel_w)
    px2 = min(w - 1, px1 + panel_w)
    py2 = min(h - 1, margin + panel_h)

    # Translucent backing rather than solid black, so the panel obscures as
    # little of the scene as possible.
    roi = annotated[margin:py2, px1:px2]
    if roi.size:
        annotated[margin:py2, px1:px2] = cv2.addWeighted(
            roi, 0.35, np.zeros_like(roi), 0.65, 0,
        )
    cv2.rectangle(annotated, (px1, margin), (px2, py2), (255, 255, 255), 1)

    for i, text in enumerate(overlay_lines):
        baseline_y = margin + pad + line_h * (i + 1) - max(2, int(round(6 * scale)))
        if baseline_y >= py2:
            break                       # ran out of panel; do not spill outside
        cv2.putText(
            annotated,
            text,
            (px1 + pad, baseline_y),
            font,
            font_scale,
            (255, 255, 255),
            text_thick,
            cv2.LINE_AA,
        )

    return annotated


def _open_capture(stream_url: str, source_type: str, camera_id: int) -> cv2.VideoCapture:
    """Open a VideoCapture with the appropriate backend.

    Transport and timeout options come from OPENCV_FFMPEG_CAPTURE_OPTIONS, set
    once at module import (see _ffmpeg_capture_options). They must NOT be
    appended to the URL — OpenCV does not parse a query string as FFmpeg
    options, it just forwards the text to the DVR.
    """
    source = stream_url.strip()
    logger.info("Camera %s: Opening stream: %s", camera_id, _redact_url(source))

    if source_type == "usb" or source.isdigit():
        logger.info("Camera %s: USB/webcam mode, index=%s", camera_id, source)
        return cv2.VideoCapture(int(source))

    # RTSP / HTTP — FFmpeg backend.
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    # BUFFERSIZE is honoured by some backends and ignored by FFmpeg; harmless
    # to request. The real anti-buffering controls are `fflags;nobuffer` and
    # `reorder_queue_size;0` in the capture options above.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.info("Camera %s: capture opened=%s", camera_id, cap.isOpened())
    return cap


def open_capture_with_timeout(
    stream_url: str,
    source_type: str,
    camera_id: int,
    timeout_sec: float = 12.0,
) -> Optional[cv2.VideoCapture]:
    """Open a capture, giving up after `timeout_sec` whatever FFmpeg does.

    CAP_PROP_OPEN_TIMEOUT_MSEC cannot be used for this: it is set on the object
    AFTER the constructor has already blocked on the connect, so it arrives too
    late to bound the open that just happened. The FFmpeg-level `stimeout` is
    the primary bound; this is the backstop that keeps a wedged open off a
    FastAPI request thread.

    Returns an OPEN capture, or None. The caller owns release().
    """
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"open-{camera_id}")
    future = pool.submit(_open_capture, stream_url, source_type, camera_id)
    try:
        cap = future.result(timeout=timeout_sec)
    except FuturesTimeout:
        logger.error(
            "Camera %s: stream open timed out after %.0fs (%s)",
            camera_id, timeout_sec, _redact_url(stream_url),
        )
        # Do NOT wait for the worker — it is stuck inside FFmpeg. Release the
        # capture in the background if the open eventually succeeds, so a late
        # success cannot leak an RTSP session.
        def _release_late(fut) -> None:
            try:
                late = fut.result()
                if late is not None:
                    late.release()
            except Exception:
                logger.warning("future.result failed", exc_info=True)

        future.add_done_callback(_release_late)
        pool.shutdown(wait=False)
        return None
    except Exception:
        logger.exception("Camera %s: stream open failed", camera_id)
        pool.shutdown(wait=False)
        return None

    pool.shutdown(wait=False)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        return None
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


def _face_in_person_crop(rgb: np.ndarray, box, profile) -> tuple:
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
        return None, None

    # Head region: the top slice of the body box, with a little padding.
    pad = int(bw * 0.15)
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2 = min(rgb.shape[1], x2 + pad)
    cy2 = min(rgb.shape[0], y1 + int(bh * _FACE_CROP_HEAD_RATIO))
    if cx2 - cx1 < 12 or cy2 - cy1 < 12:
        return None, None

    crop = rgb[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None, None

    scale = int(profile.face_crop_scale)
    up = cv2.resize(
        crop,
        (crop.shape[1] * scale, crop.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    faces = extract_faces_from_rgb(up)
    if not faces:
        return None, None
    # Size/pose/blur are judged by the caller through face_quality, against this
    # camera's profile and with `scale` divided back out. Doing it here as well
    # would mean two thresholds that can disagree — which is what the old
    # `_MIN_FACE_PX` check here did against the one in _analyze_person.
    #
    # The UPSCALED image is returned alongside the face because the face box
    # coordinates are in that image, not in the frame. Without it the caller
    # cannot measure blur at all — and the zoom path is precisely where blur
    # matters most, since bicubic upsampling invents no detail and an
    # interpolated smear is what scores high against the wrong employee.
    return max(faces, key=lambda f: float(f.get("confidence", 0.0))), up


def _frame_capture_time(frame_ts: float):
    """Naive-IST datetime for a frame captured at `frame_ts` (time.time()).

    Attendance is written on a background executor, behind a pipeline that
    already lags by several hundred ms, so stamping the event at write time
    drifts it away from what actually happened — and away from the snapshot
    stored alongside it. Anchor the event to when the FRAME was captured.
    """
    from datetime import timedelta

    from app.core.datetime_utils import get_ist_now

    if not frame_ts:
        return None
    lag = max(0.0, time.time() - float(frame_ts))
    return get_ist_now() - timedelta(seconds=lag)


def _build_evidence(
    frame: np.ndarray,
    box,
    face_data: dict,
    track_id: Optional[int],
    employee_id: int,
    camera_id,
) -> dict:
    """Capture WHY this match was believed, at the moment it was made.

    Must be built here, not at attendance-write time: the write happens later on
    a background executor, by which point the frame is gone and the tracker has
    moved on. Never raises — evidence is an audit aid, and losing it must not
    cost an attendance record.
    """
    evidence = {
        "match_score": float(face_data.get("score") or 0.0),
        "match_margin": float(face_data.get("margin") or 0.0),
        "track_id": int(track_id) if track_id is not None else None,
        "snapshot_path": None,
    }
    try:
        from app.services.attendance_snapshot import save_face_snapshot

        evidence["snapshot_path"] = save_face_snapshot(
            frame, box, employee_id=int(employee_id), camera_id=str(camera_id),
        )
    except Exception:
        logger.exception("evidence snapshot failed camera=%s", camera_id)
    return evidence


# How long after an attendance write before a track is re-matched. Not zero —
# re-running recognition every tick on someone already recorded is wasted
# inference — and not infinite, which is what it effectively was before (a
# recorded track was skipped for the rest of its life, welding its identity on).
_POST_MARK_REVERIFY_SEC = float(os.getenv("CCTV_POST_MARK_REVERIFY_SEC", "4.0"))


def _log_decision(
    w: "CameraWorker",
    track_id: int,
    face_data: dict,
    quality,
    decision,
    employee_id,
    matched: bool,
) -> None:
    """One structured line per attendance decision, allowed or not.

    This is the debugging surface the system did not have. Previously a
    non-decision produced either nothing or a DEBUG line with a score, so
    "why was this person not marked?" was unanswerable in production — and
    equally, "why WAS this person marked?" had only a score to show for it.

    Every field the gate weighed appears here, so a day of logs can be grepped
    into a distribution: how many decisions died on low_quality, how many on
    inconsistent_track, what the score/margin spread looked like. That is also
    the input to tuning the camera profile.
    """
    fields = dict(quality.as_log_fields())
    fields.update(decision.as_log_fields())
    fields["reason"] = decision.reason or "allowed"
    level = logging.INFO if (decision.allowed or matched) else logging.DEBUG
    logger.log(
        level,
        "DECISION camera=%s [%s] track=%s employee=%s(%s) allowed=%s %s",
        w.camera_id, w.camera_purpose, track_id,
        face_data.get("employee_name") or "Unknown", employee_id,
        decision.allowed, fields,
    )


def _mark_from_track(
    w: "CameraWorker",
    track,
    employee_id: int,
    decision,
    event_time=None,
) -> bool:
    """Record attendance for a track the gate approved. Returns True if queued.

    The per-camera cooldown is checked and recorded BEFORE the write is queued,
    so a duplicate can never be enqueued even though the write itself happens on
    a background thread.

    ``attendance_marked`` is set here rather than inside the gate: the gate is
    pure so it can be tested and so a caller can evaluate a decision without
    causing one. Setting it here also means a track blocked by the cooldown is
    NOT flagged as marked, so it can try again once the window passes instead of
    being silently dropped for its whole life.
    """
    if not w.can_mark_attendance(employee_id):
        logger.info(
            "Camera %s [%s]: track=%s %s within cooldown (%.0fs) -> not re-marked",
            w.camera_id, w.camera_purpose, getattr(track, "track_id", "?"),
            getattr(track, "employee_name", None), w.attendance_cooldown,
        )
        return False

    track.attendance_marked = True
    w.note_attendance_marked(employee_id)
    w.state.recognition_status = "recognized"

    # Carry the gate's evidence onto the stored row alongside the match score,
    # so a disputed record shows not just "0.61" but how many observations,
    # how consistent they were, and which decision path allowed it.
    evidence = dict(getattr(track, "last_evidence", None) or {})
    evidence.setdefault("match_score", decision.score)
    evidence.setdefault("match_margin", decision.margin)
    evidence.setdefault("track_id", getattr(track, "track_id", None))

    logger.info(
        "ATTENDANCE camera=%s [%s] track=%s employee=%s(%s) via=%s %s",
        w.camera_id, w.camera_purpose, getattr(track, "track_id", "?"),
        getattr(track, "employee_name", None), employee_id,
        decision.path, decision.as_log_fields(),
    )
    _submit_attendance(
        employee_id,
        camera_id=str(w.camera_id),
        camera_purpose=w.camera_purpose,
        evidence=evidence,
        event_time=event_time,
    )
    return True


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


def _assign_faces_to_tracks(faces: list, ptracks: list) -> dict:
    """Map track_id -> face, giving each detected face to AT MOST ONE person.

    Person boxes overlap constantly on a crowded top-down view — someone
    standing behind a seated colleague produces two boxes covering the same
    pixels. Asking each track independently "is there a face inside me?" then
    hands the SAME face to both, so two different people are recognised as one
    employee and both get that name. Observed live: person 1 (conf 0.77) and
    person 3 (conf 0.18) were both handed the same 30px face and both scored
    identically against Rakhi Channa.

    Resolved greedily by containment: the track whose box holds the face most
    tightly wins it, and that face is then unavailable to anyone else. A tight
    box around a face is far more likely to be its actual owner than a large
    box that merely overlaps.
    """
    pairs = []
    for track in ptracks:
        tx1, ty1, tx2, ty2 = track.box
        t_area = max(1.0, float((tx2 - tx1) * (ty2 - ty1)))
        for idx, face in enumerate(faces):
            fb = face.get("box") or []
            if len(fb) < 4:
                continue
            cx, cy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
            if not (tx1 <= cx <= tx2 and ty1 <= cy <= ty2):
                continue
            # Smaller enclosing box = tighter fit = more likely the real owner.
            # Detector confidence breaks ties between equally tight boxes.
            pairs.append((1.0 / t_area, float(face.get("confidence", 0.0)), track.track_id, idx))

    pairs.sort(reverse=True)
    assigned: dict = {}
    used_faces: set = set()
    for _tightness, _conf, track_id, face_idx in pairs:
        if track_id in assigned or face_idx in used_faces:
            continue
        assigned[track_id] = faces[face_idx]
        used_faces.add(face_idx)
    return assigned


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
        last_retrieve = 0.0   # when a frame was last DECODED (see retrieve_period)
        consecutive_retrieve_failures = 0   # grab() ok but decode failed

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

                    # Persistent failure: back right off. Retrying a refused
                    # camera every 30s does not fix it and, against a Hikvision
                    # DVR, keeps re-triggering the illegal-login IP lockout so
                    # it can NEVER recover — even after the real problem is
                    # resolved. See _PERSISTENT_FAILURES.
                    if consecutive_failures >= _PERSISTENT_FAILURES:
                        reconnect_delay = _PERSISTENT_RETRY_SEC
                        w.state.last_error = (
                            f"Cannot open stream — {consecutive_failures} consecutive "
                            f"failures. Backing off to {_PERSISTENT_RETRY_SEC / 60:.0f} min "
                            "between attempts. Check credentials, DVR IP/port, and "
                            "whether the DVR has locked out this host after repeated "
                            "failed logins."
                        )
                        if consecutive_failures == _PERSISTENT_FAILURES:
                            logger.error(
                                "Camera %s: %d consecutive failures — backing off to "
                                "%.0fs. Fast retries against rejected credentials keep "
                                "a Hikvision lockout alive indefinitely.",
                                w.camera_id, consecutive_failures, _PERSISTENT_RETRY_SEC,
                            )
                    else:
                        reconnect_delay = min(reconnect_delay * 1.5, _RECONNECT_MAX_DELAY)
                        w.state.last_error = (
                            f"Cannot open stream (attempt {w.state.reconnect_count}). "
                            "Check DVR IP, RTSP port, credentials, and H.264 codec."
                        )
                        logger.error("Camera %s: %s", w.camera_id, w.state.last_error)

                    logger.info(
                        "Camera %s: Retrying in %.1fs", w.camera_id, reconnect_delay
                    )
                    self._stop_evt.wait(reconnect_delay)
                    continue

                consecutive_failures = 0
                reconnect_delay = _RECONNECT_INIT_DELAY
                w.state.status = "running"
                w.state.last_error = None
                # CRITICAL: restart the watchdog clock on every successful open.
                #
                # Without this the stale check below compares the brand-new
                # connection against the PREVIOUS session's timestamp, finds it
                # older than _STALE_TIMEOUT, and tears the connection down
                # before cap.read() is ever reached — a permanent
                # connect/drop loop in which the camera never streams again.
                #
                # It also arms the watchdog for the "connected but silent" case:
                # previously last_frame_time stayed 0 until the first frame
                # arrived, so a camera that opened but never delivered anything
                # was never caught at all.
                w.state.last_frame_time = time.time()
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
                        logger.debug("ignored, non-critical", exc_info=True)
                cap = None
                w.state.status = "reconnecting"
                w.state.reconnect_count += 1
                # Back off before retrying. This branch used to `continue`
                # immediately, so a camera that could not deliver frames opened
                # a fresh RTSP session as fast as the DVR would accept one —
                # Hikvision units cap concurrent sessions, so the loop locked
                # out other clients as well as burning CPU.
                reconnect_delay = min(reconnect_delay * 1.5, _RECONNECT_MAX_DELAY)
                self._stop_evt.wait(reconnect_delay)
                continue

            # ── grab a frame ───────────────────────────────────────────────
            # grab() advances the stream; retrieve() does the colour conversion
            # and hands back a usable array. Splitting them lets us drain the
            # stream at full rate — which is what stops the FFmpeg queue (and
            # therefore latency) from growing — while only paying the
            # conversion + allocation cost for frames a consumer will actually
            # look at. The pipeline consumes 1-8 fps; it was converting and
            # copying ~12 fps per camera across 12 cameras.
            try:
                ok = cap.grab()
            except Exception as exc:
                logger.warning("Camera %s: Grab exception: %s", w.camera_id, exc)
                ok = False

            if not ok:
                consecutive_failures += 1
                logger.warning(
                    "Camera %s: Frame read failed (%d consecutive)",
                    w.camera_id, consecutive_failures,
                )
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        logger.debug("ignored, non-critical", exc_info=True)
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
            # The watchdog cares that the STREAM is alive, so it is fed by
            # grab(), not by retrieve() — otherwise a camera nobody is watching
            # would look stale and be torn down.
            w.state.last_frame_time = now
            w.state.status = "running"

            # Rolling FPS calculation (stream rate — grabs per second)
            w.state._fps_ts.append(now)
            if len(w.state._fps_ts) >= 2:
                span = w.state._fps_ts[-1] - w.state._fps_ts[0]
                w.state.fps = round((len(w.state._fps_ts) - 1) / span, 1) if span > 0 else 0.0

            # ── decode only if somebody needs this frame ───────────────────
            if now - last_retrieve < w.retrieve_period():
                continue

            try:
                ok_dec, frame = cap.retrieve()
            except Exception as exc:
                logger.warning("Camera %s: Retrieve exception: %s", w.camera_id, exc)
                ok_dec, frame = False, None

            if not ok_dec or frame is None:
                # A failed retrieve after a good grab is usually a decode hiccup
                # (one corrupt packet), so skip the frame rather than tear down a
                # working connection.
                #
                # But it must be BOUNDED. grab() keeps succeeding on a stream
                # whose pictures cannot be decoded — wrong codec, or an H.265
                # feed the build cannot handle — so an unbounded `continue` here
                # spins forever with status "running", a fresh last_frame_time
                # (so the stale watchdog never fires) and not one usable frame.
                # The camera looks healthy and delivers nothing.
                consecutive_retrieve_failures += 1
                if consecutive_retrieve_failures >= _MAX_RETRIEVE_FAILURES:
                    logger.error(
                        "Camera %s: %d consecutive decode failures — forcing "
                        "reconnect. The stream is arriving but cannot be decoded "
                        "(check the codec: H.264 is supported, H.265 often is not).",
                        w.camera_id, consecutive_retrieve_failures,
                    )
                    w.state.last_error = (
                        f"Stream arrives but {consecutive_retrieve_failures} frames "
                        "in a row could not be decoded. Check the channel's codec "
                        "(H.264 vs H.265)."
                    )
                    consecutive_retrieve_failures = 0
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            logger.debug("ignored, non-critical", exc_info=True)
                    cap = None
                    w.state.status = "reconnecting"
                    self._stop_evt.wait(reconnect_delay)
                continue

            consecutive_retrieve_failures = 0
            last_retrieve = now
            w.state._retr_ts.append(now)
            if len(w.state._retr_ts) >= 2:
                span = w.state._retr_ts[-1] - w.state._retr_ts[0]
                w.state.retrieve_fps = (
                    round((len(w.state._retr_ts) - 1) / span, 1) if span > 0 else 0.0
                )

            # Store latest frame + its capture time for the recognition and
            # display threads. retrieve() allocates a fresh array per call, so
            # no defensive copy is needed.
            with w._frame_lock:
                w._latest_frame = frame
                w._latest_frame_ts = now

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
                logger.debug("ignored, non-critical", exc_info=True)
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
        # track_id -> [employee_id, consecutive_agreeing_reads], used ONLY by the
        # MONITOR seat/Re-ID anchoring path (_anchor_identities) to decide when a
        # positional guess may put a name on a box. Face identification and the
        # attendance decision no longer use it — see services/attendance_gate.
        self._pending_ident: dict = {}

    def stop(self) -> None:
        self._stop_evt.set()

    def _anchor_identities(
        self, w: "CameraWorker", frame: np.ndarray, ptracks: list, face_confirmed: list
    ) -> None:
        """Keep a name on people whose face is not currently visible.

        This is the answer to "only her hair is visible": the BODY is detected
        and tracked regardless, and identity is attached from whatever evidence
        exists — a face when one is visible, otherwise the seat.

        Two independent signals, deliberately separated:

          SEAT   — where this person was standing/sitting when their face WAS
                   confirmed. In a fixed-desk room this is strong and cheap, and
                   it survives losing the track entirely (which appearance-based
                   tracking does not).
          RE-ID  — OSNet body appearance. DISABLED by default and kept behind
                   its own flag: measured on these cameras, a true match scored
                   0.71-0.82 and a FALSE one 0.77, so no threshold separates
                   them. It used to be bundled with seat anchoring, which meant
                   turning off the broken signal also turned off the good one.

        Never marks attendance — callers gate on `is_monitor`, and
        `_mark_attendance` refuses MONITOR cameras independently.
        """
        from app.services.identity_manager import identity_manager

        # 1. LEARN. Every track whose identity was just confirmed by a real face
        #    teaches this camera where that person is.
        for pt, fd in face_confirmed:
            emb = None
            if _REID_ENABLED:
                emb = self._body_embedding(frame, pt)
            try:
                identity_manager.enroll(
                    employee_id=int(fd["employee_id"]),
                    camera_id=str(w.camera_id),
                    embedding=emb,
                    centroid=pt.centroid(),
                    score=float(fd.get("score") or 0.0),
                    name=fd.get("employee_name"),
                    code=fd.get("employee_code"),
                )
            except Exception:
                logger.exception("identity enrol failed camera=%s", w.camera_id)

        # 2. APPLY. Put a name on tracks with no face, from the seat they occupy.
        #    An employee already bound to another live track is excluded — one
        #    person cannot be in two places.
        taken = {pt.employee_id for pt in ptracks if pt.employee_id is not None}
        for pt in ptracks:
            if pt.employee_id is not None or pt.consecutive_misses != 0:
                continue
            emp_id = None
            if _SEAT_ANCHOR:
                try:
                    emp_id = identity_manager.seat_match(
                        str(w.camera_id), pt.centroid(), taken
                    )
                except Exception:
                    logger.exception("seat match failed camera=%s", w.camera_id)
            if emp_id is None:
                continue

            name, code = identity_manager.label(emp_id)
            # source="seat" so the overlay can show this as a positional
            # inference rather than a face identification, and so a later real
            # face match always overrides it.
            pt.bind_identity(int(emp_id), name, code, True, 0.0, source="seat")
            taken.add(int(emp_id))
            logger.info(
                "IDENTITY camera=%s track=%d employee=%s (id=%s) via=seat "
                "(no face visible — positional inference)",
                w.camera_id, pt.track_id, name, emp_id,
            )

    def _body_embedding(self, frame: np.ndarray, pt) -> Optional[np.ndarray]:
        """OSNet body embedding for one track, or None. Re-ID path only."""
        try:
            from app.services import reid_service

            if not reid_service.is_available():
                return None
            out = reid_service.extract_body_embeddings(frame, [pt.box])
            return out[0] if out else None
        except Exception:
            logger.exception("body embedding failed")
            return None

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
            if self._pending_ident[pt.track_id][1] < _ANCHOR_CONFIRM:
                logger.info(
                    "IDENTITY camera=%s track=%d candidate=%s score=%.3f via=%s "
                    "(%d/%d confirmations — still Person #%d)",
                    w.camera_id, pt.track_id, name, float(score), source,
                    self._pending_ident[pt.track_id][1], _ANCHOR_CONFIRM, pt.track_id,
                )
                continue

            pt.bind_identity(int(emp_id), name, code, True, float(score))
            taken.add(int(emp_id))
            logger.info(
                "IDENTITY camera=%s track=%d employee=%s (id=%s) score=%.3f via=%s",
                w.camera_id, pt.track_id, name, emp_id, float(score), source,
            )

    def _analyze_person(
        self, w: "CameraWorker", frame: np.ndarray, rgb: np.ndarray,
        skip_faces: bool = False, frame_ts: float = 0.0,
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
            # OpenCV can execute the configured YOLO ONNX model even when the
            # optional Ultralytics package is unavailable. Monitor cameras use
            # their permissive model/confidence so seated and partly occluded
            # employees are boxed instead of falling back to face-only tracks.
            # NOTE: the ONNX fallback runs at the model's fixed 960x960 export
            # size, so there is no size to pass -- yolo_monitor_imgsz applies to
            # the Ultralytics/ByteTrack path above, not here.
            from app.core.config import get_settings as _get_settings
            _s = _get_settings()
            _monitor_model = (
                getattr(_s, "yolo_monitor_model_path", "") or ""
            ) if w.is_monitor else ""
            persons = person_detector.detect_persons(
                frame,
                conf_threshold=(_STEEP_CONF if w.is_monitor else _PERSON_CONF),
                model_path=(_monitor_model or None),
            )
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

        # Anchor any attendance event to the frame's capture time, not to
        # whenever the background writer gets to it.
        frame_ist_time = _frame_capture_time(frame_ts)
        profile = w.profile()

        # Precompute the doorway line position in pixels (if crossing enabled).
        line_px = None
        if w.crossing_enabled:
            h, wpx = frame.shape[:2]
            line_px = w.line_position * (wpx if w.line_orientation == "vertical" else h)

        # Each detected face belongs to exactly one person. Overlapping body
        # boxes would otherwise both claim it and be recognised as the same
        # employee — see _assign_faces_to_tracks.
        face_by_track = _assign_faces_to_tracks(faces, ptracks)

        any_match = False
        face_confirmed: list = []   # tracks whose identity came from a FACE this tick
        for pt in ptracks:
            fresh = pt.consecutive_misses == 0

            # (a) Bind identity: recognise a face inside this body when the track
            #     is still unknown or a periodic re-verify is due.
            if fresh and pt.needs_recognition(_PERSON_REVERIFY_SEC):
                face = face_by_track.get(pt.track_id)
                face_scale = 1.0
                face_image = rgb          # the image `face`'s box refers to
                # Zoom into this person when the full-frame pass found no face on
                # them. On a ceiling camera the face is far too small to detect at
                # frame scale — this is what makes identification possible at all.
                if face is None and _FACE_CROP_ENABLED and not skip_faces:
                    face, face_image = _face_in_person_crop(rgb, pt.box, profile)
                    if face is not None:
                        # Coordinates (and therefore the measured width) are in
                        # the upscaled crop, not the frame.
                        face_scale = float(profile.face_crop_scale)

                # ── Quality gate ────────────────────────────────────────────
                # Replaces a bare `_face_px_width(face) < _MIN_FACE_PX` check.
                # Size alone cannot tell a usable face from a 30px full profile,
                # and a profile embedding is what lands on an arbitrary employee.
                quality = None
                if face is not None:
                    quality = face_quality.assess(
                        face, face_image, limits=profile.limits, scale=face_scale,
                    )
                    if not quality.ok:
                        # The full-frame detector can find a 16px face but the
                        # quality gate quite correctly rejects it. Retry only
                        # the size failure on an upscaled head crop; this adds
                        # pixels for SCRFD without weakening the payroll
                        # threshold, margin, pose, or blur rules.
                        if (
                            quality.reason == "face_too_small"
                            and _FACE_CROP_ENABLED
                            and not skip_faces
                            and face_scale == 1.0
                        ):
                            cropped_face, cropped_image = _face_in_person_crop(
                                rgb, pt.box, profile,
                            )
                            if cropped_face is not None:
                                cropped_quality = face_quality.assess(
                                    cropped_face, cropped_image,
                                    limits=profile.limits,
                                    scale=float(profile.face_crop_scale),
                                )
                                if cropped_quality.ok:
                                    face = cropped_face
                                    face_image = cropped_image
                                    face_scale = float(profile.face_crop_scale)
                                    quality = cropped_quality
                        if not quality.ok:
                            logger.debug(
                                "QUALITY camera=%s track=%d rejected=%s %s",
                                w.camera_id, pt.track_id, quality.reason,
                                quality.as_log_fields(),
                            )
                            face = None

                if face is not None:
                    # Fuse across every face seen on this body track before
                    # matching. This matters more here than at the entrance: a
                    # seated person is in view for minutes, so there are many
                    # observations to combine, and a room camera's faces are the
                    # smallest and noisiest in the system. Matching each glance
                    # in isolation throws all of that evidence away.
                    match_face = face
                    accepted = True
                    if _EMBEDDING_FUSION:
                        accepted = pt.add_observation(
                            face.get("embedding"), quality=quality.score,
                        )
                        fused = pt.fused_embedding()
                        if fused is not None:
                            match_face = dict(face)
                            match_face["embedding"] = fused

                    if not accepted:
                        # Inconsistent with the rest of this track — on a body
                        # track that usually means the tracker handed this box to
                        # a different person, which is routine when people pass
                        # each other. Matching on it would name the wrong person.
                        logger.info(
                            "FUSION camera=%s track=%d observation rejected as "
                            "outlier (accepted=%d rejected=%d) — likely a track switch",
                            w.camera_id, pt.track_id,
                            pt.fuser.accepted, pt.fuser.rejected,
                        )
                    else:
                        result = recognize_face(
                            match_face, threshold=profile.threshold, source="cctv",
                            camera_id=str(w.camera_id), camera_purpose=w.camera_purpose,
                            mark_attendance=False, min_margin=profile.margin,
                        )
                        fd = (result.get("faces") or [{}])[0]
                        _prev_emp = pt.employee_id
                        _emp = fd.get("employee_id") if fd.get("matched") else None

                        if _emp is not None:
                            pt.bind_identity(
                                _emp, fd.get("employee_name") or "Person",
                                fd.get("employee_code"), True, fd.get("score", 0.0),
                            )
                            # Capture provenance NOW, while the frame still
                            # exists. Snapshot the PERSON box, not the face box:
                            # a face found via _face_in_person_crop has
                            # coordinates in the upscaled crop, not the frame,
                            # so it cannot be cropped out of `frame`.
                            if not w.is_monitor:
                                pt.last_evidence = _build_evidence(
                                    frame, pt.box, fd, pt.track_id, _emp, w.camera_id,
                                )
                            face_confirmed.append((pt, fd))

                        # Log only when the identity on this track actually
                        # changes — a periodic re-verify stays silent.
                        if pt.employee_id != _prev_emp and pt.employee_id is not None:
                            logger.info(
                                "IDENTITY camera=%s track=%d employee=%s (id=%s) score=%.3f",
                                w.camera_id, pt.track_id, pt.employee_name,
                                pt.employee_id, float(fd.get("score") or 0.0),
                            )

                        # ── Attendance decision ─────────────────────────────
                        if profile.marks_attendance:
                            decision = attendance_gate.evaluate(
                                pt,
                                employee_id=_emp,
                                matched=bool(fd.get("matched")),
                                score=float(fd.get("score") or 0.0),
                                margin=float(fd.get("margin") or 0.0),
                                profile=profile,
                            )
                            _log_decision(
                                w, pt.track_id, fd, quality, decision,
                                _emp, bool(fd.get("matched")),
                            )
                            # Carry the EMPLOYEE alongside the decision. The
                            # attendance stage runs after line-crossing state is
                            # updated and may not be reached on the same tick, and
                            # `pt.employee_id` can be rebound in between — acting
                            # on a stale approval against a newly-bound identity
                            # would write attendance the gate never approved for
                            # that person.
                            pt.pending_decision = (decision, _emp)
                            if decision.reason == "no_match":
                                unknown_id = unknown_faces.capture(
                                    camera_id=str(w.camera_id),
                                    embedding=match_face.get("embedding"),
                                    quality=quality,
                                    frame_bgr=frame,
                                    box=pt.box,
                                    best_score=fd.get("score"),
                                    best_margin=fd.get("margin"),
                                    best_employee_id=fd.get("employee_id"),
                                )
                                if unknown_id is not None and not pt.unknown_event_marked:
                                    unknown_attendance.record(
                                        camera_id=str(w.camera_id),
                                        purpose=w.camera_purpose,
                                        event_time=frame_ist_time,
                                        track_id=pt.track_id,
                                        unknown_face_id=unknown_id,
                                        crop_path=None,
                                        quality_score=getattr(quality, "score", None),
                                        match_score=fd.get("score"),
                                        match_margin=fd.get("margin"),
                                    )
                                    pt.unknown_event_marked = True
            if pt.matched:
                any_match = True

            # A MONITOR camera NEVER records attendance — it only detects, tracks
            # and labels people. Skip all attendance logic for it.
            if w.is_monitor:
                continue

            # (b) Attendance.
            #
            # Line crossing used to be an ALTERNATIVE trigger:
            #     trigger = pt.crossed or confirmed
            # so someone walking through the doorway was marked on whatever
            # identity happened to be bound to their track, with no evidence
            # requirement at all — the weakest path in the whole system, and the
            # one most likely to fire, because a crossing is guaranteed while a
            # good face is not.
            #
            # Crossing is now a DIRECTION signal, not an identity signal. The
            # evidence gate must pass either way; a crossing only removes the
            # requirement to wait for further observations once it has.
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

                pending = getattr(pt, "pending_decision", None)
                if pending is not None:
                    decision, decided_employee = pending
                    # Only act while the track still holds the identity the gate
                    # approved. A rebind between the two stages invalidates the
                    # approval; dropping it costs at most a few hundred
                    # milliseconds, since the next tick re-evaluates.
                    if (
                        decision.allowed
                        and decided_employee is not None
                        and pt.employee_id == decided_employee
                    ):
                        _mark_from_track(
                            w, pt, int(decided_employee), decision,
                            event_time=frame_ist_time,
                        )
                    pt.pending_decision = None

        # (a2) Identity anchoring — MONITOR cameras only, and only AFTER the
        #      attendance loop above has run. Gating on `is_monitor` guarantees a
        #      non-face match can never reach attendance: only ArcFace on an
        #      IN/OUT camera may mark. This purely puts a name on a box.
        if w.is_monitor and ptracks and (_REID_ENABLED or _SEAT_ANCHOR):
            self._anchor_identities(w, frame, ptracks, face_confirmed)
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
        # Attendance cameras get priority on the shared inference gate. A person
        # walks past an entrance in ~2 seconds; a MONITOR camera watches people
        # who sit still for minutes. Under the old first-come-first-served lock
        # the entrance queued behind the monitor cameras' face-crop passes,
        # which is why they were throttled to a 1.5s interval as a workaround.
        from app.services.inference_gate import set_inference_priority

        set_inference_priority(not w.is_monitor)
        logger.info(
            "Camera %s: Recognition thread started (inference priority=%s)",
            w.camera_id, "high" if not w.is_monitor else "low",
        )

        while not self._stop_evt.is_set():
            self._stop_evt.wait(max(0.02, w.analysis_interval))
            if self._stop_evt.is_set():
                break

            # Analysis paused → keep streaming video but skip the expensive AI.
            #
            # Person detection costs ~2.3s of CPU per frame on this hardware, and
            # several cameras analysing at once starve a 4-core box: scheduled
            # jobs were observed running 8 MINUTES late and the app appeared to
            # "shut down" when it was really just unresponsive. Pausing a room
            # camera's analysis frees that CPU instantly while the live picture
            # keeps working (the grab thread is cheap and is untouched).
            if getattr(w, "analysis_paused", False):
                continue

            with w._frame_lock:
                frame = w._latest_frame
                frame_ts = w._latest_frame_ts

            if frame is None:
                continue

            # Grayscale is computed ONCE here and reused by both the blur check
            # and the motion gate below (previously two full-frame conversions).
            # Keep frame validation inside the loop's recovery boundary: a
            # malformed/partially decoded frame must be skipped, not terminate
            # recognition for this camera permanently.
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception as exc:
                w.state.last_error = f"invalid frame: {exc}"
                logger.warning(
                    "Camera %s: skipping invalid frame: %s", w.camera_id, exc
                )
                continue

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
                    self._analyze_person(
                        w, frame, rgb, skip_faces=blurry, frame_ts=frame_ts,
                    )
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
                # Identification is side-effect free — attendance is decided by
                # services/attendance_gate from the WHOLE track's evidence.
                profile = w.profile()
                _match_ms = 0.0
                if not skip_recognition:
                    for track in tracks:
                        if track.consecutive_misses != 0 or track.face is None:
                            continue

                        # A recognised track used to be skipped forever once its
                        # attendance had been written. That welded the identity
                        # on: a wrong match owned the box for the rest of the
                        # track's life, and a tracker ID-switch handed that name
                        # to whoever inherited it. Keep re-verifying, at a slower
                        # cadence, so the fused template keeps improving and a
                        # wrong label can correct itself.
                        if track.attendance_marked and not track.needs_reverify(
                            _POST_MARK_REVERIFY_SEC
                        ):
                            continue

                        # ── Quality gate ────────────────────────────────────
                        # The one check that did not exist on this path. Before,
                        # any detected face — a 16px full profile, a motion-blur
                        # smear — was folded into the template and could name an
                        # employee. Pose in particular was unreachable: the value
                        # it would have read was hardcoded to 0 by a bad
                        # attribute lookup in face_service.
                        quality = face_quality.assess(
                            track.face, rgb, limits=profile.limits,
                        )
                        if not quality.ok:
                            logger.debug(
                                "QUALITY camera=%s track=%d rejected=%s %s",
                                w.camera_id, track.track_id, quality.reason,
                                quality.as_log_fields(),
                            )
                            continue

                        logger.debug(
                            "STAGE-identify camera=%s track=%d box=%s quality=%s",
                            w.camera_id, track.track_id, track.box,
                            quality.as_log_fields(),
                        )

                        # Accumulate into the track's fused template, weighted by
                        # QUALITY (not raw pixel width), then match on the fusion.
                        #
                        # Matching a single frame of a small face is the core
                        # accuracy problem on a fixed ceiling camera: the
                        # embedding is mostly noise. Combining the frames of an
                        # approach cuts that noise ~sqrt(N), and quality
                        # weighting means the sharp frontal frames dominate the
                        # blurred profiles instead of merely the larger ones.
                        match_face = track.face
                        if _EMBEDDING_FUSION:
                            accepted = track.add_observation(
                                track.face.get("embedding"), quality=quality.score,
                            )
                            if not accepted:
                                # The fuser judged this face inconsistent with
                                # the rest of the track — usually two people
                                # sharing one track. Do not match on it.
                                logger.info(
                                    "FUSION camera=%s track=%d observation rejected "
                                    "as outlier (accepted=%d rejected=%d) — likely "
                                    "a track switch",
                                    w.camera_id, track.track_id,
                                    track.fuser.accepted, track.fuser.rejected,
                                )
                                continue
                            fused = track.fused_embedding()
                            if fused is not None:
                                match_face = dict(track.face)
                                match_face["embedding"] = fused

                        # Match from the already-computed embedding (no re-detect,
                        # no attendance side effect).
                        _t_match0 = time.time()
                        result = recognize_face(
                            match_face,
                            threshold=profile.threshold,
                            source="cctv",
                            camera_id=str(w.camera_id),
                            camera_purpose=w.camera_purpose,
                            mark_attendance=False,
                            min_margin=profile.margin,
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

                        # Capture provenance NOW, while the frame and the match
                        # result still coexist. The attendance write happens
                        # later and off-thread, where neither is available.
                        if matched and emp_id is not None and not w.is_monitor:
                            track.last_evidence = _build_evidence(
                                frame, track.box, face_data, track.track_id,
                                emp_id, w.camera_id,
                            )

                        # ── Attendance decision ─────────────────────────────
                        decision = attendance_gate.evaluate(
                            track,
                            employee_id=emp_id,
                            matched=matched,
                            score=float(face_data.get("score") or 0.0),
                            margin=float(face_data.get("margin") or 0.0),
                            profile=profile,
                        )
                        _log_decision(
                            w, track.track_id, face_data, quality, decision,
                            emp_id, matched,
                        )

                        if decision.allowed and profile.marks_attendance:
                            _mark_from_track(
                                w, track, int(emp_id), decision,
                                event_time=_frame_capture_time(frame_ts),
                            )
                        elif (
                            not matched
                            and decision.reason == "no_match"
                            and profile.marks_attendance
                        ):
                            # A usable face that nobody in the gallery claims.
                            # Queued for review instead of discarded — this is
                            # the only signal that tells us WHICH employees the
                            # cameras keep failing on. See unknown_faces.
                            unknown_id = unknown_faces.capture(
                                camera_id=str(w.camera_id),
                                embedding=match_face.get("embedding"),
                                quality=quality,
                                frame_bgr=frame,
                                box=track.box,
                                best_score=face_data.get("score"),
                                best_margin=face_data.get("margin"),
                                best_employee_id=face_data.get("employee_id"),
                            )
                            if unknown_id is not None and not track.unknown_event_marked:
                                unknown_attendance.record(
                                    camera_id=str(w.camera_id),
                                    purpose=w.camera_purpose,
                                    event_time=_frame_capture_time(frame_ts),
                                    track_id=track.track_id,
                                    unknown_face_id=unknown_id,
                                    crop_path=None,
                                    quality_score=getattr(quality, "score", None),
                                    match_score=face_data.get("score"),
                                    match_margin=face_data.get("margin"),
                                )
                                track.unknown_event_marked = True

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
        # Edge-triggered: warn ONCE when the pipeline starts lagging, and once
        # again when it recovers — not on every encoded frame.
        self._age_warned = False

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
                frame_ts = w._latest_frame_ts
                tracks = list(w._latest_tracks)

            if frame is None:
                continue

            # How far behind real life this picture is. A flat value means the
            # pipeline keeps up; a steadily CLIMBING one means the reader is
            # losing to the source and the FFmpeg queue is growing without
            # bound — which ends in a stall. FPS looks healthy the whole time
            # this is happening, so it cannot be diagnosed without this number.
            age_ms = (time.time() - frame_ts) * 1000.0 if frame_ts else 0.0
            w.state.frame_age_ms = round(age_ms, 1)
            if age_ms > _FRAME_AGE_WARN_MS and not self._age_warned:
                self._age_warned = True
                logger.warning(
                    "Camera %s: frame age %.0fms (>%.0fms) — pipeline falling "
                    "behind the stream. Check CPU load, or move analysis to the "
                    "sub-stream.",
                    w.camera_id, age_ms, _FRAME_AGE_WARN_MS,
                )
            elif age_ms <= _FRAME_AGE_WARN_MS and self._age_warned:
                self._age_warned = False
                logger.info(
                    "Camera %s: frame age back to %.0fms — pipeline caught up",
                    w.camera_id, age_ms,
                )

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
        self.camera_purpose = camera_purpose.upper()   # "IN" | "OUT" | "MONITOR"
        # `threshold` is retained only for the status payload. The value that
        # actually gates a match comes from the live profile (with its own floor
        # and per-camera overrides) so a change takes effect without a restart —
        # see CameraWorker.profile().
        self.threshold = float(threshold)
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
        # time.time() when _latest_frame was decoded. Lets the display thread
        # report how far behind real life the picture is (state.frame_age_ms).
        self._latest_frame_ts: float = 0.0
        self._latest_tracks: list = []       # published by recog, drawn by display
        self._frame_counter = 0  # For frame skipping
        self._last_view_ts = 0.0  # last time the preview JPEG was requested

        # Per-employee last-marked timestamp for the camera-level cooldown.
        self._last_marked: dict[int, float] = {}

        # Attendance cameras (IN/OUT) analyse fast; display-only MONITOR cameras
        # analyse slowly so they don't starve the shared inference lock.
        self.is_monitor = self.camera_purpose == "MONITOR"
        # From the profile: MONITOR cameras analyse far more slowly so they do
        # not monopolise the shared inference gate and starve the attendance
        # cameras. Read once here because track-expiry frame counts below are
        # derived from it; the recognition loop re-reads the live value each tick.
        _profile = camera_profile.get_profile(camera_id, self.camera_purpose)
        self.analysis_interval = _profile.analysis_interval
        logger.info("Camera %s: profile %s", camera_id, _profile.describe())
        # When True the AI analysis is skipped but the video keeps streaming.
        # Lets an operator drop CPU load without deleting or stopping a camera.
        self.analysis_paused: bool = False

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
                # Permissive tracker for room cameras.
                #
                # A ceiling-mounted view of SEATED people produces detection
                # scores well below the strict config's new_track_thresh of
                # 0.30 — measured on this very room: 0.58 / 0.25 / 0.19 / 0.16,
                # so only ONE of four people became a track and the overlay
                # read "People: 1".
                #
                # This used to require listing camera ids in CCTV_STEEP_CAMERAS
                # by hand, so any monitor camera the operator forgot to add
                # silently saw a fraction of the room. MONITOR cameras now get
                # the permissive config by DEFAULT: they can never mark
                # attendance, so a spurious box on an empty chair is cosmetic,
                # whereas a missed person defeats the camera's only purpose.
                #
                # IN/OUT cameras keep the strict config — there a false track
                # feeds attendance, so precision matters more than recall.
                steep = self.is_monitor or str(camera_id) in _STEEP_CAMERAS
                # One engine (and therefore one YOLO model + one ByteTrack state)
                # PER CAMERA — tracker state must never be shared between feeds.
                # MONITOR cameras never mark attendance, so they may use a
                # lighter/faster model + smaller imgsz than the IN/OUT cameras.
                # Empty settings => identical model to before (no behaviour change).
                from app.core.config import get_settings as _get_settings
                _s_cfg = _get_settings()
                _mon_model = (getattr(_s_cfg, "yolo_monitor_model_path", "") or "") if self.is_monitor else ""
                _mon_imgsz = int(getattr(_s_cfg, "yolo_monitor_imgsz", 0) or 0) if self.is_monitor else 0
                self.bytetrack_engine = bytetrack_engine.ByteTrackEngine(
                    conf=(_STEEP_CONF if steep else _PERSON_CONF),
                    max_misses=_body_misses,
                    camera_id=str(camera_id),
                    tracker_cfg=(_STEEP_TRACKER_CFG if steep else None),
                    imgsz=(_mon_imgsz or None),
                    model_path=(_mon_model or None),
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

    # ── capture pacing ──────────────────────────────────────────────────────
    def retrieve_period(self) -> float:
        """Minimum seconds between DECODED frames for this camera.

        The stream is drained at full rate with grab() (that is what keeps the
        FFmpeg queue — and therefore latency — from growing), but a frame is
        only decoded when something will look at it. Two consumers:

          * the display thread, at _DISPLAY_FPS, and only while someone is
            actually watching (_DISPLAY_IDLE_SEC since the last preview request)
          * the recognition thread, at 1 / analysis_interval

        With nobody watching a MONITOR camera that is ~0.7 fps instead of the
        full stream rate. Across twelve cameras on four cores that is the
        difference between the decode threads getting CPU and being starved by
        inference — which is what makes the picture fall behind.
        """
        watched = (time.time() - self._last_view_ts) < _DISPLAY_IDLE_SEC
        display_fps = _DISPLAY_FPS if watched else 0.0
        analysis_fps = 0.0 if self.analysis_paused else 1.0 / max(0.02, self.analysis_interval)
        needed = max(display_fps, analysis_fps)
        if needed <= 0:
            # Nothing needs frames. Still decode occasionally so a viewer
            # arriving gets a current picture rather than a stale one.
            return 1.0
        return 1.0 / needed

    # ── recognition profile ─────────────────────────────────────────────────
    def profile(self):
        """This camera's live recognition profile (thresholds, quality limits,
        evidence requirements).

        Read every analysis tick rather than cached on the worker, so an
        administrator changing a threshold in the UI takes effect within seconds
        without restarting the process — which previously meant dropping every
        camera's stream to change one number. camera_profile does the caching.
        """
        return camera_profile.get_profile(self.camera_id, self.camera_purpose)

    @property
    def attendance_cooldown(self) -> float:
        return self.profile().attendance_cooldown

    # ── attendance cooldown ─────────────────────────────────────────────────
    def can_mark_attendance(self, employee_id: int) -> bool:
        """True if this employee is outside the per-camera cooldown window."""
        last = self._last_marked.get(employee_id)
        return last is None or (time.time() - last) >= self.attendance_cooldown

    def note_attendance_marked(self, employee_id: int) -> None:
        now = time.time()
        self._last_marked[employee_id] = now
        # Bound the dict. It only ever grew, for the process lifetime.
        if len(self._last_marked) > 256:
            cutoff = now - max(self.attendance_cooldown, _COOLDOWN_HINT) * 4
            self._last_marked = {
                emp: ts for emp, ts in self._last_marked.items() if ts > cutoff
            }

    def _warm_cooldown_from_db(self) -> None:
        """Seed the per-camera cooldown from recent events in the database.

        `_last_marked` is in-memory only, so it was cleared by a restart — and
        also by any camera edit, since `add_camera` REPLACES the worker. Either
        one reopened the cooldown window and let the same person be marked
        again immediately. Reading back the recent events closes that gap.

        Never raises: a camera must still start if this query fails.
        """
        try:
            from datetime import timedelta

            from app.core.datetime_utils import get_ist_now
            from app.db.session import SessionLocal
            from app.models import AttendanceEvent
            from app.services.attendance_event_service import business_date, to_naive_ist

            now_ist = get_ist_now()
            since = now_ist - timedelta(seconds=self.attendance_cooldown)
            with SessionLocal() as db:
                rows = (
                    db.query(AttendanceEvent.employee_id, AttendanceEvent.event_time)
                    .filter(
                        AttendanceEvent.camera_id == str(self.camera_id),
                        AttendanceEvent.attendance_date == business_date(now_ist),
                        AttendanceEvent.event_time >= since,
                    )
                    .all()
                )

            now_mono = time.time()
            for employee_id, event_time in rows:
                age = (now_ist - to_naive_ist(event_time)).total_seconds()
                # Back-date the monotonic stamp so the remaining cooldown is
                # what is actually left, not a fresh full window.
                self._last_marked[int(employee_id)] = now_mono - age
            if rows:
                logger.info(
                    "Camera %s: warmed attendance cooldown for %d employee(s)",
                    self.camera_id, len(rows),
                )
        except Exception:
            logger.exception(
                "Camera %s: cooldown warm-up failed (starting with an empty "
                "cooldown — a duplicate mark is possible in the next %.0fs)",
                self.camera_id, _COOLDOWN_HINT,
            )

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._stream_thread and self._stream_thread.is_alive():
            return
        # Restore the cooldown before any frame can be analysed, so a restart
        # cannot re-mark someone who was just marked.
        if not self.is_monitor:
            self._warm_cooldown_from_db()
        self._stream_thread = _StreamThread(self)
        self._recog_thread  = _RecognitionThread(self)
        self._display_thread = _DisplayThread(self)
        self._stream_thread.start()
        self._recog_thread.start()
        self._display_thread.start()
        logger.info(
            "Camera %s [%s]: Worker started (purpose=%s url=%s)",
            self.camera_id, self.name, self.camera_purpose,
            _redact_url(self.stream_url),
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
        # Report what the camera is ACTUALLY running, not the value it was
        # constructed with. They diverge as soon as an operator edits the
        # profile, and a status page showing a threshold the camera is not using
        # is worse than showing none — it is what lets a mis-set camera look
        # correct on the dashboard.
        profile = self.profile()
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "location": getattr(self, "location", None),
            # Redacted: this dict is returned by the camera status/list APIs and
            # is logged. The un-redacted URL stays available on the DB row for
            # the edit form (GET /cameras/{id}).
            "stream_url": _redact_url(self.stream_url),
            "source_type": self.source_type,
            "camera_purpose": self.camera_purpose,
            "threshold": profile.threshold,
            "match_margin": profile.margin,
            "min_face_px": profile.limits.min_face_px,
            "min_observations": profile.min_observations,
            "marks_attendance": profile.marks_attendance,
            "profile_summary": profile.describe(),
            "interval_sec": self.interval_sec,
            "status": s.status,
            "last_error": s.last_error,
            "fps": s.fps,
            "capture_fps": s.fps,          # frames GRABBED per second (stream rate)
            "retrieve_fps": s.retrieve_fps,  # frames actually DECODED per second
            "display_fps": s.display_fps,
            # How stale the operator's picture is. Watch this, not FPS: a
            # climbing frame_age_ms is the early warning for a stalling stream.
            "frame_age_ms": s.frame_age_ms,
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
                        "Camera %s: Failed to parse HCNetSDK config from %s",
                        model.id, _redact_url(model.source_url),
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
        # Worst frame age across cameras — the single best indicator that the
        # pipeline is falling behind the streams (see CameraRuntimeState).
        worst_age = max((w.state.frame_age_ms for w in workers), default=0.0)

        # ── People currently tracked, per camera ────────────────────────────
        #
        # This is BODY tracks, not faces: a person with their back to the lens
        # is counted. That is the whole point -- the face pipeline reports zero
        # for someone walking away, which is indistinguishable from an empty
        # room unless the body count is surfaced separately.
        #
        # `people_detected` is deliberately NOT called occupancy. These cameras
        # overlap (the DVR channels cover the same floor), so somebody standing
        # where two views meet is counted twice. Deduplicating across cameras
        # needs cross-camera Re-ID, which this does not attempt. It answers
        # "what is each camera seeing right now", which is the question a single
        # number burned into one video overlay could never answer across four
        # feeds.
        #
        # `analysis_age_sec` is the honest part: a count is only as current as
        # the last completed analysis pass. On this hardware a pass costs
        # seconds, so a stale count must be visibly stale rather than silently
        # wrong -- that is exactly how a plainly visible person came to read as
        # "People: 0".
        now_ts = time.time()
        by_camera = []
        for w in workers:
            updated = float(getattr(w.state, "updated_at", 0.0) or 0.0)
            by_camera.append({
                "camera_id": w.camera_id,
                "name": w.name,
                "purpose": w.camera_purpose,
                "people": int(w.state.active_tracks or 0),
                "body_tracking": bool(w.use_person_tracking),
                "status": w.state.status,
                "analysis_age_sec": round(now_ts - updated, 1) if updated else None,
            })
        by_camera.sort(key=lambda c: str(c["camera_id"]))

        tracking = [c for c in by_camera if c["body_tracking"]]

        stats = {
            "ffmpeg_ok": self._ffmpeg_ok,
            "total_cameras": total,
            "running_cameras": running,
            "error_cameras": error,
            "total_frames_processed": frames,
            "total_reconnects": reconnects,
            "worst_frame_age_ms": round(worst_age, 1),
            # Sum over body-tracking cameras only. A face-only camera reporting
            # 0 would otherwise drag the total down while meaning "not measured".
            "people_detected": sum(c["people"] for c in tracking),
            "cameras_body_tracking": len(tracking),
            "people_by_camera": by_camera,
        }
        try:
            from app.services.inference_gate import get_gate

            # high_avg_wait_ms is the number that matters: if attendance
            # cameras are still queueing, the slot count or the detector cost
            # (FACE_DETECTION_SIZE) needs attention.
            stats["inference_gate"] = get_gate().stats()
        except Exception:
            logger.debug("inference gate stats unavailable", exc_info=True)
        return stats

    def is_ffmpeg_ok(self) -> bool:
        if self._ffmpeg_ok is None:
            self._ffmpeg_ok = _check_ffmpeg()
        return bool(self._ffmpeg_ok)


# ---------------------------------------------------------------------------
# Module-level singleton used by all API routes and lifespan hooks
# ---------------------------------------------------------------------------
camera_manager = CameraManager()
