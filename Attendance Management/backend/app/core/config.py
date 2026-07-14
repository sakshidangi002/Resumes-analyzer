"""
Application configuration. Database is PostgreSQL; email uses simple SMTP.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from functools import lru_cache

# Load .env into os.environ.
#
# Pydantic's `env_file` only feeds the Settings object below — it does NOT put the
# variables into os.environ. The camera pipeline reads its tuning with plain
# os.getenv("CCTV_..."), so without this every CCTV_* line in .env was silently
# ignored and the hard-coded defaults were used instead (i.e. editing .env had no
# effect whatsoever). Loading it here — in the module everything imports — makes
# both config styles see the same values.
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"   # app/core/ -> backend/.env
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH, override=False)


class Settings(BaseSettings):
    """Load from environment. Use .env for local overrides."""

    # App
    app_name: str = "Attendance & HRMS"
    debug: bool = False

    # PostgreSQL (override via .env on each system)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = ""
    postgres_db: str = "attendance_hrms"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # JWT
    # SECURITY: no default â€” must come from environment / .env so that a
    # missing config fails fast at startup instead of silently signing tokens
    # with a guessable placeholder. Recommended: `openssl rand -hex 32`.
    secret_key: str
    algorithm: str = "HS256"
    # Long-lived session: the user stays logged in until they explicitly log
    # out (default 30 days). Override with ACCESS_TOKEN_EXPIRE_MINUTES.
    access_token_expire_minutes: int = 60 * 24 * 30

    # SMTP (simple SMTP for all email)
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    smtp_from_email: str = "noreply@company.com"
    smtp_from_name: str = "HRMS"

    # Comma-separated address(es) that receive HR-bound notifications
    # (e.g. "new leave request"). If empty, falls back to all users with the
    # HR role in the database.
    hr_notification_email: str = ""

    # Public URL where the HRMS is reachable (used in outbound emails like the
    # 5 PM DSR reminder). Leave empty in dev; in production set e.g.
    #   APP_BASE_URL=https://hrms.softwiz.local
    app_base_url: str = ""

    # ---- Web Push (VAPID) -------------------------------------------------
    # Generate once with `python scripts/gen_vapid_keys.py` and paste the
    # output into your .env. Browsers verify pushes against the public key
    # registered at subscribe time; the private key signs each push.
    vapid_public_key: str = ""
    vapid_private_key: str = ""
    vapid_claim_email: str = "mailto:noreply@company.com"

    # Face recognition (webcam / upload)
    # 0.45/0.05 was too permissive for CCTV and allowed impostor matches.
    # A stricter margin requires the top candidate to clearly beat the runner-up
    # before a match is accepted, which — together with CCTV stable confirmation
    # — prevents false attendance for people who are not present.
    default_threshold: float = 0.45
    min_match_margin: float = 0.10

    # If an employee is recognised at the OUT camera but has no check-in today
    # (their entrance read was missed), record the missed check-in instead of
    # dropping them entirely. Prevents "recognised but marked Absent".
    attendance_checkin_on_missing_in: bool = True

    # ---- Face detector backend -------------------------------------------
    # "insightface" -> SCRFD detector from buffalo_l (default, no extra deps)
    # "yolo"        -> YOLOv8-face for DETECTION; ArcFace (buffalo_l) still
    #                  produces the recognition embedding.
    # Enabling "yolo" requires `pip install ultralytics` and a face weights
    # file (with 5 landmarks) at FACE_YOLO_MODEL_PATH, e.g. yolov8n-face.pt.
    face_detector: str = "insightface"
    yolo_face_model_path: str = "models/yolov8n-face.pt"
    yolo_conf: float = 0.35

    # ---- Person (body) tracking -------------------------------------------
    # When enabled AND the model files exist, CCTV workers detect & track whole
    # bodies (OpenCV DNN MobileNet-SSD, CPU-friendly, no torch). A face that is
    # recognised is bound to the person's track, so the employee's name stays on
    # them even when the face turns away — until they leave the frame.
    person_tracking_enabled: bool = False
    person_model_proto: str = "models/mobilenet_ssd/deploy.prototxt"
    person_model_weights: str = "models/mobilenet_ssd/mobilenet_iter_73000.caffemodel"
    person_conf: float = 0.5          # person-detection confidence threshold
    person_reverify_sec: float = 5.0  # re-check a bound identity every N seconds
    # YOLO11 + ByteTrack (preferred for office monitoring; needs `ultralytics`).
    # Falls back to MobileNet-SSD + IoU tracking when unavailable.
    # Model size matters far more than thresholds for this seated/occluded office
    # view. Measured on a live dev-room frame at imgsz 960:
    #   yolo11s -> 4 detections, but scores 0.58 / 0.25 / 0.19 / 0.16
    #              => only ONE clears the 0.30 track threshold (the room showed
    #                 "People: 1"). The rest sit down in empty-chair noise, so no
    #                 threshold can rescue them without also boxing furniture.
    #   yolo11m -> 3 detections, scores 0.66 / 0.52 / 0.34
    #              => ALL THREE clear the threshold. ~50% slower (6.1s vs 4.1s),
    #                 which is affordable now that imgsz dropped 1600 -> 960.
    # Do NOT drop the thresholds to compensate for a weak model — upgrade the model.
    # ONNX Runtime, NOT PyTorch. PyTorch is a *training* runtime — on this CPU
    # yolo11m.pt takes ~7s per frame, so every box on screen was up to 7 SECONDS
    # STALE: a person walking across the room had her box drawn where she used to
    # be, several metres behind her. The identical model exported to ONNX runs
    # ~2.7x faster with the same detections and the same ByteTrack ids (verified),
    # cutting the lag to ~2.7s. Detection quality is unchanged — this is purely a
    # faster way to execute the same network.
    # Re-export after swapping models:
    #   YOLO('models/yolo11m.pt').export(format='onnx', imgsz=960, simplify=True)
    yolo_person_model_path: str = "models/yolo11m.onnx"
    # Inference size. After the dev-room camera was re-aimed, people are ~250px
    # tall (was 120-180), and 960 was measured to detect them just as well as 1600
    # (0.67/0.61/0.37 vs 0.65/0.60/0.36) at roughly HALF the cost. Smaller = faster
    # analysis = names appear on screen sooner.
    yolo_person_imgsz: int = 960
    # NMS IoU. Ultralytics defaults to 0.7, which is too permissive for this
    # ceiling view: two overlapping boxes on ONE person (e.g. a tight box on the
    # torso plus an oversized one running down over the chair/bag) both survive,
    # so a single person gets two boxes and two track ids. 0.5 merges them;
    # measured lower (0.3) starts suppressing genuinely separate people.
    yolo_person_iou: float = 0.5
    # Nested-box suppression — DISABLED (a value > 1.0 can never match).
    #
    # It existed to kill "bloated" boxes (a person merged with their chair/bag).
    # Those only appeared at the OLD steep camera angle, at low confidence. After
    # the camera was re-aimed the raw detections are clean, so the filter has no
    # duplicates left to remove — but it DOES do harm: in a top-down view people
    # sit one behind another, so one person's box legitimately nests inside
    # another's, and the rule silently deleted real people (3 detections collapsed
    # to 1 track). Confidence (new_track_thresh) is the safe guard instead.
    # Set to e.g. 0.9 only if bloated duplicate boxes ever return.
    person_nested_contain: float = 1.01
    person_nested_area_ratio: float = 1.6
    # Draw ONLY people detected in the latest analysis cycle.
    #
    # This was briefly True to stop boxes blinking out when a chair hid a seated
    # person. It is now False because that side-effect is worse than the problem:
    # a "held" track is kept for several missed cycles, and each cycle takes ~13s
    # on this CPU (two monitor cameras share one inference lock) — so a NAMED box
    # sat on an empty chair for up to two minutes after the person walked away.
    #
    # Detection is now reliable enough (yolo11m finds every seated person) that
    # holding is unnecessary. The track itself still survives inside ByteTrack
    # (track_buffer), so a person who is briefly hidden keeps their ID and NAME
    # when they reappear — it just isn't DRAWN while they cannot be seen.
    person_publish_held: bool = False
    # Torch device for person detection: "" = auto (CUDA if present, else CPU),
    # or pin explicitly e.g. "cpu" / "0".
    yolo_person_device: str = ""
    # Tuned ByteTrack config. Falls back to ultralytics' built-in bytetrack.yaml
    # when the file is missing.
    bytetrack_config_path: str = "models/bytetrack_person.yaml"

    # ---- Person Re-Identification (cross-camera identity without a face) ----
    # OSNet appearance embeddings keep an employee's name on their body track when
    # their face is not visible. LABELLING ONLY — a body match never marks
    # attendance (only ArcFace on an IN/OUT camera does).
    reid_enabled: bool = True
    reid_model_path: str = "models/osnet_x0_25_msmt17.onnx"
    # Cosine similarity to accept a match against embeddings from the SAME camera
    # (same viewpoint → trustworthy).
    # Body Re-ID accept threshold.
    #
    # RAISED after a real false positive: a MAN was labelled "Saloni Pathania" at
    # 0.77. Measured evidence: a TRUE same-person match on the same camera scores
    # 0.93-0.96, while that wrong match scored 0.77. OSNet on a top-down seated
    # crop largely encodes CLOTHING COLOUR, so a light shirt matches a light top —
    # 0.65 was well inside the range where different people collide.
    # 0.88 sits above every false match seen and below every true one.
    # A wrong name is far worse than "Person #N": an unrecognised person simply
    # stays Unknown until a good frame comes along.
    reid_threshold: float = 0.75
    # CROSS-CAMERA Re-ID — ENABLED. Ch1 and Ch3 watch the SAME dev room from
    # opposite ends, so a person facing one camera has their back to the other.
    # This is what carries a name from the camera that saw their face to the one
    # that only sees their back.
    #
    # RE-MEASURED after the camera was re-aimed (the old steep angle gave
    # 0.477/0.505/0.496 for three DIFFERENT people — useless, which is why this
    # was previously off). With the current geometry:
    #     same person  : 0.508, 0.548, 0.659, 0.700
    #     other people : 0.387, 0.407, 0.452, 0.487
    # The bands barely touch, so the score alone is not enough — the MARGIN over
    # the runner-up (consistently 0.10-0.29) is the trustworthy signal, and both
    # must be satisfied.
    #
    # 0.55 proved TOO LOW in production — it bound a wrong name ("Adarsh Maurya")
    # at exactly 0.55. The measured genuine cross-camera matches reached 0.659 and
    # 0.700, so 0.68 still binds the strong frames while sitting far above both the
    # observed wrong-person band (<=0.487) and the 0.55 failure. It will bind a
    # little later, but it will not bind the wrong person — and once bound, the
    # identity sticks to the body track.
    reid_cross_camera_threshold: float = 0.88
    # Best must beat the runner-up by this, so two similarly-dressed people are
    # never confidently confused (mirrors the face matcher's margin rule). Raised
    # to 0.08 for the cross-camera case: measured true matches beat the runner-up
    # by 0.10-0.29, while the raw scores of right and wrong people nearly touch —
    # so the margin, not the score, is what actually separates them.
    reid_min_margin: float = 0.15
    # Seat anchoring: in a fixed-desk room, "who sits here" is the strongest signal.
    # Anchors are LEARNED from face matches — no zone configuration needed.
    # Seat anchoring — DISABLED.
    #
    # It labels whoever is sitting at a desk with the name of the person last
    # identified there. That is a WRONG-NAME risk: if a colleague borrows the chair
    # (or someone moves seats), they silently inherit the other person's identity.
    # A wrong name is worse than "Person #N", so identity must come only from a
    # real face match or a high-confidence body Re-ID.
    seat_anchor_enabled: bool = False
    seat_anchor_radius_px: int = 120

    # ---- DVR auto-start on application boot -------------------------------
    # When dvr_autostart is True and credentials are set, the app connects to
    # the Hikvision DVR and starts all camera streams automatically on startup,
    # so nobody has to press "Connect" in the UI.
    dvr_autostart: bool = False
    dvr_ip: str = ""
    dvr_port: int = 8000
    dvr_username: str = ""
    dvr_password: str = ""
    # Recognition threshold used by DVR camera workers. 0.05 (the old hard-coded
    # value) accepts near-random matches; 0.30–0.40 is realistic for CCTV.
    dvr_recognition_threshold: float = 0.35
    # Comma-separated DVR channel IDs that are CHECK-OUT cameras; every other
    # channel is treated as CHECK-IN. e.g. DVR_OUT_CHANNELS="2" makes channel 2
    # the exit camera and channel 1 the entrance camera.
    dvr_out_channels: str = ""
    # Comma-separated DVR channel IDs that are MONITOR (office) cameras — body
    # tracking + name display, NEVER attendance. e.g. DVR_MONITOR_CHANNELS="1".
    dvr_monitor_channels: str = ""
    # Doorway line-crossing for DVR camera workers (per-DVR, since DVR channels
    # are not CameraConfig rows). Requires person tracking to be enabled.
    dvr_crossing_enabled: bool = False
    dvr_line_orientation: str = "horizontal"
    dvr_line_position: float = 0.5
    dvr_entry_direction: str = "down"
    # Crossing direction for OUT cameras (people leaving usually move the
    # opposite way in-frame). Leave blank to reuse dvr_entry_direction.
    dvr_out_entry_direction: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()

