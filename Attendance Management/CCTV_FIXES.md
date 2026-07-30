# CCTV Subsystem — Remediation Plan

Companion to [CCTV_REVIEW.md](./CCTV_REVIEW.md). That document says *what is wrong*; this one
says *how to fix it*, with the actual patch for each issue.

**Baseline:** commit `cecf7a9`, branch `master`.

**How to read an entry:** every fix has **Root cause → Patch → Verify → Risk**. Do not skip
*Verify* — several of these fail silently if applied incorrectly, which is how they got here.

---

## Fix order (and why)

| Tier | Scope | Ships as | Effort |
|---|---|---|---|
| **1** | Restore service — 6 surgical patches, no behaviour change anyone relies on | one PR | ~1 day |
| **2** | Attendance integrity — changes semantics, needs a migration | 3–4 PRs | 2–4 weeks |
| **3** | Performance, scale, UX — architectural | ongoing | 1–3 months |

Two sequencing rules that matter more than the individual patches:

1. **C1 and C2 are one bug wearing two hats.** The URL-query FFmpeg options (C2) mean the stream
   runs over UDP and stalls; the un-reset `last_frame_time` (C1) means it can never recover from
   that stall. Fixing either alone leaves you broken. Ship both together.
2. **Do A6 (evidence columns + snapshot) *before* touching any recognition threshold.** Every
   threshold in the codebase today was set from anecdote. Once `match_score` / `match_margin` are
   persisted per event, one week of production data tells you the real separation between true
   and false matches, and the tuning in §5 stops being guesswork.

---

# TIER 1 — Restore service

## C1 · Reconnect livelock

### Root cause

`_StreamThread.run` resets `consecutive_failures`, `reconnect_delay` and `status` on a successful
open, but not `w.state.last_frame_time`. The stale watchdog immediately below then compares the
brand-new connection against the *previous* session's timestamp, finds it >15 s old, and tears
the connection down before `cap.read()` is ever reached.

Two further problems compound it: the stale branch `continue`s with **no back-off**, so this is a
hot loop; and the `last_frame_time > 0` guard means a camera that connects but never delivers a
first frame is never caught at all.

### Patch

`backend/app/services/camera_service.py` — in `_StreamThread.run`, at the end of the successful-open
block (currently lines 654-658):

```python
                consecutive_failures = 0
                reconnect_delay = _RECONNECT_INIT_DELAY
                w.state.status = "running"
                w.state.last_error = None
                # CRITICAL: restart the watchdog clock on every successful open.
                # Without this the stale check below compares this new connection
                # against the PREVIOUS session's timestamp, finds it stale, and
                # tears it down before the first read -> permanent connect/drop
                # loop with the camera never streaming again.
                # It also arms the watchdog for the "connected but silent" case:
                # if no frame arrives within _STALE_TIMEOUT of the open, we
                # reconnect instead of blocking forever inside cap.read().
                w.state.last_frame_time = time.time()
                logger.info("Camera %s [%s]: Connected successfully", w.camera_id, w.name)
```

And give the stale branch a back-off so it cannot hot-loop (currently lines 660-676):

```python
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
                w.state.reconnect_count += 1
                # Back off. Previously this `continue`d immediately, so a camera
                # that could not deliver frames opened a fresh RTSP session as
                # fast as the DVR would accept one — Hikvision units cap
                # concurrent sessions, so this locked out other clients too.
                reconnect_delay = min(reconnect_delay * 1.5, _RECONNECT_MAX_DELAY)
                self._stop_evt.wait(reconnect_delay)
                continue
```

### Verify

```bash
# Watch a single camera come up. You want ONE "Connected successfully" followed by
# frame counters climbing — NOT a connect/stale pair on the same millisecond.
grep -E "Camera 53.*(Connected|No frame|frames)" app-run.log | tail -30
```

Expected within 60 s of restart: one `Connected successfully`, then
`Camera 53: 200 frames, 12.4 FPS, reconnects=N` lines every ~16 s. If you still see
`Connected successfully` immediately followed by `No frame for 15s`, the patch did not apply.

### Risk

None. Strictly a bug fix — no configuration or downstream behaviour changes.

---

## C2 · FFmpeg options never reach FFmpeg

### Root cause

`_open_capture` builds a query string and appends it to the RTSP URL:

```
rtsp://…/Streaming/Channels/201?rtsp_transport=tcp&fflags=nobuffer&…
```

OpenCV does not parse URL query strings as FFmpeg options. The DVR receives them as part of the
request URI and ignores them. TCP transport, `nobuffer` and `low_delay` have therefore **never
been active** — streams run over the DVR's default (usually UDP), and packet loss produces the
stalls that C1 then makes permanent.

OpenCV's actual mechanism is the `OPENCV_FFMPEG_CAPTURE_OPTIONS` environment variable, read by
the FFmpeg backend when a capture is created. Format: `key;value` pairs joined by `|`.

### Patch

`backend/app/services/camera_service.py` — add near the tuning constants (~line 74), at **module
import time**, before any `VideoCapture` is constructed:

```python
def _ffmpeg_capture_options() -> str:
    """FFmpeg options for cv2.CAP_FFMPEG, in the format OpenCV actually reads.

    OpenCV takes these from OPENCV_FFMPEG_CAPTURE_OPTIONS as `key;value` pairs
    joined by `|`. They CANNOT be passed as a URL query string — appending
    `?rtsp_transport=tcp` to the URL just sends that text to the DVR as part of
    the request URI, which is what this code used to do (so TCP transport was
    never actually enabled).

    NOTE on the connect timeout key: FFmpeg renamed `stimeout` -> `timeout` for
    the RTSP demuxer in 5.0. We emit BOTH; the demuxer ignores the one it does
    not recognise. Confirm with `ffmpeg -h demuxer=rtsp` for your build.
    """
    micros = _OPEN_TIMEOUT_MS * 1000
    default = "|".join([
        "rtsp_transport;tcp",     # TCP — no packet loss, the whole point of this
        "rtsp_flags;prefer_tcp",
        "fflags;nobuffer",        # do not accumulate a decode buffer
        "flags;low_delay",
        "reorder_queue_size;0",   # do not wait to reorder late RTP packets
        f"stimeout;{micros}",     # FFmpeg < 5 connect/read timeout (microseconds)
        f"timeout;{micros}",      # FFmpeg >= 5 equivalent
        "analyzeduration;2000000",
        "probesize;2000000",
    ])
    return os.getenv("CCTV_FFMPEG_OPTS", default)


# Must be set BEFORE the first VideoCapture is created — the FFmpeg backend
# reads it at capture-construction time. (CCTV_FFMPEG_OPTS was documented in
# this module's docstring but read nowhere; it now works as advertised.)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _ffmpeg_capture_options()
logger.info("FFmpeg capture options: %s", os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"])
```

Then simplify `_open_capture` (replacing lines 486-509):

```python
    # RTSP / HTTP — FFmpeg backend. Transport and timeout options come from
    # OPENCV_FFMPEG_CAPTURE_OPTIONS (set at module import); they must NOT be
    # appended to the URL.
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.info("Camera %s: capture opened=%s", camera_id, cap.isOpened())
    return cap
```

### Verify

```bash
# 1. The option string is applied at startup:
grep "FFmpeg capture options" app-run.log

# 2. The DVR now shows a TCP session rather than UDP. On the server:
netstat -ano | findstr ":554"     # expect ESTABLISHED TCP to the DVR IP
```

The old `Using FFmpeg options: rtsp_transport=tcp&…` log line disappears — that line was the
symptom, printing options that were being thrown away.

### Risk

Low. If your DVR genuinely does not support RTSP-over-TCP (rare; Hikvision does), set
`CCTV_FFMPEG_OPTS` with `rtsp_transport;udp` to override. Because the value is now honoured, a
*bad* value can break streaming where before it was ignored — check the startup log line after
first deploy.

---

## C3 · Capture timeouts have no effect

### Root cause

`cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, …)` is called *after* `cv2.VideoCapture(...)` has
already performed a synchronous connect. The property arrives too late to affect the open that
just happened. An unreachable DVR therefore blocks on FFmpeg's internal default (tens of
seconds) — in `recognize-cctv-frame` that blocks a FastAPI request worker.

### Patch

**Camera workers:** already fixed by C2 — `stimeout`/`timeout` in the capture options are the
mechanism that actually bounds the connect.

**Request-path endpoints** need a hard ceiling as well, because a hung open must never occupy a
request worker. Add to `backend/app/services/camera_service.py`:

```python
def open_capture_with_timeout(
    stream_url: str, source_type: str, camera_id: int, timeout_sec: float = 12.0
) -> Optional[cv2.VideoCapture]:
    """Open a capture, giving up after `timeout_sec` no matter what FFmpeg does.

    CAP_PROP_OPEN_TIMEOUT_MSEC cannot be used for this: it is set on the object
    AFTER the constructor has already blocked on the connect. The FFmpeg-level
    timeout (see _ffmpeg_capture_options) is the primary bound; this is the
    backstop that keeps a wedged open off the request thread.

    Returns an OPEN capture, or None. The caller owns release().
    """
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"open-{camera_id}")
    fut = pool.submit(_open_capture, stream_url, source_type, camera_id)
    try:
        cap = fut.result(timeout=timeout_sec)
    except FuturesTimeout:
        # Do NOT pool.shutdown(wait=True) — the thread is stuck inside FFmpeg.
        # Let it finish and release in the background.
        logger.error("Camera %s: open timed out after %.0fs", camera_id, timeout_sec)
        fut.add_done_callback(
            lambda f: (f.exception() is None and f.result() is not None) and f.result().release()
        )
        pool.shutdown(wait=False)
        return None
    except Exception:
        logger.exception("Camera %s: open failed", camera_id)
        pool.shutdown(wait=False)
        return None
    pool.shutdown(wait=False)
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        return None
    return cap
```

with `from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout` at the
top of the module.

Then use it in `backend/app/api/routes/cameras.py::test_camera_connection` (replacing the
`cv2.VideoCapture` block at lines 212-234) and in
`backend/app/api/routes/recognition.py::recognize_cctv_frame`:

```python
    cap = open_capture_with_timeout(stream_url, source_type, camera_id=0, timeout_sec=12.0)
    if cap is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not open stream within 12s. Check: DVR IP, RTSP port 554, "
                "credentials, and that the DVR's RTSP service is enabled."
            ),
        )
```

Better still for `recognize-cctv-frame`: make the route `async def` and wrap the whole
capture-and-recognise in `await asyncio.to_thread(...)`, so even the bounded version never
occupies the event loop.

### Verify

```bash
# Point test-connection at a black-holed IP; it must return 503 in ~12s, not hang.
curl -m 20 -X POST localhost:8000/api/cameras/test-connection \
  -H 'Authorization: Bearer <token>' -H 'Content-Type: application/json' \
  -d '{"stream_url":"rtsp://10.255.255.1:554/x","source_type":"rtsp"}'
```

### Risk

Low. The leaked thread on timeout is bounded (it exits when FFmpeg gives up) and the
`add_done_callback` releases the capture if the open eventually succeeds.

---

## C4 · RTSP credentials in plaintext logs

### Root cause

`camera_service.py:480` logs the full source URL at INFO on every connect attempt. With C1
looping, `app-run.log.err` accumulated 26 MB containing
`rtsp://anilchanna:test%40123@192.168.29.181:554/…` thousands of times.

### Patch

**Step 1 — redact at the call site.** Add to `backend/app/services/camera_service.py`:

```python
from urllib.parse import urlparse, urlunparse


def _redact_url(url: str) -> str:
    """Strip the password from a stream URL so it never reaches a log or an API
    response. Keeps the username — it is useful for diagnosis and is not a secret.
    """
    try:
        p = urlparse(url)
    except Exception:
        return "<unparseable stream url>"
    if not p.hostname:
        return "<redacted>" if "@" in url else url
    netloc = p.hostname + (f":{p.port}" if p.port else "")
    if p.username:
        netloc = f"{p.username}:***@{netloc}"
    return urlunparse(p._replace(netloc=netloc))
```

Then replace every log/serialisation of a stream URL:

| File | Line | Change |
|---|---|---|
| `camera_service.py` | 480 | `logger.info("Camera %s: Opening stream: %s", camera_id, _redact_url(source))` |
| `camera_service.py` | 508 | delete (dead after C2) |
| `camera_service.py` | 1514 | `url=%s` → `_redact_url(self.stream_url)` |
| `camera_service.py` | 1571 | `serialize_state()` → `"stream_url": _redact_url(self.stream_url)` |
| `dvr_manager.py` | 322-325 | already splits on `@`; switch to `_redact_url` for consistency |
| `api/routes/recognition.py` | 195-199 | stop logging the decoded URL |

**Step 2 — belt and braces.** A logging filter that scrubs anything that slips through, in
`backend/app/main.py` before `basicConfig`:

```python
import re

_CRED_RE = re.compile(r"(?P<scheme>\w+://)(?P<user>[^:/@\s]+):(?P<pw>[^@/\s]+)@")


class RedactCredentialsFilter(logging.Filter):
    """Scrub `scheme://user:password@host` from every log record.

    Defence in depth: call sites should redact explicitly (see _redact_url), but
    a single missed f-string used to be enough to leak the DVR password on every
    reconnect. This catches those.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if "://" in msg and "@" in msg:
            record.msg = _CRED_RE.sub(r"\g<scheme>\g<user>:***@", msg)
            record.args = ()
        return True
```

Attach it to the root handler (see the Logging fix below, which installs handlers explicitly).

**Step 3 — remediate what already leaked.**

```bash
# The existing logs contain the DVR password. Treat them as compromised.
cd "c:/sakshi folder/application/Resume analyzer"
rm -f app-run.log.err app-run.log        # or archive to an access-controlled location
```

Then **change the DVR account password** and update the `cameras.source_url` rows. Confirm
`*.log` is in `.gitignore` so this never reaches the repository.

### Verify

```bash
grep -cE "rtsp://[^:]+:[^@]+@" app-run.log     # must be 0
```

### Risk

None functionally. Note that `serialize_state()` feeding the API means the **camera manager UI
will stop showing the password** in the stream-URL field — that is the intent, but tell the
operators, since the edit form reads that value. Keep the un-redacted value available only on the
authenticated `GET /cameras/{id}` detail route if editing requires it.

---

## C5 · HCNetSDK DVR preview endpoints crash

### Root cause

`HCNetSDKCameraWorker` exposes `get_latest_frame()` and `state.latest_jpeg`, but no
`get_latest_jpeg()`. `cameras.py:765` and `:808` call `camera.worker.get_latest_jpeg()`
unconditionally → `AttributeError` → 500. `CameraManager` guards the same call with `hasattr`
(`camera_service.py:1760`), which is why only the DVR routes fail.

The real fix is to stop having two worker classes with divergent interfaces.

### Patch

**Step 1 — give the HCNetSDK worker the missing method.** In
`backend/app/services/hcnetsdk_camera.py`, add `self._last_view_ts = 0.0` to `__init__`
(alongside `self._frame_counter`, ~line 205) and this method next to `get_latest_frame`
(~line 745):

```python
    def get_latest_jpeg(self) -> Optional[bytes]:
        """Latest annotated JPEG. Mirrors CameraWorker.get_latest_jpeg so both
        worker types satisfy the same interface — the DVR routes call this
        without knowing which class they hold.
        """
        self._last_view_ts = time.time()   # keeps the DVR idle-reaper from culling us
        with self._frame_lock:
            return self.state.latest_jpeg
```

**Step 2 — remove the duplicated `hasattr` dispatch.** Define the contract once so it cannot
drift again. New file `backend/app/services/camera_worker_base.py`:

```python
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class CameraWorkerProtocol(Protocol):
    """Interface every camera worker must satisfy.

    Both CameraWorker (RTSP/USB) and HCNetSDKCameraWorker implement this. The
    API routes depend on THIS, never on a concrete class — a missing method is
    then a type error at review time instead of a 500 in production (which is
    exactly how get_latest_jpeg went missing on the HCNetSDK worker).
    """

    camera_id: int
    name: str

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def is_alive(self) -> bool: ...
    def get_latest_jpeg(self) -> Optional[bytes]: ...
    def serialize_state(self) -> dict: ...
```

Then simplify `CameraManager.get_latest_jpeg` / `get_status` / `list_statuses`
(`camera_service.py:1755-1806`) to call the methods directly, and in `cameras.py` replace the
`camera.worker or camera.rtsp_worker` dance with a single `dvr_manager.get_worker(channel_id)`
helper that returns `Optional[CameraWorkerProtocol]`.

Add a test that pins the contract:

```python
# backend/tests/test_camera_worker_contract.py
import pytest
from app.services.camera_service import CameraWorker
from app.services.camera_worker_base import CameraWorkerProtocol

@pytest.mark.parametrize("cls_path", [
    "app.services.camera_service:CameraWorker",
    "app.services.hcnetsdk_camera:HCNetSDKCameraWorker",
])
def test_worker_satisfies_protocol(cls_path):
    mod, name = cls_path.split(":")
    cls = getattr(__import__(mod, fromlist=[name]), name)
    for method in ("start", "stop", "is_alive", "get_latest_jpeg", "serialize_state"):
        assert callable(getattr(cls, method, None)), f"{name} is missing {method}()"
```

### Verify

`pytest backend/tests/test_camera_worker_contract.py`, then open a DVR channel configured as
`source_type=hcnetsdk` in the dashboard — the preview must render rather than 500.

### Risk

Low. Step 1 alone fixes the crash; step 2 is the refactor that stops it recurring and can follow
in a separate PR if you want the hotfix out first.

---

## Logging · rotation + hot-path noise

### Root cause

`logging.basicConfig` only (`main.py:39`) — no rotation, no retention. Every `recognize_face`
emits four INFO lines (`recognition.py:371, 401, 269, 423`), per face, per analysis tick, per
camera. At 8 ticks/s × 3 cameras that is ~100 lines/second. `app-run.log.err` reached 26 MB and
would eventually fill the disk and take the application down.

### Patch

Replace `basicConfig` in `backend/app/main.py:39-44`:

```python
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(os.getenv("HRMS_LOG_DIR", PROJECT_ROOT / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter(
    # Date included: the old "%H:%M:%S" format made a multi-day log impossible
    # to correlate — every day looked like the same 24 hours.
    fmt="%(asctime)s.%(msecs)03d %(levelname)s %(name)s [%(threadName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_file = RotatingFileHandler(
    LOG_DIR / "hrms.log", maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8",
)
_file.setFormatter(_fmt)
_console = logging.StreamHandler()
_console.setFormatter(_fmt)

_redact = RedactCredentialsFilter()          # see C4
_file.addFilter(_redact)
_console.addFilter(_redact)

_root = logging.getLogger()
_root.handlers[:] = [_file, _console]        # replaces uvicorn's default handler
_root.setLevel(logging.INFO)

# The recognition pipeline logs STEP-1..STEP-11 per face per frame. That is a
# debugging trace, not an operational log — at 3 cameras it was ~100 lines/sec
# and produced a 26 MB file in a few hours. Keep it available, off by default.
logging.getLogger("app.services.recognition").setLevel(
    os.getenv("RECOGNITION_LOG_LEVEL", "WARNING").upper()
)
```

Then in `backend/app/services/recognition.py`, demote the per-frame trace: `STEP-1`
(line 371), `STEP-2` (401), `STEP-3` both branches (208, 269), `STEP-4` (220), `STEP-5` (226) and
`STEP-11` (423) become `logger.debug`. Keep at INFO only the ones that record a **state change**:
`STEP-6 attendance_trigger`, `STEP-9 database_insert_successful`, `STEP-10`, and every
`logger.warning` / `logger.exception`.

Same treatment in `camera_service.py`: `PIPELINE` (873) and `MULTI-FACE` (1249) → DEBUG;
`IDENTITY`, `ATTENDANCE`, `LINE-CROSS` and connection state changes stay INFO.

### Verify

```bash
# Idle system with cameras running: well under 10 lines/minute at INFO.
wc -l logs/hrms.log && sleep 60 && wc -l logs/hrms.log
ls -la logs/            # hrms.log + up to 5 rotations, none over 50 MB
```

Re-enable the full trace when debugging with `RECOGNITION_LOG_LEVEL=DEBUG`.

### Risk

None, provided you keep DEBUG reachable via the env var — the STEP trace is genuinely useful and
should not be deleted, only silenced by default.

---

## API · `create_camera` silently drops fields

### Root cause

`CameraCreateRequest` accepts `frame_skip`, `tracking_max_distance` and `tracking_cooldown`
(`cameras.py:64-66`), but the `CameraConfig(...)` construction at `cameras.py:299-308` never
passes them. The API returns 200 and the values vanish. Separately, the line-crossing fields
exist on the model and the worker but appear in **no** request schema, so the feature is
reachable only by direct SQL.

### Patch

`backend/app/api/routes/cameras.py` — add the missing fields to both request models:

```python
class CameraCreateRequest(BaseModel):
    # ... existing fields ...
    # Doorway line crossing. Fully implemented in person_tracker.check_line_crossing
    # and CameraWorker, but previously absent from every schema — so it could only
    # be enabled with a manual UPDATE against the cameras table.
    crossing_enabled: bool = False
    line_orientation: str = Field(default="horizontal", pattern="^(horizontal|vertical)$")
    line_position: float = Field(default=0.5, ge=0.0, le=1.0)
    entry_direction: str = Field(default="down", pattern="^(up|down|left|right)$")
```

and persist **every** field rather than an ad-hoc subset:

```python
        source_url = _validate_camera_source(payload.source_url, payload.source_type)
        data = payload.model_dump()
        data["source_url"] = source_url
        # camera_type is a legacy NOT NULL column kept in sync with camera_purpose
        # until it can be dropped (see the schema-drift note in CCTV_REVIEW.md §9).
        data["camera_type"] = data["camera_purpose"]
        cam = CameraConfig(**data)
        db.add(cam)
```

This construction fails loudly if a schema field has no column, instead of dropping it silently.
Add a regression test:

```python
def test_create_camera_persists_all_fields(client, db):
    r = client.post("/api/cameras", json={
        "name": "T", "source_url": "rtsp://192.168.1.9:554/s", "camera_purpose": "IN",
        "frame_skip": 3, "tracking_cooldown": 7.5, "crossing_enabled": True,
        "line_position": 0.4,
    })
    cam = db.query(CameraConfig).get(r.json()["id"])
    assert (cam.frame_skip, float(cam.tracking_cooldown)) == (3, 7.5)
    assert cam.crossing_enabled and float(cam.line_position) == 0.4
```

Apply the same to `update_camera` (it already uses `model_dump(exclude_unset=True)` + `setattr`,
so it only needs the new schema fields), and add the same four fields to the camera form in
`frontend/src/pages/CctvCameraManager.tsx`.

### Verify

Create a camera with `crossing_enabled: true` via the API, then confirm the worker log shows the
crossing line active and the overlay renders the cyan line.

### Risk

Low. `CameraConfig(**data)` will raise if `payload` ever gains a field with no matching column —
that is the point, but it means schema and model must be changed together.

---

## API · DVR `set_recognition_enabled` is a no-op

### Root cause

`dvr_manager.py:397-400` sets `worker.recognition_enabled = enabled`. Neither `CameraWorker` nor
`HCNetSDKCameraWorker` ever reads that attribute — Python happily creates it and nothing happens.
The dashboard toggle has never done anything.

### Patch

`CameraWorker` already has the mechanism under a different name: `analysis_paused`
(`camera_service.py:1416`), honoured by the recognition thread at line 1040. Wire the toggle to
it:

```python
    def set_recognition_enabled(self, channel_id: int, enabled: bool) -> bool:
        # ...
            camera.recognition_enabled = enabled
            worker = camera.worker or camera.rtsp_worker
            if worker is not None:
                # `recognition_enabled` was set on the worker and read by nobody.
                # `analysis_paused` is the flag the recognition thread actually
                # checks — pausing it stops detection/recognition while the video
                # keeps streaming, which is what this toggle promises.
                worker.analysis_paused = not enabled
                logger.info(
                    "DVR channel %d: analysis %s", channel_id,
                    "resumed" if enabled else "paused",
                )
            return True
```

Add `self.analysis_paused: bool = False` to `HCNetSDKCameraWorker.__init__` and an early
`if self.analysis_paused: continue` in its `_run` loop before
`self._process_frame_for_recognition(frame)` (`hcnetsdk_camera.py:464-466`).

### Verify

Toggle recognition off in the dashboard; video keeps playing, boxes stop updating, and
per-camera CPU drops. Log shows `analysis paused`.

### Risk

None.

---

# TIER 2 — Attendance integrity

## A6 · Evidence trail (do this first)

### Why first

Every recognition threshold in the codebase was set from a single anecdote. Persisting the score
and margin on each event gives you the distribution of true vs. false matches within a week, so
§5's tuning becomes measurement instead of guesswork. It is also what makes a disputed attendance
record adjudicable.

### Patch

**Migration** `backend/alembic/versions/031_attendance_event_evidence.py`:

```python
"""Add recognition evidence to attendance events."""
from alembic import op
import sqlalchemy as sa

revision = "031_attendance_event_evidence"
down_revision = "030_add_camera_seat_assignments"


def upgrade():
    op.add_column("attendance_events", sa.Column("match_score", sa.Float(), nullable=True))
    op.add_column("attendance_events", sa.Column("match_margin", sa.Float(), nullable=True))
    op.add_column("attendance_events", sa.Column("track_id", sa.Integer(), nullable=True))
    op.add_column("attendance_events", sa.Column("snapshot_path", sa.String(300), nullable=True))
    # Nullable throughout: existing rows have no evidence and must not be
    # invented. A NULL score means "recorded before evidence capture existed".


def downgrade():
    for col in ("snapshot_path", "track_id", "match_margin", "match_score"):
        op.drop_column("attendance_events", col)
```

Mirror on `AttendanceEvent` in `backend/app/models/attendance.py`.

**Snapshot capture** — new `backend/app/services/attendance_snapshot.py`:

```python
"""Persist the face crop that produced an attendance event.

~5 KB per event. This is the only artefact that makes a disputed record
reviewable, and the labelled data set that lets thresholds be tuned from
measurement rather than anecdote.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SNAPSHOT_ROOT = Path(__file__).resolve().parents[2] / "data" / "attendance_snapshots"
_PAD = 0.35          # context around the face box — an operator needs to see the person
_JPEG_Q = 85
RETENTION_DAYS = 90  # prune with the closeout job below


def save_face_snapshot(
    frame_bgr: np.ndarray, box, employee_id: int, camera_id: str, when
) -> Optional[str]:
    """Write the cropped face and return a path relative to SNAPSHOT_ROOT.

    Never raises: failing to save evidence must not block the attendance write.
    """
    try:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        px, py = int((x2 - x1) * _PAD), int((y2 - y1) * _PAD)
        crop = frame_bgr[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
        if crop.size == 0:
            return None
        rel = Path(when.strftime("%Y-%m-%d")) / f"{employee_id}_{camera_id}_{when.strftime('%H%M%S')}.jpg"
        out = SNAPSHOT_ROOT / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), crop, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_Q]):
            return None
        return str(rel).replace("\\", "/")
    except Exception:
        logger.exception("snapshot failed emp=%s camera=%s", employee_id, camera_id)
        return None


def prune_snapshots(retention_days: int = RETENTION_DAYS) -> int:
    """Delete snapshot day-folders older than the retention window."""
    import shutil
    from datetime import timedelta

    cutoff = date.today() - timedelta(days=retention_days)
    removed = 0
    if not SNAPSHOT_ROOT.exists():
        return 0
    for day_dir in SNAPSHOT_ROOT.iterdir():
        try:
            if day_dir.is_dir() and date.fromisoformat(day_dir.name) < cutoff:
                shutil.rmtree(day_dir, ignore_errors=True)
                removed += 1
        except ValueError:
            continue      # not a date-named folder — leave it alone
    return removed
```

**Thread the evidence through.** `_submit_attendance` (`camera_service.py:247`) currently takes
only ids; extend it to carry the evidence, and pass `frame`, `face["box"]`, `score`, `margin` and
`track.track_id` from both call sites (the face path at `:1224` and the body path `_mark` at
`:885`). `mark_cctv_attendance` → `_mark_attendance` → `record_face_attendance` →
`add_attendance_event` gain an `evidence: dict | None = None` parameter written onto the
`AttendanceEvent` row.

### Verify

Mark an attendance from a live camera, then:

```sql
SELECT id, employee_id, event_type, match_score, match_margin, track_id, snapshot_path
FROM attendance_events ORDER BY id DESC LIMIT 5;
```

`snapshot_path` must resolve to a readable JPEG of that person's face.

### Risk

Disk growth: ~5 KB × events/day × 90 days (a 200-person office ≈ 400 MB/quarter). `prune_snapshots`
bounds it. Snapshots are **biometric personal data** — store them inside the existing
access-controlled `data/` tree, serve them only through an authenticated route, and confirm the
retention window against your local data-protection obligations before enabling.

---

## C6 · Night shift recorded as next-day check-in

### Root cause

`add_attendance_event` keys events to `now_dt.date()` (`attendance_event_service.py:483`). Someone
leaving at 00:30 is evaluated against a fresh calendar day where their state is `ABSENT`; with
`attendance_checkin_on_missing_in=True`, `resolve_camera_event` returns `CHECK_IN`. Leaving the
building creates a check-in, and the real day is left open.

### Patch

Introduce an explicit business-day boundary. `backend/app/core/config.py`:

```python
    # Attendance days start at this hour (IST). An event before it belongs to the
    # PREVIOUS day, so a shift running past midnight stays on one record instead
    # of the exit being read as the next day's entry. Set to 0 to disable.
    attendance_day_start_hour: int = 5
```

`backend/app/services/attendance_event_service.py`:

```python
def business_date(dt: datetime, day_start_hour: int | None = None) -> date:
    """The attendance day an event belongs to.

    A calendar date is the wrong key for shift work: an exit at 00:30 lands on a
    day where the employee has no check-in, and the OUT camera's event is then
    resolved as a CHECK_IN for that new day (see resolve_camera_event: state
    ABSENT + allow_missing_in -> CHECK_IN). Anchoring to a day-start hour keeps
    the whole shift on the day it began.
    """
    if day_start_hour is None:
        from app.core.config import get_settings
        day_start_hour = get_settings().attendance_day_start_hour
    d = to_naive_ist(dt)
    if day_start_hour and d.hour < day_start_hour:
        return (d - timedelta(days=1)).date()
    return d.date()
```

Replace **every** `now_dt.date()` / `now_naive.date()` used as an attendance-day key:

| Location | Change |
|---|---|
| `add_attendance_event:483` | `d = business_date(now_dt)` |
| `is_within_event_cooldown:122` | `today = business_date(now_dt)` |
| `add_attendance_event:475` | `get_or_create_attendance(db, employee_id, business_date(now_dt))` |
| `calculate_intervals_from_events:283` | `if business_date(last_t) == business_date(now):` |
| `validate_event_time:435` | `if business_date(event_time) > business_date(get_ist_now()):` (also fixes A5 — it currently compares against server-local `date.today()`) |

Tests:

```python
@pytest.mark.parametrize("clock,expected_day", [
    ("2026-07-28 09:15", "2026-07-28"),   # normal morning
    ("2026-07-28 23:50", "2026-07-28"),   # late but before midnight
    ("2026-07-29 00:30", "2026-07-28"),   # exit after midnight -> previous day
    ("2026-07-29 04:59", "2026-07-28"),   # still the night shift
    ("2026-07-29 05:00", "2026-07-29"),   # boundary: new day
])
def test_business_date(clock, expected_day):
    assert business_date(datetime.fromisoformat(clock), 5).isoformat() == expected_day


def test_after_midnight_exit_is_an_out_not_a_checkin(db, employee):
    add_attendance_event(db, employee.id, datetime(2026, 7, 28, 21, 0), camera_purpose="IN")
    ev, rec, action = add_attendance_event(
        db, employee.id, datetime(2026, 7, 29, 0, 30), camera_purpose="OUT",
    )
    assert action == "BREAK_OUT"                     # was "CHECK_IN" before the fix
    assert rec.date == date(2026, 7, 28)
```

### Verify

Beyond the tests, check for the historical damage this bug produced:

```sql
-- Spurious check-ins created by someone leaving after midnight.
SELECT employee_id, attendance_date, event_time, event_type, camera_id
FROM attendance_events
WHERE event_type IN ('IN','CHECK_IN')
  AND EXTRACT(hour FROM event_time) < 5
ORDER BY event_time DESC;
```

### Risk

**This is the highest-risk change in the plan.** `attendance_date` is the grouping key for every
report, the payroll export and the monthly grid. Consequences:

- Set `attendance_day_start_hour = 0` to get exactly today's behaviour — deploy with `0`, verify
  nothing shifts, then raise it to `5`.
- Historical rows are **not** rewritten. Past days keep their old grouping, so a report spanning
  the change will show the old boundary before it and the new one after. If a backfill is needed,
  write it as a separate reviewed migration, not as part of this change.
- If any employee genuinely starts work before 05:00, that hour must be lower than their earliest
  start or their check-in lands on the previous day. Confirm against actual shift patterns.

---

## C7 · No end-of-day closeout

### Root cause

`calculate_intervals_from_events` adds an open interval from the last work-start to *now*
(`attendance_event_service.py:276-287`). If the OUT camera misses someone, hours accrue until
midnight and the day then freezes with `sign_out_time = NULL` and an inflated total. The only
scheduled job in the app is the DSR reminder (`main.py:205`).

### Patch

New `backend/app/services/attendance_closeout.py`:

```python
"""Close attendance days the OUT camera never closed.

Cameras miss departures routinely (back of head, tailgating, steep angles). The
open-interval rule in calculate_intervals_from_events then accrues hours until
midnight and freezes the day with no sign-out. This job closes those days with a
clearly-marked synthetic event so the record is honest: it says "the system
closed this, please review", not "the employee left at this time".
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timedelta

from app.core.config import get_settings
from app.core.datetime_utils import get_ist_now
from app.db.session import SessionLocal
from app.models import AttendanceEvent
from app.services.attendance_event_service import (
    business_date, current_state, recalculate_attendance_summary, to_naive_ist,
)

logger = logging.getLogger(__name__)

CLOSEOUT_SOURCE = "AUTO_CLOSE"


def run_closeout(now: datetime | None = None) -> int:
    """Close every still-open attendance day older than the cutoff. Idempotent."""
    s = get_settings()
    now = to_naive_ist(now or get_ist_now())
    cutoff_hour = int(getattr(s, "attendance_closeout_hour", 23))
    target_day = business_date(now - timedelta(hours=24))

    closed = 0
    with SessionLocal() as db:
        open_emp_ids = [
            r[0] for r in db.query(AttendanceEvent.employee_id)
            .filter(AttendanceEvent.attendance_date == target_day)
            .distinct().all()
        ]
        for emp_id in open_emp_ids:
            events = (
                db.query(AttendanceEvent)
                .filter(
                    AttendanceEvent.employee_id == emp_id,
                    AttendanceEvent.attendance_date == target_day,
                )
                .order_by(AttendanceEvent.event_time.asc(), AttendanceEvent.id.asc())
                .all()
            )
            if not events:
                continue
            last = events[-1]
            if current_state(last.event_type) != "WORKING":
                continue                      # already closed
            if last.source == CLOSEOUT_SOURCE:
                continue                      # our own event — idempotency guard

            # Close at the LAST SEEN time, not the cutoff: crediting a full day to
            # someone the camera stopped seeing at 14:00 would silently inflate
            # payroll. Capping at last-seen under-reports instead, which is the
            # safe direction and is visibly wrong so HR corrects it.
            close_at = min(
                to_naive_ist(last.event_time),
                datetime.combine(target_day, time(cutoff_hour, 0)),
            )
            db.add(AttendanceEvent(
                employee_id=emp_id,
                attendance_record_id=last.attendance_record_id,
                attendance_date=target_day,
                event_time=close_at,
                event_type="OUT",
                source=CLOSEOUT_SOURCE,
                camera_id=None,
            ))
            db.flush()
            rec = recalculate_attendance_summary(db, emp_id, target_day)
            rec.source = "CORRECTION"          # surfaces in the UI as needing review
            db.commit()
            closed += 1
            logger.warning(
                "AUTO-CLOSE employee_id=%s day=%s closed_at=%s "
                "(OUT camera missed the departure — needs HR review)",
                emp_id, target_day, close_at.isoformat(),
            )

    if closed:
        logger.warning("Attendance closeout: %d day(s) closed for %s", closed, target_day)
    return closed
```

Register it in `backend/app/main.py::_start_background_scheduler`, next to the DSR job:

```python
        from apscheduler.triggers.cron import CronTrigger

        sched.add_job(
            _attendance_closeout_tick,
            trigger=CronTrigger(hour=2, minute=30, timezone="Asia/Kolkata"),
            id="attendance_closeout",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=3600,   # a restart must not skip the night's run
        )
```

with

```python
async def _attendance_closeout_tick() -> None:
    import asyncio
    from app.services.attendance_closeout import run_closeout
    from app.services.attendance_snapshot import prune_snapshots
    try:
        await asyncio.to_thread(run_closeout)      # blocking DB work off the loop
        await asyncio.to_thread(prune_snapshots)
    except Exception:
        logger.exception("attendance closeout tick failed")
```

Run at 02:30, i.e. **after** the 05:00 business-day boundary has passed for the day being closed,
so a genuine night shift is never truncated mid-shift.

### Verify

```python
def test_closeout_uses_last_seen_not_cutoff(db, employee):
    add_attendance_event(db, employee.id, datetime(2026,7,27,9,0),  camera_purpose="IN")
    add_attendance_event(db, employee.id, datetime(2026,7,27,14,0), camera_purpose="OUT")
    add_attendance_event(db, employee.id, datetime(2026,7,27,14,5), camera_purpose="IN")
    assert run_closeout(now=datetime(2026,7,28,2,30)) == 1
    rec = get_attendance(db, employee.id, date(2026,7,27))
    assert rec.sign_out_time == time(14,5)     # NOT 23:00
    assert run_closeout(now=datetime(2026,7,28,2,31)) == 0   # idempotent
```

### Risk

Medium. The job writes attendance rows. Mitigations: `source="AUTO_CLOSE"` makes every synthetic
event identifiable and reversible with one `DELETE`; the idempotency guard prevents repeats;
closing at last-seen under-reports rather than over-reports. Run it in dry-run (log only, no
commit) for a week and review what it *would* have closed before letting it write.

---

## A2 · Attendance write failures are silently lost

### Root cause

`_mark` (`camera_service.py:885-892`) sets `pt.attendance_marked = True` and records the cooldown,
then submits the DB write to a 2-worker executor whose **return value is discarded**
(`_submit_attendance:247-267`). If the write fails, the track never retries and nothing surfaces.

### Patch

Make the submission inspect the outcome and retry the transient cases:

```python
# Outcomes meaning "no event was written, and retrying cannot help": the state
# machine or business rules refused it. Anything else that failed is transient.
_TERMINAL_ACTIONS = {
    "monitor_camera", "cooldown", "unknown",
    "duplicate_check_in_already_working", "duplicate_out_already_away",
    "check_out_without_check_in",
}
_RETRY_ACTIONS = {"attendance_failed", "validation_failed"}


def _submit_attendance(
    employee_id: int, camera_id: str, camera_purpose: str,
    evidence: dict | None = None, attempt: int = 1, max_attempts: int = 3,
) -> None:
    """Queue an attendance write off the recognition thread.

    Previously this discarded the result, so a DB failure lost the event
    permanently: the track had already set attendance_marked=True and the
    cooldown was already recorded, so nothing ever retried.
    """
    from app.services.recognition import mark_cctv_attendance

    def _run() -> None:
        t0 = time.time()
        try:
            _payload, action = mark_cctv_attendance(
                employee_id, camera_id=camera_id,
                camera_purpose=camera_purpose, evidence=evidence,
            )
        except Exception:
            logger.exception("ATTN-WRITE crashed emp=%s camera=%s", employee_id, camera_id)
            action = "attendance_failed"

        took = (time.time() - t0) * 1000
        if action in _RETRY_ACTIONS and attempt < max_attempts:
            delay = 2 ** attempt
            logger.warning(
                "ATTN-WRITE retry %d/%d in %ds emp=%s camera=%s action=%s",
                attempt, max_attempts, delay, employee_id, camera_id, action,
            )
            t = threading.Timer(
                delay, _submit_attendance,
                args=(employee_id, camera_id, camera_purpose, evidence, attempt + 1, max_attempts),
            )
            t.daemon = True
            t.start()
            return
        if action in _RETRY_ACTIONS:
            # Exhausted. Loud, and countable by monitoring — this is a lost
            # attendance event and somebody has to key it in by hand.
            logger.error(
                "ATTN-WRITE LOST emp=%s camera=%s purpose=%s action=%s after %d attempts "
                "— attendance NOT recorded, manual entry required",
                employee_id, camera_id, camera_purpose, action, max_attempts,
            )
            _ATTENDANCE_LOST.labels(camera_id=str(camera_id)).inc()   # see Metrics
            return
        logger.info(
            "ATTN-WRITE done emp=%s camera=%s purpose=%s action=%s took=%.0fms",
            employee_id, camera_id, camera_purpose, action, took,
        )

    _attendance_executor.submit(_run)
```

Also bound the queue so a DB stall cannot grow memory without limit:

```python
_attendance_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("CCTV_ATTENDANCE_WRITERS", "2")),
    thread_name_prefix="attn-write",
)
_ATTENDANCE_QUEUE_MAX = int(os.getenv("CCTV_ATTENDANCE_QUEUE_MAX", "500"))
# before submit():
if _attendance_executor._work_queue.qsize() > _ATTENDANCE_QUEUE_MAX:
    logger.error("ATTN-WRITE queue overflow (%d) — dropping write for emp=%s",
                 _attendance_executor._work_queue.qsize(), employee_id)
    return
```

### Verify

Stop PostgreSQL, trigger a recognition, restart PostgreSQL within ~6 s: the log must show two
retries and then a successful write. Leave it down longer and you must get exactly one
`ATTN-WRITE LOST` line.

### Risk

Low. Retries are idempotent-safe — the DB cooldown and state machine reject a duplicate if the
first attempt actually committed before failing on the response.

---

## A1 · Cooldown lost on restart

### Root cause

`_last_marked` (`camera_service.py:1408`) is a per-worker in-memory dict, cleared by restart,
reconnect, or any config change (`add_camera` replaces the worker).

### Patch

The DB-level directional cooldown (`EVENT_COOLDOWN_SECONDS = 25`) already survives restarts and
is the real guard. Two changes make it sufficient:

1. Raise it and make it configurable — 25 s is shorter than the time a person spends in view of a
   doorway camera:

   ```python
   # app/core/config.py
   attendance_event_cooldown_seconds: int = 90
   ```

   and have `add_attendance_event` default `cooldown_seconds` from settings rather than the
   module constant.

2. Warm the in-memory cooldown from the DB on worker start, so a restart does not reopen the
   window:

   ```python
       def _warm_cooldown_from_db(self) -> None:
           """Seed the per-camera cooldown from recent DB events.

           Without this, restarting the service (or merely editing a camera,
           which replaces the worker) clears _last_marked and lets the same
           person be re-marked immediately.
           """
           try:
               from app.db.session import SessionLocal
               from app.models import AttendanceEvent
               from app.services.attendance_event_service import business_date, to_naive_ist
               from app.core.datetime_utils import get_ist_now

               since = get_ist_now() - timedelta(seconds=_ATTENDANCE_COOLDOWN)
               with SessionLocal() as db:
                   rows = (
                       db.query(AttendanceEvent.employee_id, AttendanceEvent.event_time)
                       .filter(
                           AttendanceEvent.camera_id == str(self.camera_id),
                           AttendanceEvent.attendance_date == business_date(get_ist_now()),
                           AttendanceEvent.event_time >= since,
                       ).all()
                   )
               now_mono, now_wall = time.time(), to_naive_ist(get_ist_now())
               for emp_id, ev_time in rows:
                   age = (now_wall - to_naive_ist(ev_time)).total_seconds()
                   self._last_marked[int(emp_id)] = now_mono - age
               if rows:
                   logger.info("Camera %s: warmed cooldown for %d employee(s)",
                               self.camera_id, len(rows))
           except Exception:
               logger.exception("Camera %s: cooldown warm-up failed", self.camera_id)
   ```

   Call it at the top of `CameraWorker.start()`.

Also prune `_last_marked` periodically — it currently grows for the process lifetime (bounded by
headcount, so minor, but free to fix):

```python
    def note_attendance_marked(self, employee_id: int) -> None:
        now = time.time()
        self._last_marked[employee_id] = now
        if len(self._last_marked) > 256:
            cut = now - _ATTENDANCE_COOLDOWN * 4
            self._last_marked = {k: v for k, v in self._last_marked.items() if v > cut}
```

### Risk

Low. Raising the cooldown to 90 s could suppress a genuine rapid IN→OUT→IN, but the cooldown is
already **directional** (`is_within_event_cooldown`, line 139), so opposite transitions still pass.

---

## A3 · IN/OUT state machine strands people

### Root cause

Once someone is `AWAY`, every subsequent OUT is rejected as `duplicate_out_already_away`
(`resolve_camera_event:205`). Because the OUT camera misses re-entries regularly, one missed IN
means the real end-of-day departure is dropped and `sign_out_time` freezes at the earlier OUT.

### Patch

Make the rejection time-aware. A second OUT thirty seconds later is a duplicate; a second OUT
three hours later means the re-entry was missed:

```python
# A repeat OUT this long after the previous one is not a duplicate — it means the
# IN camera missed the person coming back. Rejecting it strands them on the wrong
# side and freezes their sign-out at the earlier departure.
MISSED_ENTRY_GRACE_SECONDS = 30 * 60


def resolve_camera_event(
    camera_type: str,
    last_type: str | None,
    allow_missing_in: bool,
    seconds_since_last: float | None = None,
) -> tuple[str | None, str | None]:
    state = current_state(last_type)
    cam = (camera_type or "IN").upper()
    stale = (
        seconds_since_last is not None
        and seconds_since_last > MISSED_ENTRY_GRACE_SECONDS
    )

    if cam == "IN":
        if state == "ABSENT":
            return "CHECK_IN", None
        if state == "AWAY":
            return "BREAK_IN", None
        if stale:
            # Already WORKING but the last event is hours old: the OUT was missed.
            # Synthesise the departure so the record stays coherent.
            return "BREAK_IN", "recovered_missed_out"
        return None, "duplicate_check_in_already_working"

    if state == "WORKING":
        return "BREAK_OUT", None
    if state == "ABSENT":
        return ("CHECK_IN", None) if allow_missing_in else (None, "check_out_without_check_in")
    if stale:
        return "BREAK_OUT", "recovered_missed_in"
    return None, "duplicate_out_already_away"
```

Callers pass the elapsed time, and when a `recovered_*` reason comes back the event is written
**and** flagged (`rec.source = "CORRECTION"`) so HR sees the day needs review. Note the second
return slot changes meaning from "reject reason" to "reason", so update the `if reject:` branch
at `attendance_event_service.py:517` to reject only when `new_type is None`.

### Risk

Medium — it makes the system accept events it used to reject. The 30-minute grace is the safety
margin; tune it from the `duplicate_*` rejection rate in your logs before rolling out.

---

## A4 · Fixed 09:00 / 18:00 workday

`_apply_late_and_early` (`attendance_event_service.py:344-362`) hardcodes both times. Add
`work_start_time` / `work_end_time` to the company config model (alongside the existing
`grace_time_minutes`), read them there, and — when shift work is needed — add a `shifts` table
with a per-employee assignment, falling back to the company default. This is a schema and
product decision rather than a bug fix; scope it separately.

---

## §5 · Recognition accuracy

**Do not change these numbers until A6 has collected a week of `match_score` / `match_margin`
data.** With that in hand:

```sql
-- The separation you are actually working with.
SELECT event_type,
       percentile_cont(0.05) WITHIN GROUP (ORDER BY match_score) AS p05,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY match_score) AS p50,
       min(match_score), count(*)
FROM attendance_events
WHERE match_score IS NOT NULL AND event_time > now() - interval '7 days'
GROUP BY event_type;
```

Review the snapshots for the lowest-scoring 20 events. Every one that is the wrong person sets
your floor.

Target end state, applied in this order:

| Setting | Now | Target | Precondition |
|---|---|---|---|
| Anti-spoof gate | none | enabled | — do this first, it is the biggest gap |
| `_MIN_FACE_PX` | 16 | 50 | camera repositioned to eye level |
| `_CONFIRM_FRAMES` | 1 | 3 | face size fixed, else nobody confirms |
| `_IDENT_CONFIRM` | 1 | 2 | same |
| `_MIN_THRESHOLD` | 0.35 | 0.45 | measured p05 of true matches above it |

**Anti-spoof integration.** A passive model (MiniFASNet / Silent-Face, ~2 ms CPU) is the only
control that stops a printed photo at the IN camera. Gate it *before* the attendance write, never
before the display:

```python
# in _RecognitionThread, immediately before w.note_attendance_marked(...)
from app.services.liveness import is_live

if not w.is_monitor and not is_live(frame, track.box):
    logger.warning(
        "SPOOF-REJECT camera=%s track=%d employee=%s — attendance NOT recorded",
        w.camera_id, track.track_id, track.employee_name,
    )
    _SPOOF_REJECTED.labels(camera_id=str(w.camera_id)).inc()
    continue
```

Run it in log-only mode first (record the verdict, do not block) so you can measure the
false-reject rate on real employees before it starts refusing attendance.

**Camera placement is the highest-leverage change here** and needs no code: IN/OUT cameras at eye
level, 1.5–3 m from the subject, facing the approach path, with the light behind the camera. The
`_STEEP_CAMERAS` special case, the 3× upscale in `_face_in_person_crop` and the 16 px floor all
exist to compensate for mounting that cannot deliver a usable face. No threshold recovers
information the optics never captured.

---

# TIER 3 — Performance, scale, UX

Design direction rather than patches; each needs its own design note.

### Drop the global inference lock

`face_service._inference_lock` serialises all detection and embedding process-wide. Replace with
a **per-camera ONNX session** (thread-local `FaceAnalysis` instances) and cap threads per session
via `intra_op_num_threads` so N sessions do not oversubscribe the cores. The
`_MONITOR_ANALYSIS_INTERVAL` hack exists only to work around this lock and can be removed
afterwards.

### Use the sub-stream

`dvr_manager.py:282` hardcodes `…/{ch:03d}01` (main stream). Hikvision sub-streams are `…02`. Make
it a per-camera column (`stream_profile: main|sub`), default `sub` for analysis. This alone cuts
decode and detection cost 4–9× and is the cheapest performance win available.

### Push instead of poll

- MJPEG (`cameras.py:501-522`): replace the 30 Hz poll with a `threading.Condition` the display
  thread notifies, so viewers wake only on a new frame.
- Live dashboard (`LiveAttendanceDashboard.tsx:85`): replace the 30 s `setInterval` with SSE
  published from the attendance writer.
- Camera manager (`CctvCameraManager.tsx:229`): use the existing MJPEG endpoint instead of
  cache-busting JPEG URLs.

### Move the overlay to the client

`_draw_enhanced_overlay` burns boxes into the JPEG server-side — unstyleable, unreadable at tile
size, and CPU spent per camera per frame. Publish tracks as JSON on the SSE channel and draw them
on a `<canvas>` over the video.

### Split the vision pipeline out of the API process

`CameraManager._workers` is process-local, so the deployment cannot scale horizontally
(`main.py:349-352` documents the pin). Target: a `cctv-worker` service owning 2–4 cameras,
publishing recognitions over Redis/NATS; the API process holds no camera state. This is also the
prerequisite for GPU inference on a dedicated node.

### Metrics

Add Prometheus counters and alert on them — they replace most of what the log spam was used for:

| Metric | Alert |
|---|---|
| `cctv_camera_up{camera_id}` | any camera down > 5 min |
| `cctv_reconnects_total` | rate > 1/min (catches C1-class regressions) |
| `cctv_inference_seconds` | p95 > 1 s |
| `cctv_attendance_marked_total` | zero for > 2 h during working hours |
| `cctv_attendance_lost_total` | any (A2) |
| `cctv_spoof_rejected_total` | spike |

---

# Deployment checklist — Tier 1

```
[ ] Back up the DB and archive existing logs to a controlled location
[ ] Apply C1, C2, C3, C4, C5, logging, create_camera, recognition-toggle patches
[ ] Confirm the startup line: "FFmpeg capture options: rtsp_transport;tcp|..."
[ ] Confirm ONE "Connected successfully" per camera, then climbing frame counters
[ ] Confirm no connect/stale pair on the same millisecond
[ ] grep -cE "rtsp://[^:]+:[^@]+@" logs/hrms.log   -> 0
[ ] Confirm log growth < 10 lines/min at idle; rotation files appear
[ ] Walk past the IN camera -> attendance event within ~2 s
[ ] Walk past the OUT camera -> OUT event, correct direction
[ ] Open every camera in the dashboard; each renders live video
[ ] Change the DVR account password; update cameras.source_url
```

**Rollback:** every Tier 1 change is behavioural-neutral and independently revertible. The one to
watch is C2 — if a DVR refuses TCP, set `CCTV_FFMPEG_OPTS` with `rtsp_transport;udp` rather than
reverting the patch, since reverting also restores the livelock.
