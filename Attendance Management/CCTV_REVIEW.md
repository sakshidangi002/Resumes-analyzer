# CCTV Subsystem Review

**Scope:** CCTV functionality only — camera workers, DVR integration, video streaming, face
detection, face recognition, tracking, attendance write path, CCTV API routes, and the React
CCTV pages. Non-CCTV features (payroll, leave, DSR, resume analyzer, etc.) are out of scope.

**Reviewed at:** commit `cecf7a9` (branch `master`)

---

## 1. Architecture as built

```
CameraConfig (DB) ──> CameraManager (singleton)
                        └── CameraWorker (per camera, 3 daemon threads)
                              ├── _StreamThread       cv2.VideoCapture → _latest_frame
                              ├── _RecognitionThread  detect → track → identify → attendance
                              └── _DisplayThread      draw overlay → JPEG → MJPEG endpoint
                        └── HCNetSDKCameraWorker (source_type=hcnetsdk, single thread)

DVRManager (separate singleton) ──> on-demand preview CameraWorkers (forced MONITOR)
```

Key source files:

| File | Role |
|---|---|
| `backend/app/services/camera_service.py` (1835 ln) | Camera manager, stream/recognition/display threads, overlay |
| `backend/app/services/face_service.py` | InsightFace (buffalo_l) / YOLO detection + ArcFace embedding |
| `backend/app/services/recognition.py` | Match → attendance orchestration, STEP-1..11 logging |
| `backend/app/services/match.py` | Cosine similarity, best-match + margin gate |
| `backend/app/services/embedding_cache.py` | In-process employee embedding gallery |
| `backend/app/services/face_tracker.py` | Centroid face tracker |
| `backend/app/services/person_tracker.py` | IoU body tracker + line crossing |
| `backend/app/services/bytetrack_engine.py` | YOLO11 + ByteTrack body tracking |
| `backend/app/services/attendance_event_service.py` | IN/OUT state machine, cooldown, daily summary |
| `backend/app/services/dvr_manager.py` | Hikvision DVR connect / on-demand preview streams |
| `backend/app/services/hcnetsdk_camera.py` | HCNetSDK direct DVR worker |
| `backend/app/api/routes/cameras.py` | Camera CRUD, preview, MJPEG, DVR endpoints |
| `backend/app/api/routes/recognition.py` | Upload/one-shot recognition endpoints |
| `frontend/src/pages/{CctvAttendance,CctvCameraManager,DvrCameraDashboard,FaceDetection,LiveAttendanceDashboard}.tsx` | Operator UI |

**Structural observation:** three parallel camera stacks exist — `camera_service.CameraWorker`,
`dvr_manager` (which instantiates `CameraWorker` again for previews), and
`hcnetsdk_camera.HCNetSDKCameraWorker` (partially implemented). This duplication is the root
cause of several defects listed below.

---

## 2. Critical bugs

### C1 — Reconnect livelock: cameras permanently stuck, never stream again

The stale-frame watchdog compares against `last_frame_time`, but the reconnect path **never
resets it**. After the first stall, every successful reconnect is immediately judged stale and
torn down before `cap.read()` is ever called.

- `backend/app/services/camera_service.py:654-676` — on successful open, `consecutive_failures`,
  `reconnect_delay` and `status` are reset; `w.state.last_frame_time` is not.

Evidence from `app-run.log.err` (same millisecond):

```
12:45:53.342 INFO  Camera 53 [Entrance]: Connected successfully
12:45:53.342 WARN  Camera 53: No frame for 15s – forcing reconnect
12:45:53.346 WARN  Camera 53 [Entrance]: Connecting (attempt 5) ...
```

All three cameras (53 Entrance, 54 Exit, 55 Dev-room) are in this loop for the entire tail of
the log. **Attendance capture is dead while this runs**, and each cycle opens a fresh RTSP
session against the DVR — Hikvision units cap concurrent sessions, so this can also lock out
other clients.

**Fix:** set `w.state.last_frame_time = time.time()` on successful open (and on entry to the
stale branch), and add a separate "connected but no first frame" timeout.

### C2 — FFmpeg options are appended to the RTSP URL, not passed to FFmpeg

`camera_service.py:489-508` builds
`rtsp://…/Streaming/Channels/201?rtsp_transport=tcp&fflags=nobuffer&…` and hands that to
`VideoCapture`. OpenCV does not parse query strings as FFmpeg options — the DVR receives them
as part of the request URI and ignores them. **TCP transport, `nobuffer` and `low_delay` are all
silently inactive**; streams run over whatever the DVR defaults to (usually UDP), which explains
packet-loss stalls and the reconnects in C1.

The correct mechanism is the `OPENCV_FFMPEG_CAPTURE_OPTIONS` environment variable
(`;`-separated `key;value` pairs), set before the capture is created.

Related: the module docstring advertises `CCTV_FFMPEG_OPTS` (`camera_service.py:32`) — that
variable is read nowhere in the codebase.

### C3 — Open/read timeouts have no effect

`camera_service.py:503-506` calls `cap.set(CAP_PROP_OPEN_TIMEOUT_MSEC, …)` *after* the
`VideoCapture(...)` constructor has already performed the connect. The open is synchronous, so
the timeout arrives too late to apply. An unreachable DVR blocks the stream thread on FFmpeg's
default timeout (tens of seconds).

Same defect in `api/routes/cameras.py:217-220` (`test-connection`) and
`api/routes/recognition.py` (`recognize-cctv-frame`) — the latter blocks a request worker.

### C4 — RTSP passwords written to logs in plaintext

`camera_service.py:480` logs the full source URL at INFO on every connect attempt. Combined with
C1, `app-run.log.err` (26 MB) contains
`rtsp://anilchanna:test%40123@192.168.29.181:554/…` thousands of times.

**Fix:** redact credentials before logging (`urlparse` → strip userinfo). Treat the existing log
file as compromised and rotate the DVR account password.

### C5 — HCNetSDK DVR preview endpoints will 500

`HCNetSDKCameraWorker` has no `get_latest_jpeg()` method (it exposes `get_latest_frame()` and
`state.latest_jpeg`), but `api/routes/cameras.py:765` and `:808` call
`camera.worker.get_latest_jpeg()` unconditionally → `AttributeError`.

`CameraManager` guards this with `hasattr` (`camera_service.py:1760`); the DVR routes do not.

### C6 — Night-shift / after-midnight exit is recorded as a check-**in**

Events are keyed to `now.date()` (`attendance_event_service.py:483`). An employee leaving at
00:30 hits the OUT camera on a new calendar day where their state is `ABSENT`; with
`attendance_checkin_on_missing_in=True` (the default), `resolve_camera_event`
(`attendance_event_service.py:171-205`) returns `CHECK_IN`.

**Leaving the building creates a check-in for the next day**, and the previous day is left open
forever.

**Fix:** shift-aware "business day" boundary (e.g. events before 05:00 belong to the prior day
when that day is still open).

### C7 — No end-of-day closeout: work hours inflate indefinitely

`attendance_event_service.py:276-287` adds an open interval from the last IN to *now* for today.
If the OUT camera misses someone (the common case — see A3), hours accrue until midnight, then
the day freezes with `sign_out_time = NULL` and inflated `total_work_hours`.

The only scheduled job in the app is the DSR reminder (`main.py:205`); there is no auto-closeout
task.

---

## 3. Video streaming

| Issue | Location | Impact |
|---|---|---|
| Sub-stream never used | `dvr_manager.py:282`, URLs end `…/{ch:03d}01` | Always main stream (1080p/4MP). Decode + detect cost 4–9× higher than needed. `…02` sub-stream at D1/720p is sufficient for doorway face detection |
| MJPEG delivery is a 30 Hz poll loop | `cameras.py:501-522` | One asyncio task per viewer waking 30×/s, taking the manager `RLock` each tick. Should be a `threading.Condition` / `asyncio.Event` signalled by the display thread |
| No viewer cap or per-camera fan-out limit | `cameras.py:488` | N tabs × M cameras = unbounded open streams |
| Stream not validated before it opens | `cameras.py:489-522` | Non-existent `camera_id` yields an empty stream forever instead of 404 |
| Media token expires but stream does not | `core/security.py:82` (120 s TTL) | Auth checked only at connect; a stream opened once runs indefinitely |
| No H.265 handling | throughout | Error text tells the user "use H.264 not H.265" (`cameras.py:243`) instead of supporting it. Modern Hikvision units default to H.265 |
| No recording / ring buffer | — | Nothing to review after a disputed event |

**Recommended**

1. Point attendance workers at the sub-stream; reserve the main stream for on-demand recording.
2. Drive MJPEG off a condition variable instead of polling.
3. For the operator dashboard, consider WebRTC (`go2rtc` / `mediamtx` sidecar) or
   fMP4-over-WebSocket. MJPEG at 25 FPS × 1080p is roughly 10× the bandwidth of H.264 for the
   same picture.

---

## 4. Face detection

- **SCRFD at 1024×1024 on CPU** (`face_service.py:25`) costs ~300–800 ms/frame. With
  `_ANALYSIS_INTERVAL = 0.12` (`camera_service.py:96`) the loop is detector-bound by ~5×, so the
  configured interval is fiction.
- **One global `_inference_lock`** (`face_service.py:27`) serialises *all* face detection and
  embedding across every camera. The comments at `camera_service.py:97-104` describing monitor
  cameras starving the IN/OUT cameras are describing this lock. The current mitigation (slower
  monitor analysis interval) is a workaround, not a fix.
- **The YOLO backend is worse, not better:** `_extract_faces_yolo`
  (`face_service.py:121-193`) takes the lock *twice per face* (predict + recogniser), serialising
  the whole system per detection.
- **Blur gate is full-frame Laplacian** (`camera_service.py:530`, threshold 80) — a sharp face
  against a blurred background is rejected. It should measure the *face crop*, as enrollment
  correctly does (`employee_face_service.py:78-87`).
- **Motion gate uses whole-frame mean absolute difference** (`camera_service.py:1078`); a person
  entering a 1080p frame changes ~2 % of pixels, below the threshold of 3.0. Camera shake or
  lighting flicker trips it constantly. Use MOG2 background subtraction with a contour-area gate,
  or restrict to a doorway ROI.

**Recommended**

- ONNX Runtime with tuned `intra_op_num_threads` and a **per-camera session** (drop the global lock).
- Detect at 640 on the sub-stream; escalate resolution only when a candidate face is small.
- Move to GPU — a single T4 / RTX 3050 turns this from ~1 fps/camera into 30+.

---

## 5. Face recognition

**The accuracy settings are currently unsafe for payroll-grade data.**

| Setting | Value | Location | Concern |
|---|---|---|---|
| `_MIN_THRESHOLD` | 0.35 | `camera_service.py:228` | ArcFace/buffalo_l's usual operating point is 0.45–0.55. 0.35 carries a materially non-zero false-accept rate |
| `dvr_recognition_threshold` | 0.35 | `core/config.py:325` | same |
| `_CONFIRM_FRAMES` | **1** | `camera_service.py:82` | A single frame writes attendance |
| `_IDENT_CONFIRM` | **1** | `camera_service.py:216` | A single read names a person |
| `_MIN_FACE_PX` | 16 | `camera_service.py:194` | A 16 px face carries no usable identity signal |

The in-code comments acknowledge the exact failure mode ("a man was once labelled *Saloni
Pathania* at 77 %") and then set every guard to 1 because the guards made recognition unusable.
That treats the symptom. The upstream problem is that a 16 px face plus a 3× bicubic upscale
(`_face_in_person_crop`, `camera_service.py:539-585`) cannot produce a trustworthy embedding at
any threshold.

Other recognition issues:

- **Linear scan over all employees per face** — `match.find_best_match` computes cosine against
  every enrolled embedding in Python. Invisible at 5 employees; dominant at 500 employees ×
  several photos × N faces × 8 ticks/s. Should be a single stacked `(N,512)` matrix multiply, or
  FAISS / pgvector.
- **Embedding cache is never invalidated across processes** — `embedding_cache.py` caches
  process-wide with manual `invalidate_embedding_cache()`. Multi-worker deployment ⇒ stale
  galleries.
- **`min_match_margin = 0.10`** (`core/config.py:111`) is sound in principle, but with only 5
  enrolled employees the runner-up is nearly always far away, so the margin gate passes
  trivially. It provides much less protection than assumed.
- **No liveness / anti-spoofing.** Documented honestly in `api/routes/recognition.py:57-63`.
  A printed photo or a phone screen held to the IN camera marks attendance. This is the single
  biggest integrity gap.
- **Body Re-ID is disabled** (`camera_service.py:151`) with well-documented reasoning — that call
  was correct, but ~200 lines of `_apply_reid` plus the `identity_manager` / `reid_service` layer
  remain live behind a flag that must not be turned on.

**Recommended, in order**

1. Add a passive anti-spoof model (MiniFASNet / Silent-Face, ~2 ms CPU) before `_mark_attendance`.
2. Raise `_MIN_FACE_PX` to ~50 and `_CONFIRM_FRAMES` to 3 **once camera placement delivers faces
   that size** — this is primarily a mounting problem (IN/OUT cameras at eye level, 1.5–3 m,
   facing the approach path).
3. Vectorise matching.
4. Persist `score` / `margin` on the attendance event so thresholds can be tuned from real data.

---

## 6. Tracking

- **`FaceTracker` uses centroid distance only** (`face_tracker.py:261-284`) with greedy per-track
  matching in dict-insertion order — not globally optimal (Hungarian /
  `scipy.optimize.linear_sum_assignment` would be). Two people crossing within
  `max_distance = 100 px` swap identities, and the identity is then held for
  `_IDENTITY_HOLD_SEC`, so the swap sticks.
- **`FaceTracker._lock = False`** (`face_tracker.py:178`) is a bool never set to `True` — a no-op
  masquerading as thread safety. Harmless today (single writer) but misleading.
- **No Kalman filter** — `predict_forward` (`face_tracker.py:72`) is exponentially-smoothed
  constant-velocity dead reckoning with 0.9 damping. Adequate for coasting, not for occlusion
  recovery.
- **`person_publish_held = True`** (`core/config.py:205`) publishes tracks YOLO no longer detects,
  so an operator sees boxes for people who have left. Deliberate and documented, but it means the
  on-screen person count is not trustworthy.
- **`_nested_ids` is O(n²) per frame** (`bytetrack_engine.py:336`) and is currently a no-op
  (`person_nested_contain = 1.01` > 1.0 can never be satisfied) — dead code worth deleting.
- **Line crossing is implemented but unreachable from the UI** — `crossing_enabled`,
  `line_orientation`, `line_position`, `entry_direction` exist on the model and worker but are
  absent from `CameraCreateRequest` / `CameraUpdateRequest` (`cameras.py:56-116`). Only a direct
  SQL edit can enable it.

---

## 7. Attendance accuracy

Beyond C6 (night shift) and C7 (no closeout):

### A1 — Cooldown state is in-memory and lost on restart

`_last_marked` (`camera_service.py:1408`) is per-worker. A restart, a config change
(`add_camera` replaces the worker), or a reconnect clears it. Only the DB-level 25 s directional
cooldown (`attendance_event_service.py:21`) survives.

### A2 — Attendance is marked before the write is known to succeed

`_mark` (`camera_service.py:885-892`) sets `pt.attendance_marked = True` and records the cooldown,
then fires the DB write onto a 2-thread executor whose result is discarded
(`_submit_attendance`, `camera_service.py:247-267`). If the write fails or the state machine
rejects it, the event is lost permanently — the track never retries and nothing surfaces to the
operator. Needs a retry queue with dead-lettering.

### A3 — The IN/OUT state machine strands people

`resolve_camera_event` (`attendance_event_service.py:171-205`): once someone is `AWAY`, every
further OUT is rejected as `duplicate_out_already_away`. Given that the OUT camera misses people
regularly (steep angles, back of head, tailgating), a single missed re-entry means their real
end-of-day departure is silently dropped and `sign_out_time` freezes at the last successful OUT.
Reconciliation should be time-aware, not purely state-machine driven.

### A4 — Fixed 09:00 / 18:00 workday

`_apply_late_and_early` (`attendance_event_service.py:344-362`) hardcodes both; only
`grace_time_minutes` is configurable. No shift table, no per-employee schedule, no weekend /
holiday awareness at this layer.

### A5 — Timezone mixing

`validate_event_time` compares an IST-derived date against `date.today()` (server-local) at
`attendance_event_service.py:435`. On a UTC-hosted server, events between 00:00–05:30 IST are
rejected as "future dates".

### A6 — No evidence trail

`AttendanceEvent` (`models/attendance.py:68-79`) stores only `camera_id` — no match score, no
margin, no track id, no face snapshot. A disputed record cannot be adjudicated.

> **Highest-value change in this section:** add `match_score`, `match_margin`, `track_id` and
> `snapshot_path` to `AttendanceEvent`, and write the 112×112 aligned face crop to disk at mark
> time. ~5 KB per event, makes disputes resolvable, and produces the labelled dataset needed to
> tune thresholds empirically instead of by anecdote.

---

## 8. Performance & scalability

Per-camera thread budget: 3 threads (stream / recognition / display) + a YOLO session + an
InsightFace session. On the 4-core box referenced in the config comments, **the system is already
saturated at 3 cameras** — the comments at `camera_service.py:1032-1040` document scheduled jobs
running 8 minutes late and the app appearing to hang.

Concrete limits:

- Global `_inference_lock` (face) + `BoundedSemaphore(1)` (YOLO, `bytetrack_engine.py:73`)
  ⇒ **effective concurrency is one inference at a time system-wide**.
- `_DISPLAY_FPS = 25` re-encodes a full-resolution JPEG 25×/s **per camera** at quality 80 — CPU
  spent on pixels nobody may be watching (mitigated by the 8 s idle gate, but only after 8 s).
- `frame.copy()` on every captured frame (`camera_service.py:718`) plus another in the overlay
  (`:337`) — ~6 MB/frame of allocation churn at 1080p × 25 fps.
- `CameraManager._workers` is process-local, so the design **cannot scale horizontally**.
  `main.py:349-352` acknowledges this and pins cameras to one instance.
- `_attendance_executor` has 2 workers and an unbounded queue — a DB stall silently grows memory.

**Path to 10–20 cameras**

1. Split the vision pipeline out of the FastAPI process into a dedicated worker service (one
   process per 2–4 cameras), publishing recognitions over Redis / NATS. The API process then
   holds no camera state and scales freely.
2. GPU inference — a single consumer card handles 15–20 cameras at 5 fps analysis.
3. Sub-streams for analysis; main stream only for on-demand recording.
4. Replace the global lock with per-process ONNX sessions and a batched detector (detect 4
   cameras' frames in one forward pass).

---

## 9. Error handling

- **Broad `except Exception` swallowing everything** — the recognition loop
  (`camera_service.py:1269-1272`) catches, logs and continues at full rate. A persistent error
  (missing model, corrupt frame) logs at ~8 Hz forever with no circuit breaker.
- **`_check_ffmpeg()` failure disables all cameras silently** (`camera_service.py:1609-1615`) —
  returns early with only a log line. No health flag, no admin notification.
- **No health / alerting surface.** `get_stats()` returns counters but nothing consumes them;
  nobody is paged when the Entrance camera has been reconnecting for 12 hours.
- **`create_camera` silently drops fields** — the request model accepts `frame_skip`,
  `tracking_max_distance`, `tracking_cooldown` but `cameras.py:299-308` never persists them. The
  API returns 200 and the values vanish.
- **`set_recognition_enabled` is a no-op** — `dvr_manager.py:397-400` sets
  `worker.recognition_enabled`, an attribute `CameraWorker` never reads. The dashboard toggle
  does nothing.
- **Schema drift:** `CameraConfig` has both `camera_type` and `camera_purpose`
  (`models/camera.py:29-30`), both NOT NULL, only the latter used. `dvr_manager` fabricates 8
  channels regardless of the real device (`dvr_manager.py:177`).

---

## 10. Logging

**The pipeline logs at INFO on the per-frame hot path.** Every `recognize_face` call emits
STEP-1, STEP-2, STEP-3 and STEP-11 lines (`recognition.py:371, 401, 269, 423`) — 4+ lines per
face per analysis tick, per camera.

- `logging.basicConfig` only (`main.py:39`) — **no rotation, no size cap, no retention**.
  `app-run.log.err` is at 26 MB and growing; left running it fills the disk and takes the app
  down.
- No date in the timestamp format (`datefmt="%H:%M:%S"`) — a multi-day log cannot be correlated.
- No structured `camera_id`; correlation requires regex across interleaved threads.
- Sensitive data (RTSP credentials) at INFO — see C4.

**Recommended**

- `RotatingFileHandler` (50 MB × 5) or `TimedRotatingFileHandler` immediately.
- Demote all `STEP-*` lines to DEBUG; keep INFO for state changes only (connect, disconnect,
  attendance marked, error).
- Emit structured JSON with `camera_id` / `track_id` / `employee_id` fields; add the date to
  `asctime`.
- Add Prometheus counters (`cctv_frames_total`, `cctv_reconnects_total`,
  `cctv_inference_seconds`, `cctv_attendance_marked_total`) — these replace most of what the log
  spam is currently used for.

---

## 11. User experience

- **Four overlapping CCTV pages** — `CctvAttendance.tsx`, `CctvCameraManager.tsx`,
  `DvrCameraDashboard.tsx`, `FaceDetection.tsx` — ~3,000 lines with duplicated camera-selection,
  preview and status logic, and no clear division of responsibility for an operator.
- **`CctvCameraManager` polls previews by cache-busting a JPEG URL**
  (`CctvCameraManager.tsx:229`) instead of using the MJPEG endpoint that already exists — one
  full HTTP round-trip per tile per interval.
- **Live dashboard refreshes every 30 s** (`LiveAttendanceDashboard.tsx:85`) — "live" attendance
  is up to half a minute stale. Natural place for SSE / WebSocket push from the recognition
  thread.
- **No camera health at a glance** — status is per-camera text, buried. An operator cannot tell
  that the Entrance camera has been down all morning.
- **Overlay is burned into the JPEG server-side** (`_draw_enhanced_overlay`,
  `camera_service.py:328`) — cannot be toggled or styled, is unreadable at small tile sizes
  (fixed 0.5-scale font, `max(180, …)` label width), and costs server CPU. Send tracks as JSON
  alongside the video and draw them in a canvas overlay.
- **No line-crossing configuration UI**, despite full backend support (§6).
- **No snapshot / event review** — an operator cannot see *why* someone was marked.

---

## 12. Suggested sequencing

### Week 1 — restore service (all small, all high impact)

| # | Change | Ref |
|---|---|---|
| 1 | Reset `last_frame_time` on reconnect → fixes the livelock | C1 |
| 2 | `OPENCV_FFMPEG_CAPTURE_OPTIONS` for real TCP / low-latency transport | C2 |
| 3 | Set timeouts before capture creation | C3 |
| 4 | Redact credentials in logs; rotate the DVR password | C4 |
| 5 | Add `RotatingFileHandler`; demote `STEP-*` to DEBUG | §10 |
| 6 | Fix `create_camera` field drop and the DVR `get_latest_jpeg` crash | §9, C5 |

### Weeks 2–4 — attendance integrity

| # | Change | Ref |
|---|---|---|
| 7 | Evidence columns + face snapshot on `AttendanceEvent` (**do this before tuning thresholds**) | A6 |
| 8 | Business-day boundary for night shifts | C6 |
| 9 | Nightly auto-closeout job with configurable cutoff | C7 |
| 10 | Retry queue + failure surfacing for attendance writes | A2 |
| 11 | Passive anti-spoof model before `_mark_attendance` | §5 |

### Month 2 — performance & scale

| # | Change | Ref |
|---|---|---|
| 12 | Sub-streams for analysis; MJPEG off a condition variable | §3 |
| 13 | Per-process ONNX sessions, drop the global inference lock; GPU past ~5 cameras | §4, §8 |
| 14 | Vectorised matching (or pgvector) for >100 employees | §5 |
| 15 | Extract the vision pipeline into its own service | §8 |

### Month 3 — UX & operations

| # | Change | Ref |
|---|---|---|
| 16 | Consolidate the four pages into one operator console with a health grid | §11 |
| 17 | Client-side canvas overlay + JSON tracks; SSE for live events | §11 |
| 18 | Line-crossing configuration UI | §6 |
| 19 | Prometheus metrics + alerting on camera down / no attendance in N hours | §9, §10 |

---

## 13. Closing observation

The code comments in `camera_service.py` and `core/config.py` are unusually thorough and honest —
they record measured numbers and failed experiments. But several of them describe tuning a
threshold *down* to compensate for a camera mounted too high and too far away (16 px faces, 0.11
detection scores, dedicated "steep camera" special cases). No threshold can recover identity
information the optics never captured.

**Repositioning the IN/OUT cameras to eye level on the approach path would likely improve
recognition accuracy more than every software change on this list combined** — and would allow
the safety guards (`_CONFIRM_FRAMES`, `_IDENT_CONFIRM`, `_MIN_FACE_PX`, `_MIN_THRESHOLD`) to be
restored to sane values.
