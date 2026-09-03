"""The doorway must get the inference resource, and keep its evidence.

Every test here pins a behaviour that production MEASUREMENT showed was missing.
The numbers in the docstrings are from the live system (hrms.log PERF lines,
25 Aug - 1 Sep 2026, host-suspend outliers excluded) and from a component
benchmark run on the same 4-physical-core box.
"""
import threading
import time

import pytest
from app.services import bytetrack_engine
from app.services.inference_gate import PriorityInferenceGate

# ── Scenario F: the doorway gets priority on the expensive stage ────────────
#
# The face pipeline already had a priority gate. YOLO -- which is 95% of the
# doorway cycle (2.40s inside YOLO + 2.10s queueing, of a 4.75s median) -- used
# a bare BoundedSemaphore with no ordering guarantee at all.

def test_doorway_is_admitted_before_a_room_camera_that_asked_first():
    """A room camera already queueing must not beat a doorway to a free slot."""
    gate = PriorityInferenceGate(slots=1, low_priority_max_wait=30.0)
    order = []
    holding = threading.Event()
    release = threading.Event()

    def occupy():
        with gate.acquire(high_priority=False):
            holding.set()
            release.wait(5)

    def low():
        with gate.acquire(high_priority=False):
            order.append("room")

    def high():
        with gate.acquire(high_priority=True):
            order.append("doorway")

    blocker = threading.Thread(target=occupy); blocker.start()
    assert holding.wait(5), "blocker never acquired the slot"

    t_low = threading.Thread(target=low); t_low.start()
    time.sleep(0.15)                      # the room camera is queueing FIRST
    t_high = threading.Thread(target=high); t_high.start()
    time.sleep(0.15)

    release.set()
    for t in (blocker, t_low, t_high):
        t.join(5)

    assert order == ["doorway", "room"], (
        f"expected the doorway to jump the queue, got {order}"
    )


def test_a_room_camera_is_never_starved_completely():
    """Priority must not become a monopoly — the room feed still has to update."""
    gate = PriorityInferenceGate(slots=1, low_priority_max_wait=0.2)
    served = []

    def low():
        with gate.acquire(high_priority=False):
            served.append("room")

    t = threading.Thread(target=low)
    t.start()

    stop = time.time() + 3.0
    while time.time() < stop and not served:
        with gate.acquire(high_priority=True):
            time.sleep(0.02)
    t.join(3)

    assert served == ["room"], "a continuous doorway load froze the room camera"


def test_the_yolo_gate_is_priority_aware_at_all():
    """Regression: this was a plain semaphore, so priority simply did not exist."""
    gate = bytetrack_engine._gate()
    assert isinstance(gate, PriorityInferenceGate), (
        "the YOLO gate was a bare threading.BoundedSemaphore, which has no "
        "priority or fairness guarantee — and YOLO is 95% of the doorway cycle"
    )
    # The capability, not the prose describing it.
    import inspect

    assert "high_priority" in inspect.signature(gate.acquire).parameters


# ── Provisional-track grace ────────────────────────────────────────────────
#
# Evidence lives on the track. Deleting a doorway track on its first missed pass
# reset the observation count every pass, which made the 3-observation
# attendance requirement unreachable by construction: 2,783 provisional tracks
# produced 6 allowed decisions in the production log.

def test_a_monitor_camera_still_retires_on_the_first_miss():
    """grace_floor 0 is the historical behaviour and stays the default."""
    assert bytetrack_engine.should_retire_provisional(
        missed_for=0.01, grace_floor=0.0, cycle_sec=4.0
    )


def test_a_doorway_track_survives_one_missed_pass():
    """At a 4s pass rate, a person missed once must keep their evidence."""
    assert not bytetrack_engine.should_retire_provisional(
        missed_for=4.0, grace_floor=2.0, cycle_sec=4.0
    )


def test_the_grace_scales_with_the_measured_pass_rate():
    """The floor is a floor: a slower camera needs a longer window, not a fixed one."""
    fast = bytetrack_engine.effective_provisional_grace(grace_floor=2.0, cycle_sec=0.4)
    slow = bytetrack_engine.effective_provisional_grace(grace_floor=2.0, cycle_sec=3.0)
    assert fast == 2.0, "never shorter than the floor"
    assert slow == pytest.approx(4.5), "1.5x the interval actually being achieved"


def test_the_grace_is_capped_however_slow_the_camera_gets():
    """A stalled camera must not hold a phantom person on the overlay."""
    grace = bytetrack_engine.effective_provisional_grace(
        grace_floor=2.0, cycle_sec=600.0
    )
    assert grace == bytetrack_engine._PROVISIONAL_GRACE_MAX_SEC


def test_a_genuinely_departed_person_is_still_retired():
    assert bytetrack_engine.should_retire_provisional(
        missed_for=30.0, grace_floor=2.0, cycle_sec=4.0
    )


# ── Host suspend must not be reported as pipeline latency ──────────────────
#
# `PERF camera=60 infer=2816ms wait=3357839ms cycle=3360.65s` read as a
# 56-minute stall on camera 60. Every thread in the process was silent for
# exactly that window: the machine slept.

def test_a_host_suspend_is_excluded_from_latency_statistics(caplog):
    cam = "test-suspend-cam"
    with bytetrack_engine._perf_lock:
        bytetrack_engine._perf.pop(cam, None)

    bytetrack_engine._record_perf(cam, wait_ms=120.0, infer_ms=2000.0)
    with caplog.at_level("WARNING"):
        stats = bytetrack_engine._record_perf(
            cam, wait_ms=3_357_839.0, infer_ms=2816.0
        )

    assert stats["wait_ms"] == 0.0, "a suspend must not be reported as queue time"
    assert stats["suspensions"] == 1
    assert "host suspend" in caplog.text.lower()

    with bytetrack_engine._perf_lock:
        bytetrack_engine._perf.pop(cam, None)


def test_a_real_queue_wait_is_still_recorded():
    """The guard must not swallow genuine contention — that is the signal."""
    cam = "test-realwait-cam"
    with bytetrack_engine._perf_lock:
        bytetrack_engine._perf.pop(cam, None)
    stats = bytetrack_engine._record_perf(cam, wait_ms=2100.0, infer_ms=2400.0)
    assert stats["wait_ms"] == 2100.0
    assert stats.get("suspensions", 0) == 0
    with bytetrack_engine._perf_lock:
        bytetrack_engine._perf.pop(cam, None)


# ── Deferred embedding ─────────────────────────────────────────────────────
#
# BENCHMARKED on the production box: AdaFace ir_50 costs 660ms PER FACE on CPU,
# against 823ms to detect every face in the whole frame. It used to run on every
# detected face BEFORE the quality gate — so a 16px downward profile that was
# always going to be rejected still cost 660ms first.

def test_quality_can_be_judged_without_an_embedding():
    """The safety property the whole optimisation rests on.

    Deferring is only sound because the quality gate never consults the
    embedding. If that ever stops being true, this test fails and the deferral
    must be reconsidered — not the other way round.
    """
    from app.services import face_quality

    face = {
        "box": [100.0, 100.0, 140.0, 150.0],
        "confidence": 0.82,
        "kps": [[110, 118], [130, 118], [120, 130], [112, 140], [128, 140]],
        "pose": {"yaw": 4.0, "pitch": -6.0, "roll": 2.0},
        "embedding": None,          # deferred — not computed yet
    }
    graded = face_quality.assess(face, None, limits=face_quality.ENROLLMENT_LIMITS)
    assert graded is not None
    assert isinstance(graded.ok, bool)


def test_ensure_embedding_pays_once_and_caches():
    from app.services import face_service

    calls = []

    def compute():
        calls.append(1)
        return [0.1, 0.2, 0.3]

    face = {"embedding": None, "_embed": compute}
    assert face_service.ensure_embedding(face) is True
    assert face["embedding"] == [0.1, 0.2, 0.3]
    assert face_service.ensure_embedding(face) is True      # idempotent
    assert len(calls) == 1, "the expensive call must happen at most once"


def test_a_rejected_face_never_pays_for_an_embedding():
    """The point of the change: no compute() call for a face nobody matches."""
    from app.services import face_service

    calls = []
    face = {"embedding": None, "_embed": lambda: calls.append(1)}

    # Caller decides the face failed quality and simply never asks for it.
    assert calls == []
    assert face.get("embedding") is None
    # Sanity: it WOULD have cost something had we asked.
    face_service.ensure_embedding(face)
    assert len(calls) == 1


def test_an_eager_face_is_left_alone():
    from app.services import face_service

    face = {"embedding": [1.0, 0.0]}
    assert face_service.ensure_embedding(face) is True
    assert face["embedding"] == [1.0, 0.0]


def test_a_failing_embedding_is_reported_not_raised():
    """A torch error must degrade to 'no usable face', never kill the thread."""
    from app.services import face_service

    def boom():
        raise RuntimeError("model exploded")

    face = {"embedding": None, "_embed": boom}
    assert face_service.ensure_embedding(face) is False
    assert face["embedding"] is None


def test_a_face_with_no_embedding_and_no_deferral_is_rejected():
    from app.services import face_service

    assert face_service.ensure_embedding({"embedding": None}) is False


# ── Motion gate ────────────────────────────────────────────────────────────
#
# The gate decides whether ANY detection runs, so a person it suppresses is
# invisible to every stage after it. MEASURED on the 253 real person crops these
# cameras captured, replayed as a person moving half a body width:
#
#                     people suppressed    fires on a static scene?
#   mean >= 3.0            48%             yes (+/-8 noise -> 4.24, +6 brightness -> 6.00)
#   %changed >= 0.15%       6%             no  (0.000% on every static case)
#
# Camera 57's MEDIAN person is 243px tall, and the old rule suppressed 100% of
# everyone under 300px.

def _motion_pct(prev, cur, delta=25.0):
    """The production metric, reimplemented here so the test pins the RULE."""
    import cv2
    import numpy as np

    d = cv2.absdiff(cur, prev)
    return float(np.count_nonzero(d > delta)) / float(d.size) * 100.0


def _blank(value=78):
    import numpy as np

    return np.full((1080, 1920), value, dtype=np.uint8)


def test_a_distant_person_is_not_suppressed():
    """A 240px person — camera 57's median — must wake the detector."""
    import numpy as np
    from app.services import camera_service

    prev, cur = _blank(), _blank()
    person = np.full((240, 120), 190, dtype=np.uint8)
    prev[400:640, 300:420] = person
    cur[400:640, 360:480] = person          # moved half a body width

    pct = _motion_pct(prev, cur)
    assert pct >= camera_service._MOTION_CHANGED_PCT, (
        f"a 240px person scored {pct:.3f}% against a "
        f"{camera_service._MOTION_CHANGED_PCT}% threshold — this is the case the "
        "frame-wide mean suppressed 100% of the time"
    )


def test_a_global_brightness_change_is_not_motion():
    """Auto-exposure stepping must not burn the single inference slot.

    The old mean-based rule scored 6.00 on a +6 brightness shift, well over its
    own 3.0 threshold, so a lighting change looked exactly like a person.
    """
    import numpy as np
    from app.services import camera_service

    prev = _blank(78)
    cur = _blank(84)                        # every pixel +6, nothing moved
    pct = _motion_pct(prev, cur)
    assert pct < camera_service._MOTION_CHANGED_PCT, (
        f"a uniform brightness step scored {pct:.3f}% and would trigger detection"
    )


def test_sensor_noise_is_not_motion():
    import numpy as np
    from app.services import camera_service

    rng = np.random.default_rng(5)
    prev = _blank()
    cur = np.clip(prev.astype(int) + rng.integers(-8, 9, prev.shape), 0, 255).astype("uint8")
    pct = _motion_pct(prev, cur)
    assert pct < camera_service._MOTION_CHANGED_PCT, (
        f"+/-8 sensor noise scored {pct:.3f}% — the old mean rule scored 4.24 "
        "against its 3.0 threshold and treated this as a person"
    )


def test_an_empty_scene_is_idle():
    from app.services import camera_service

    assert _motion_pct(_blank(), _blank()) < camera_service._MOTION_CHANGED_PCT


# ── Face detection must follow the bodies ──────────────────────────────────
#
# In the body-tracking pipeline a detected face reaches recognition ONLY by
# being assigned to a person track. With no tracks the assignment is empty and
# every face found is discarded. MEASURED 2026-09-02: SCRFD costs 859-961ms a
# pass, and 73% of camera 57's passes had no person track at all.

def test_face_detection_is_skipped_when_nobody_is_tracked():
    """The ordering that makes the skip safe, pinned against the source."""
    import inspect
    from app.services import camera_service

    src = inspect.getsource(camera_service._RecognitionThread._analyze_person)
    # The bodies must be resolved BEFORE the face stage decides whether to run.
    bodies = src.index("_t[\"person_detect_track_ms\"]")
    guard = src.index("if skip_faces or not ptracks:")
    assign = src.index("_assign_faces_to_tracks(faces, ptracks)")
    assert bodies < guard < assign, (
        "the face stage must be gated on ptracks, and ptracks must already be "
        "known at that point"
    )


def test_faces_are_only_consumed_through_the_track_assignment():
    """If a future change reads `faces` directly, the skip above becomes unsafe.

    This is the assumption the optimisation rests on: nothing looks at the face
    list except the code that maps faces onto person tracks (plus logging).
    """
    import ast
    import inspect
    import textwrap
    from app.services import camera_service

    src = inspect.getsource(camera_service._RecognitionThread._analyze_person)
    # Parse rather than grep: the docstring and the comments both say "faces"
    # in prose, and a text scan cannot tell that from a read of the variable.
    tree = ast.parse(textwrap.dedent(src))
    lines = textwrap.dedent(src).splitlines()
    uses = sorted({
        lines[node.lineno - 1].strip()
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "faces"
    })
    assert uses, "expected to find real reads of `faces`"
    for ln in uses:
        assert (
            "len(faces)" in ln                      # logging only
            or "_assign_faces_to_tracks" in ln      # the one real consumer
            or "faces = " in ln                     # assignment
            or "bool(faces)" in ln                  # heartbeat gate
            or "skip_faces" in ln
        ), f"`faces` used in a way the skip does not account for: {ln}"


# ── Routes must be pinned by their EFFECTIVE url ───────────────────────────
#
# The camera-enrolment route was declared "/employees/{id}/enrol-from-camera"
# on a router already mounted at /employees, so it landed on
# /api/employees/employees/{id}/... and the browser got 405 Method Not Allowed.
#
# A smoke test that asked `any("enrol-from-camera" in path)` passed, because the
# substring was there. Only the fully-mounted path catches it.

def test_camera_enrolment_route_is_mounted_where_the_ui_calls_it():
    from app.main import app

    urls = {
        path: sorted(getattr(r, "methods", []) or [])
        for r in app.routes
        if "enrol-from-camera" in (path := getattr(r, "path", ""))
    }
    assert urls == {"/api/employees/{employee_id}/enrol-from-camera": ["POST"]}, (
        f"route is not where api/client.ts calls it: {urls}"
    )


def test_no_route_has_a_doubled_prefix():
    """A whole class of the same mistake, across every router."""
    from app.main import app

    doubled = [
        p for r in app.routes
        if (p := getattr(r, "path", "")) and "/employees/employees/" in p
    ]
    assert not doubled, f"routes with a doubled prefix: {doubled}"


# ── A gallery must hold ONE person ─────────────────────────────────────────
#
# On 2026-09-02 clusters were filed under the wrong person from 54x88px crops.
# One gallery ended up holding three people; its coherence fell to 0.248 against
# ~0.50 for a clean one, and it then matched 133 of 405 unassigned sightings
# (32.8%) against 1-20 for everyone else. The symptom reported was "it detects
# the wrong person as Seema".

def test_a_mismatched_cluster_is_refused_not_silently_enrolled():
    """The assign endpoint must reject an assignment that contradicts the gallery."""
    import inspect
    from app.api.routes import face_review

    src = inspect.getsource(face_review.assign_unknown_cluster)
    assert "cluster_agreement" in src, "assignment is not checked at all"
    assert "mismatch" in src and "409" in src, (
        "a cluster that does not look like the employee must be refused, not "
        "enrolled — a wrong enrolment is silent and permanent"
    )
    assert "payload.force" in src, "there must be a deliberate override, not none"


def test_agreement_thresholds_sit_between_strangers_and_real_matches():
    """MEASURED on the live queue: real employee clusters agree 0.36-0.47 with
    their own gallery; strangers score below 0.13. The gate has to fall in that
    gap or it either blocks every enrolment or catches nothing."""
    from app.services import unknown_faces

    assert 0.13 < unknown_faces._ASSIGN_AGREE_WARN < 0.36
    assert unknown_faces._ASSIGN_AGREE_WARN < unknown_faces._ASSIGN_AGREE_OK <= 0.47


def test_agreement_is_undecidable_for_a_first_enrolment():
    """With no gallery there is nothing to contradict, so it must not refuse."""
    import inspect
    from app.services import unknown_faces

    src = inspect.getsource(unknown_faces.cluster_agreement)
    assert "no_gallery" in src, (
        "a first enrolment has nothing to compare against and must not be blocked"
    )


def test_every_quality_rejection_tells_the_person_what_to_do():
    """They are standing at the camera and can fix pose or distance instantly —
    but only if told which. A generic "face is not clear" sends them away
    guessing, which is what happened at the exit gate on 2026-09-02."""
    import inspect
    import re
    from app.services import face_quality
    from app.services.employee_face_service import capture_enrolment_from_camera

    codes = set(re.findall(r'_fail\(\s*"([a-z_]+)"', inspect.getsource(face_quality)))
    codes.add("no_face")
    mapping = inspect.getsource(capture_enrolment_from_camera)
    missing = sorted(c for c in codes if f'"{c}"' not in mapping)
    assert not missing, f"no instruction for quality reasons: {missing}"


# ── A gate must not starve the rooms polling an empty corridor ─────────────
#
# The 2s forced poll exists for one case: somebody standing motionless at the
# gate whose track expired. That is only possible if a person was recently
# there. MEASURED: 54-58% of gate passes had nobody in them, and two gates
# polling that hard pushed the room cameras' cycle to 15.9-28.7s.

def test_a_gate_backs_off_once_nobody_has_been_seen():
    from app.services import camera_service

    assert camera_service._MAX_IDLE_SKIP_QUIET_SEC > camera_service._MAX_IDLE_SKIP_SEC, (
        "an empty corridor must be polled less often than an occupied one"
    )


def test_a_gate_stays_fast_while_someone_is_around():
    """The backoff must not undo the stationary-person fix."""
    import inspect
    from app.services import camera_service

    src = inspect.getsource(camera_service._RecognitionThread._analyze_person)
    assert "_last_person_ts" in src, "nothing records when a person was last seen"
    run = inspect.getsource(camera_service._RecognitionThread.run)
    assert "_GATE_QUIET_AFTER_SEC" in run and "_MAX_IDLE_SKIP_SEC" in run, (
        "the fast poll must still apply while a person was recently present"
    )


def test_motion_still_wakes_a_gate_immediately():
    """The backoff governs the FORCED pass only — real motion must not wait."""
    import inspect
    from app.services import camera_service

    run = inspect.getsource(camera_service._RecognitionThread.run)
    forced = run.index("_idle_for >= _idle_limit")
    # The forced-pass branch is guarded on `static`; motion bypasses it entirely.
    assert "if static and not w.is_monitor" in run[forced - 400:forced + 80]


# ── AdaFace ONNX must produce the SAME vectors ─────────────────────────────
#
# Every enrolled face in the database is an AdaFace vector and a match is a
# cosine between a live vector and those. If the ONNX build drifted, every
# enrolment would degrade silently and it would look like "the cameras got
# worse". MEASURED on 60 real crops: cosine min 0.99999988, max abs diff
# 7.75e-07 — six orders of magnitude below the 0.18 margin bar.

def test_the_model_version_did_not_change_with_the_onnx_switch():
    """The gallery stays valid ONLY because the vectors are identical.

    If someone swaps in a differently-trained model they must also change this
    string, which invalidates the gallery loudly instead of degrading quietly.
    """
    from app.services.face_service import EMBEDDING_MODEL_VERSION

    assert EMBEDDING_MODEL_VERSION == "adaface_ir50_webface4m_v1"


def test_adaface_falls_back_to_pytorch_without_the_onnx_build():
    """A missing export must slow recognition down, never stop it."""
    from app.services import adaface_service

    real = adaface_service._ONNX_PATH
    try:
        adaface_service._ONNX_PATH = type(real)("no-such-file.onnx")
        adaface_service._onnx_session.cache_clear()
        assert adaface_service._onnx_session() is None
    finally:
        adaface_service._ONNX_PATH = real
        adaface_service._onnx_session.cache_clear()


def test_preprocessing_is_contiguous():
    """AdaFace uses .view() internally and raises on a transposed view — the
    exact error the export hit before this was fixed."""
    import numpy as np
    from app.services import adaface_service

    img = np.zeros((200, 260, 3), dtype=np.uint8)
    kps = np.array([[70, 95], [125, 95], [98, 130], [74, 175], [122, 175]],
                   dtype=np.float32)
    batch = adaface_service._preprocess(img, kps)
    assert batch.flags["C_CONTIGUOUS"]
    assert batch.shape == (1, 3, 112, 112)
    assert batch.dtype == np.float32
