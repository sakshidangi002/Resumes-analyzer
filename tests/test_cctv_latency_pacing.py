"""Capture pacing and stream selection — the two levers on CCTV latency.

Background. The capture loop used cv2.VideoCapture.read() (grab + decode +
colour-convert) on EVERY frame, then stored only the newest one. Across twelve
cameras on a four-core host the decode threads lost CPU to SCRFD (~500 ms/call)
and YOLO (~1000 ms/call), so the reader fell behind the source, the FFmpeg
queue grew, and the picture drifted seconds behind reality before stalling.

Two changes, both tested here:

  * `retrieve_period()` — drain the stream at full rate with grab(), but only
    decode a frame when the display or recognition thread will actually use it.
  * `build_channel_rtsp_url()` — pull the SUB stream for DVR previews rather
    than the hardcoded main stream.

No camera, no DVR, no OpenCV capture is created.
"""
import pytest
from app.services.dvr_manager import build_channel_rtsp_url


# ---------------------------------------------------------------------------
# Stream profile selection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "profile,expected_suffix",
    [
        ("sub", "00102"),      # explicit opt-in to the low-res stream
        ("SUB", "00102"),      # case-insensitive
        ("main", "00101"),     # full resolution
        ("", "00101"),         # anything unrecognised falls back to MAIN
        ("nonsense", "00101"),
    ],
)
def test_stream_profile_selects_the_right_channel_suffix(profile, expected_suffix):
    url = build_channel_rtsp_url("10.0.0.5", "admin", "pw", 1, profile=profile)
    assert url.endswith(expected_suffix)


def test_main_stream_is_the_default():
    """Previews exist to be LOOKED AT.

    Sub-stream halves the decode cost, but Hikvision sub-streams are commonly
    left at CIF (352x288) — upscaled into a dashboard tile that is an unusable
    blur. An unrecognised or missing profile must therefore fall back to full
    resolution, never silently degrade the picture.
    """
    assert build_channel_rtsp_url("10.0.0.5", "admin", "pw", 1).endswith("00101")


@pytest.mark.parametrize("channel,expected", [(1, "00101"), (4, "00401"), (12, "01201")])
def test_channel_number_is_zero_padded_to_three_digits(channel, expected):
    assert build_channel_rtsp_url("10.0.0.5", "u", "p", channel).endswith(expected)


def test_credentials_are_url_encoded():
    """DVR passwords routinely contain '@', '#' and ':'.

    Left raw, the '@' terminates the userinfo early and the URL points at a
    nonsense host — the connection fails with a misleading network error.
    """
    url = build_channel_rtsp_url("10.0.0.5", "user@corp", "p@ss:w#rd", 1)
    assert "p%40ss%3Aw%23rd" in url
    assert "user%40corp" in url
    # Exactly one '@' — the real userinfo separator.
    assert url.count("@") == 1


# ---------------------------------------------------------------------------
# Decode pacing
# ---------------------------------------------------------------------------
class _FakeWorker:
    """The three fields retrieve_period() reads, without building a worker."""

    def __init__(self, analysis_interval: float, last_view_ts: float, paused: bool = False):
        self.analysis_interval = analysis_interval
        self._last_view_ts = last_view_ts
        self.analysis_paused = paused

    retrieve_period = None  # bound below


@pytest.fixture(scope="module")
def retrieve_period():
    cs = pytest.importorskip(
        "app.services.camera_service", reason="needs the vision stack (cv2)"
    )
    _FakeWorker.retrieve_period = cs.CameraWorker.retrieve_period
    return cs


def test_watched_camera_decodes_at_display_rate(retrieve_period):
    """Somebody is looking: decode fast enough to feed _DISPLAY_FPS."""
    import time

    worker = _FakeWorker(analysis_interval=1.5, last_view_ts=time.time())
    period = _FakeWorker.retrieve_period(worker)
    assert period == pytest.approx(1.0 / retrieve_period._DISPLAY_FPS, rel=1e-6)


def test_unwatched_monitor_camera_decodes_at_analysis_rate(retrieve_period):
    """Nobody watching a 1.5s-interval monitor camera: ~0.7 fps, not 12.

    This is the saving. Twelve cameras decoding at stream rate to consume ~1 fps
    is what starved the inference threads.
    """
    worker = _FakeWorker(analysis_interval=1.5, last_view_ts=0.0)
    assert _FakeWorker.retrieve_period(worker) == pytest.approx(1.5, rel=1e-6)


def test_unwatched_attendance_camera_still_decodes_fast(retrieve_period):
    """An IN/OUT camera must NOT be throttled just because nobody is watching.

    Attendance depends on sampling people who walk past in ~2 seconds, so the
    analysis rate governs regardless of viewers.
    """
    worker = _FakeWorker(analysis_interval=0.12, last_view_ts=0.0)
    assert _FakeWorker.retrieve_period(worker) == pytest.approx(0.12, rel=1e-6)


def test_paused_and_unwatched_camera_falls_back_to_a_slow_tick(retrieve_period):
    """Analysis paused AND unwatched: keep a trickle so an arriving viewer sees
    a current picture rather than a frozen one."""
    worker = _FakeWorker(analysis_interval=1.5, last_view_ts=0.0, paused=True)
    assert _FakeWorker.retrieve_period(worker) == 1.0


def test_period_is_never_zero_or_negative(retrieve_period):
    """A zero period would spin the capture loop at 100% CPU."""
    import time

    for interval in (0.0, -1.0, 0.001, 60.0):
        for viewed in (0.0, time.time()):
            period = _FakeWorker.retrieve_period(_FakeWorker(interval, viewed))
            assert period > 0
