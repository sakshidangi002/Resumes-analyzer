"""The DVR dashboard must not duplicate a configured camera's worker.

Channel 1 was being pulled twice: once by the configured DB camera (Dev-room,
/Streaming/Channels/101) and again by the dashboard preview
(/Streaming/Channels/00101). Two RTSP sessions, two YOLO models, two face
pipelines — and two INDEPENDENT identity states, so the face observations that
should have combined into one confident match were split between them.

Measured on the live log: 22 identifications on the configured worker, 2 on its
duplicate. The operator was watching the weaker half.

The channel must be compared NUMERICALLY: Hikvision accepts several spellings of
the same stream and this codebase uses two of them.
"""
import pytest

dvr = pytest.importorskip("app.services.dvr_manager", reason="needs the vision stack")
parse_channel = dvr.parse_channel_from_url


@pytest.mark.parametrize(
    "url,expected",
    [
        # The two spellings actually in use — both mean channel 1, main stream.
        ("rtsp://h/Streaming/Channels/101", 1),
        ("rtsp://h/Streaming/Channels/00101", 1),
        ("rtsp://h/Streaming/Channels/401", 4),
        ("rtsp://h/Streaming/Channels/00401", 4),
        ("rtsp://h/Streaming/Channels/102", 1),      # sub stream, same channel
        ("rtsp://h/Streaming/Channels/1201", 12),
        ("rtsp://h/Streaming/Channels/101/", 1),     # trailing slash
        ("rtsp://h/Streaming/Channels/101?x=1", 1),  # query string
    ],
)
def test_channel_parsed_numerically(url, expected):
    assert parse_channel(url) == expected


def test_the_two_spellings_of_one_channel_agree():
    """String comparison misses this, which is why the duplicate existed."""
    a = "rtsp://h/Streaming/Channels/101"
    b = "rtsp://h/Streaming/Channels/00101"
    assert a != b
    assert parse_channel(a) == parse_channel(b) == 1


@pytest.mark.parametrize(
    "url", ["", None, "rtsp://h/live", "rtsp://h/Streaming/Channels/", "0", "rtsp://h/ch/ab"]
)
def test_unparseable_urls_return_none(url):
    """None must mean 'unknown', never 0 — a false match would make the
    dashboard hand out the wrong camera's worker."""
    assert parse_channel(url) is None


def test_usb_index_is_not_mistaken_for_a_channel():
    assert parse_channel("0") is None
    assert parse_channel("1") is None


def test_shared_worker_flag_exists_on_the_camera_record():
    """Without it, stopping a preview would stop the real camera."""
    cam = dvr.LiveCamera(channel_id=1, name="x", status="online")
    assert cam.shared_worker is False


def test_stop_releases_a_shared_worker_without_stopping_it():
    """Closing a preview tab must never take down a configured camera."""
    class _Worker:
        def __init__(self):
            self.stopped = False
            self._last_view_ts = 0.0

        def stop(self):
            self.stopped = True

        def is_alive(self):
            return True

    worker = _Worker()
    manager = dvr.DVRManager.__new__(dvr.DVRManager)
    camera = dvr.LiveCamera(channel_id=1, name="x", status="online")
    camera.rtsp_worker = worker
    camera.shared_worker = True

    import threading
    conn = dvr.DVRConnection.__new__(dvr.DVRConnection)
    conn.cameras = {1: camera}
    conn.lock = threading.RLock()
    manager._connection = conn

    assert manager.stop_camera_stream(1) is True
    assert worker.stopped is False, "the configured camera's worker was stopped"
    assert camera.rtsp_worker is None, "the dashboard kept a dangling reference"
    assert camera.shared_worker is False
