"""Camera stream URLs must never carry their password into a log or an API response.

The connect path logged the full source URL at INFO on every attempt. Combined
with the reconnect livelock that wrote the DVR password to disk thousands of
times, into a 26 MB file.

Two independent layers are tested here, because one is not enough:
  1. `_redact_url` at the call sites
  2. `RedactCredentialsFilter` on every log handler, for the f-string someone
     forgets to wrap

Note the two URL shapes. They are NOT both standard:
    rtsp://user:pass@host:554/path       creds LEFT of '@'
    hcnetsdk://ip:port@user:pass?ch=1    creds RIGHT of '@'
urlparse chokes on the second (it reads `pass` as a port and raises), which is
why redaction is hand-rolled — and why that case is pinned below.
"""
import logging

import pytest
from app.main import RedactCredentialsFilter
from app.services.camera_service import _redact_url

SECRETS = ("test%40123", "pw123", "S3cret", "hunter2")


@pytest.mark.parametrize(
    "url,expected",
    [
        # Standard RTSP — the real production shape.
        (
            "rtsp://anilchanna:test%40123@192.168.29.181:554/Streaming/Channels/201",
            "rtsp://anilchanna:***@192.168.29.181:554/Streaming/Channels/201",
        ),
        # HCNetSDK — inverted layout. The ?channel= suffix must SURVIVE: which
        # channel failed is the whole point of the error logs calling this.
        (
            "hcnetsdk://192.168.1.100:8000@admin:pw123?channel=1",
            "hcnetsdk://192.168.1.100:8000@admin:***?channel=1",
        ),
        # Nothing to redact — must be returned untouched, not mangled.
        ("rtsp://192.168.1.5:554/stream", "rtsp://192.168.1.5:554/stream"),
        ("http://user@host/x", "http://user@host/x"),
        ("0", "0"),
        ("", ""),
    ],
)
def test_redact_url(url, expected):
    assert _redact_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "rtsp://anilchanna:test%40123@192.168.29.181:554/Streaming/Channels/201",
        "hcnetsdk://192.168.1.100:8000@admin:pw123?channel=1",
        "rtsp://u:hunter2@host/path",
    ],
)
def test_no_secret_survives_redaction(url):
    redacted = _redact_url(url)
    for secret in SECRETS:
        assert secret not in redacted, f"{secret!r} leaked through _redact_url"


@pytest.mark.parametrize(
    "url",
    [
        "garbage@@@",
        "://@:",
        "rtsp://",
        "@",
        "not a url at all",
        "hcnetsdk://host-only",
    ],
)
def test_redaction_never_raises(url):
    """A log call must not become an exception.

    _redact_url is called from inside error handlers. urlparse raises on the
    hcnetsdk shape, so an implementation that used it would mask the very
    failure being reported.
    """
    assert isinstance(_redact_url(url), str)


def test_logging_filter_scrubs_a_missed_fstring():
    record = logging.LogRecord(
        "t", logging.INFO, "x.py", 1,
        "Opening rtsp://admin:S3cret@10.0.0.5:554/x", None, None,
    )
    assert RedactCredentialsFilter().filter(record) is True
    message = record.getMessage()
    assert "S3cret" not in message
    assert "rtsp://admin:***@10.0.0.5:554/x" in message


def test_logging_filter_leaves_ordinary_messages_alone():
    record = logging.LogRecord(
        "t", logging.INFO, "x.py", 1, "Camera %s connected", ("53",), None,
    )
    assert RedactCredentialsFilter().filter(record) is True
    assert record.getMessage() == "Camera 53 connected"
