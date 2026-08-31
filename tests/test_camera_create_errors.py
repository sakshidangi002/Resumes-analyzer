"""Creating a camera must report WHY it was rejected.

`create_camera` wrapped its body in `except Exception` and re-raised everything
as a bare 500 "Unable to create the camera." Because HTTPException subclasses
Exception, the precise 422 from the SSRF guard was swallowed too — an operator
typing a bad host saw a 500 with no reason, while the real message
("must resolve to a private network") appeared only in the server log.

Also pins that the rejection names the offending host, and that the payload log
does not carry the DVR password.
"""
import inspect

import pytest
from app.api.routes import cameras as cameras_route
from fastapi import HTTPException


def test_http_exception_is_re_raised_before_the_generic_handler():
    """Order matters: `except HTTPException: raise` must precede
    `except Exception`, or every precise status collapses into a 500."""
    source = inspect.getsource(cameras_route.create_camera)
    http_at = source.find("except HTTPException")
    generic_at = source.find("except Exception")
    assert http_at != -1, "create_camera no longer re-raises HTTPException"
    assert generic_at != -1
    assert http_at < generic_at, (
        "`except Exception` comes first, so it swallows HTTPException and turns "
        "every 422 back into an opaque 500"
    )


@pytest.mark.parametrize(
    "host,expect_in_detail",
    [
        ("8.8.8.8", "8.8.8.8"),
        ("1.1.1.1", "1.1.1.1"),
    ],
)
def test_public_host_is_rejected_and_named(host, expect_in_detail):
    with pytest.raises(HTTPException) as exc:
        cameras_route._validate_camera_source(f"rtsp://{host}:554/s", "rtsp")
    assert exc.value.status_code == 422
    assert expect_in_detail in exc.value.detail, (
        f"the rejection does not say which host was refused: {exc.value.detail!r}"
    )


def test_unresolvable_host_names_itself():
    with pytest.raises(HTTPException) as exc:
        cameras_route._validate_camera_source(
            "rtsp://no-such-host.invalid:554/s", "rtsp"
        )
    assert exc.value.status_code == 422
    assert "no-such-host.invalid" in exc.value.detail


@pytest.mark.parametrize(
    "url",
    [
        "rtsp://192.168.29.181:554/Streaming/Channels/201",
        "rtsp://user:pw@192.168.29.181:554/Streaming/Channels/201",
        "rtsp://10.0.0.5:554/s",
    ],
)
def test_private_hosts_are_accepted(url):
    """The guard must not block the actual use case."""
    assert cameras_route._validate_camera_source(url, "rtsp") == url


def test_usb_source_bypasses_dns_resolution():
    assert cameras_route._validate_camera_source("0", "usb") == "0"


def test_create_camera_does_not_log_the_password():
    """That log line ran on every create and printed the whole payload."""
    source = inspect.getsource(cameras_route.create_camera)
    assert "_redact_url" in source, (
        "the create payload is logged unredacted — it contains the DVR password"
    )
    assert "payload.model_dump()}" not in source, (
        "the raw payload is still being f-string-logged"
    )
