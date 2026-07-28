"""Camera SSRF guard and client-address resolution.

The SSRF guard shipped once in a form that allowed 169.254.169.254 — Python
reports link-local as `is_private`, so a private-or-loopback test let the cloud
metadata endpoint straight through. That address is the whole reason the guard
exists, so it gets an explicit test.
"""
import os

import pytest
from app.core.net import client_ip, is_allowed_camera_host_ip


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    def __init__(self, host="10.1.2.3", headers=None):
        self.client = _FakeClient(host) if host else None
        self.headers = headers or {}


@pytest.mark.parametrize("address", [
    "169.254.169.254",   # AWS/GCP/Azure metadata — credentials endpoint
    "169.254.1.1",       # link-local generally
    "fe80::1",           # IPv6 link-local
    "8.8.8.8",           # public
    "1.1.1.1",           # public
    "224.0.0.1",         # multicast
])
def test_camera_hosts_outside_the_lan_are_rejected(address):
    assert not is_allowed_camera_host_ip(address)


@pytest.mark.parametrize("address", [
    "10.0.0.5",
    "192.168.1.50",
    "172.16.4.9",
    "127.0.0.1",
    "::1",
])
def test_lan_and_loopback_cameras_are_allowed(address):
    assert is_allowed_camera_host_ip(address)


def test_forwarded_headers_are_ignored_unless_explicitly_trusted():
    """X-Forwarded-For is client-supplied.

    Trusting it on a directly-exposed server would let an attacker forge a fresh
    address per request and walk past every per-IP counter.
    """
    os.environ.pop("TRUST_PROXY_HEADERS", None)
    request = _FakeRequest("10.1.2.3", {"x-forwarded-for": "1.2.3.4"})
    assert client_ip(request) == "10.1.2.3"


def test_forwarded_headers_are_honoured_when_trusted():
    """Behind a proxy the socket address is the proxy for every user, so
    without this the whole office shares one rate-limit bucket."""
    os.environ["TRUST_PROXY_HEADERS"] = "1"
    try:
        request = _FakeRequest("10.1.2.3", {"x-forwarded-for": "203.0.113.9, 10.1.2.3"})
        assert client_ip(request) == "203.0.113.9"   # left-most = original client

        request = _FakeRequest("10.1.2.3", {"x-real-ip": "203.0.113.10"})
        assert client_ip(request) == "203.0.113.10"
    finally:
        os.environ.pop("TRUST_PROXY_HEADERS", None)


def test_missing_client_does_not_raise():
    assert client_ip(_FakeRequest(host=None)) == "unknown"
