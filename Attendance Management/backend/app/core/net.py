"""Network address helpers: client identification and camera-host validation.

This lives in one place because the two limiters disagreeing is a real outage
risk, not a style problem: the login limiter honoured X-Forwarded-For while the
middleware read the raw socket address, so behind a reverse proxy one of them
saw real clients and the other saw a single shared address for the whole office.
"""
import ipaddress
import os

from fastapi import Request


def is_allowed_camera_host_ip(address: str) -> bool:
    """True if a camera source may point at this address (SSRF guard).

    Cameras live on the LAN, so only private and loopback ranges are allowed.

    Link-local is rejected EXPLICITLY even though Python reports 169.254.0.0/16
    as `is_private`: a bare private-or-loopback test lets 169.254.169.254
    through, and that is the cloud metadata endpoint which hands out instance
    credentials to anything that can ask. fe80::/10 is the IPv6 equivalent.

    Lives here, away from the camera routes, so it can be tested without
    importing OpenCV and the rest of the vision stack.
    """
    ip = ipaddress.ip_address(address)
    # Loopback first and unconditionally: it can never leave the host, so it is
    # always safe. It needs its own branch because Python marks IPv6 ::1 as
    # `is_reserved` (it falls inside ::/8), which the exclusion below would
    # otherwise reject — blocking a legitimate local test camera.
    if ip.is_loopback:
        return True
    return ip.is_private and not (
        ip.is_link_local or ip.is_reserved or ip.is_multicast
    )


def _trust_proxy_headers() -> bool:
    """Read the flag on each call so tests (and a restart-free config change)
    take effect; this is not on a hot enough path to cache."""
    return os.getenv("TRUST_PROXY_HEADERS", "").strip().lower() in ("1", "true", "yes")


def client_ip(request: Request) -> str:
    """Best-effort originating address for rate-limiting purposes.

    X-Forwarded-For is only consulted when TRUST_PROXY_HEADERS is set, because
    the header is client-supplied: trusting it on a directly-exposed server lets
    an attacker forge a fresh address per request and walk straight past any
    per-IP counter.

    Conversely, leaving it OFF *behind* a proxy makes every request look like it
    came from the proxy, so all users share one bucket. Set TRUST_PROXY_HEADERS=1
    when deployed behind a proxy you control, and make that proxy OVERWRITE
    X-Forwarded-For rather than append to it.
    """
    if _trust_proxy_headers():
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            # Left-most entry is the original client.
            return forwarded.split(",")[0].strip()[:100] or "unknown"
        real_ip = request.headers.get("x-real-ip", "").strip()
        if real_ip:
            return real_ip[:100]
    return request.client.host if request.client else "unknown"
