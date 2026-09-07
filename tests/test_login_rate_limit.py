"""Login throttling.

The first version keyed the per-account counter as "user:{ip}:{username}". That
key contains the IP, so its count could never exceed the IP counter, and both
used the same threshold — the per-account limit was unreachable dead code while
a 5-failure IP ceiling locked out whole offices behind one NAT address.
"""
from typing import ClassVar

import pytest

# Importing anything under app.api.routes pulls in the vision stack (cv2, torch,
# PIL, numpy) via the face/recognition services. Where that is not installed,
# skip cleanly instead of erroring the whole collection. CI runs pytest with -rs
# so the skip is REPORTED — a test that silently stops running is worse than no
# test, because it still looks like coverage.
try:
    from app.api.routes import auth
except ImportError as exc:  # pragma: no cover - depends on the environment
    pytest.skip(
        f"requires the vision stack pulled in by app.api.routes ({exc}); "
        "install 'requirements.txt' to run",
        allow_module_level=True,
    )


class _FakeClient:
    host = "10.0.0.7"


class _FakeRequest:
    client = _FakeClient()
    headers: ClassVar[dict] = {}


@pytest.fixture(autouse=True)
def _clear_state():
    auth._LOGIN_FAILURES.clear()
    yield
    auth._LOGIN_FAILURES.clear()


def test_account_key_is_not_scoped_by_ip():
    """Otherwise distributed guessing against one account is unlimited."""
    account, ip = auth._login_keys(_FakeRequest(), "Alice")
    assert account == "account:alice"          # normalised, no address
    assert "10.0.0.7" not in account
    assert ip == "ip:10.0.0.7"


def test_ip_ceiling_is_far_above_the_account_ceiling():
    """A shared office NAT puts every employee behind one address, so the IP
    limit must never be what trips first in normal use."""
    account, ip = auth._login_keys(_FakeRequest(), "alice")
    assert auth._limit_for(account) == auth._LOGIN_MAX_ATTEMPTS
    assert auth._limit_for(ip) == auth._LOGIN_MAX_ATTEMPTS_PER_IP
    assert auth._limit_for(ip) > auth._limit_for(account) * 5


def test_account_locks_after_the_configured_failures():
    keys = auth._login_keys(_FakeRequest(), "alice")
    for _ in range(auth._LOGIN_MAX_ATTEMPTS - 1):
        auth._record_login_failure(keys)
    assert auth._check_login_rate_limit(keys) is None       # still under the limit

    auth._record_login_failure(keys)
    retry_after = auth._check_login_rate_limit(keys)
    assert retry_after and retry_after > 0


def test_successful_login_clears_the_account_counter_but_not_the_ip_backstop():
    """Clearing the IP counter on success would let an attacker holding ONE
    valid credential reset the backstop at will: guess, guess, log in, repeat."""
    keys = auth._login_keys(_FakeRequest(), "alice")
    account_key, ip_key = keys
    for _ in range(auth._LOGIN_MAX_ATTEMPTS):
        auth._record_login_failure(keys)

    auth._clear_login_failures(keys)

    assert auth._check_login_rate_limit(keys) is None
    assert account_key not in auth._LOGIN_FAILURES
    assert len(auth._LOGIN_FAILURES.get(ip_key, [])) == auth._LOGIN_MAX_ATTEMPTS


def test_one_account_lockout_does_not_lock_a_different_account():
    victim = auth._login_keys(_FakeRequest(), "alice")
    other = auth._login_keys(_FakeRequest(), "bob")
    for _ in range(auth._LOGIN_MAX_ATTEMPTS):
        auth._record_login_failure(victim)

    assert auth._check_login_rate_limit(victim) is not None
    assert auth._check_login_rate_limit(other) is None   # IP ceiling is much higher
