"""One request must not check out two database connections.

MEASURED CONSEQUENCE, on the live system: PostgreSQL began refusing new
connections with

    FATAL: sorry, too many clients already

and CCTV transit events were lost because `unknown_attendance.record` could not
get a session.

THE CAUSE. FastAPI caches resolved dependencies per request, keyed on the
CALLABLE OBJECT (see `fastapi.dependencies.utils.solve_dependencies`, which
looks up `dependency_cache[(call, security_scopes)]`).
`app.api.deps.get_db_session` used to be a wrapper --

    def get_db_session():
        yield from get_db()

-- which is a different object from `app.db.session.get_db`. 176 route handlers
depend on the first name; the entire authentication chain depends on the second.
Two callables meant two independent resolutions, so two sessions and two pooled
connections for every authenticated request.

With `pool_size=20, max_overflow=20` and Starlette's 40-thread default for sync
handlers, that is up to 80 connections wanted against 40 available.

Aliasing them also fixes a quieter bug: `current_user` was loaded on a different
session from the one the handler committed, so mutating `current_user` and
committing `db` silently dropped the write.

These tests pin the alias. A future refactor that reintroduces a wrapper --
even an apparently harmless one -- brings the exhaustion back, and it would
surface as a production outage rather than a test failure.

No HTTP client here: every test in this directory is a pure unit test (see
conftest), and starlette 0.36's TestClient is incompatible with the installed
httpx 0.28. The dependency-cache key is asserted directly instead, which is the
exact mechanism that decides whether one or two sessions are opened.
"""
import pytest

pytest.importorskip("fastapi", reason="needs fastapi")

from app.api.deps import get_db_session
from app.db.session import get_db


def test_the_two_dependency_names_are_one_callable():
    """The whole fix. Different objects here means two connections per request."""
    assert get_db_session is get_db, (
        "get_db_session must BE get_db, not wrap it -- FastAPI caches "
        "dependencies by callable identity, so a wrapper doubles the "
        "connection checkout for every authenticated request"
    )


def test_both_names_share_one_dependency_cache_key():
    """The mechanism, asserted directly.

    FastAPI's per-request cache is keyed on (call, security_scopes). Equal keys
    are what make the handler's `Depends(get_db)` and the auth chain's
    `Depends(get_db_session)` resolve to a single session.
    """
    from fastapi.dependencies.models import Dependant

    handler_dep = Dependant(call=get_db, security_scopes=[])
    auth_dep = Dependant(call=get_db_session, security_scopes=[])

    assert handler_dep.cache_key == auth_dep.cache_key, (
        "the two dependencies do not share a cache key, so FastAPI will "
        "resolve each separately and open a session for each"
    )


def test_the_generator_closes_its_session(monkeypatch):
    """Normal completion must return the connection to the pool."""
    closed: list[bool] = []

    class _FakeSession:
        def close(self):
            closed.append(True)

    monkeypatch.setattr("app.db.session.SessionLocal", _FakeSession)

    gen = get_db()
    next(gen)                       # dependency yields the session
    with pytest.raises(StopIteration):
        next(gen)                   # FastAPI exhausts it after the response
    assert closed == [True]


def test_the_generator_closes_its_session_when_the_handler_raises(monkeypatch):
    """An exception path must not strand a pooled connection.

    This is the path that matters under load: if a handler raising left its
    connection checked out, an error spike would exhaust the pool and turn a
    transient fault into an outage.
    """
    closed: list[bool] = []

    class _FakeSession:
        def close(self):
            closed.append(True)

    monkeypatch.setattr("app.db.session.SessionLocal", _FakeSession)

    gen = get_db()
    next(gen)
    with pytest.raises(RuntimeError):
        gen.throw(RuntimeError("handler failed"))
    assert closed == [True], "session was not closed when the handler raised"
