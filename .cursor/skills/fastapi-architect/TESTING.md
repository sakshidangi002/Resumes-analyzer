# Testing — Pytest for FastAPI + SQLModel

Authoritative reference for testing FastAPI apps. Read when adding tests, designing fixtures, or improving coverage.

## Test pyramid for backend

```
       /\           E2E (real DB, real HTTP)  — few, slow, brittle
      /  \
     /----\         Integration (test DB, TestClient)  — most of your tests
    /      \
   /--------\       Unit (pure functions, mocks)  — fast, many
```

Default split: 60% integration, 30% unit, 10% E2E. Integration tests via `TestClient` cover routers + services + CRUD in one shot — biggest ROI per test.

## Project layout

```
tests/
├── conftest.py              # global fixtures: app, client, session, test DB
├── factories.py             # data factories (use factory-boy or hand-rolled)
├── test_auth.py
├── test_users.py
├── test_employees.py
├── test_leave_service.py    # unit tests for service logic
└── ...
```

## conftest.py — the foundation

```python
# tests/conftest.py
import asyncio
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlmodel import SQLModel

from app.main import app
from app.api.deps import get_session
from app.core.config import get_settings

TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/test_app"

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)
    yield engine
    await engine.dispose()

@pytest_asyncio.fixture
async def session(test_engine) -> AsyncSession:
    """Per-test session, rolled back at the end. Tests stay isolated."""
    connection = await test_engine.connect()
    transaction = await connection.begin()
    SessionLocal = async_sessionmaker(bind=connection, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as s:
        yield s
    await transaction.rollback()
    await connection.close()

@pytest_asyncio.fixture
async def client(session) -> AsyncClient:
    async def override_get_session():
        yield session
    app.dependency_overrides[get_session] = override_get_session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
```

**Key choices:**

- **Separate test database**, never the dev DB. CI uses `postgres:16-alpine` in a service container.
- **Per-test transaction rollback** — tests don't see each other's data, no manual cleanup.
- **`AsyncClient`** (not `TestClient`) for async apps — `TestClient` runs sync internally and can deadlock with async sessions.
- **`dependency_overrides`** swaps `get_session` per test — no monkey-patching.

## Auth fixtures

```python
# tests/conftest.py (continued)
@pytest_asyncio.fixture
async def admin_user(session) -> User:
    user = User(
        email="admin@test.com",
        hashed_password=get_password_hash("password123"),
        full_name="Test Admin",
        role="admin",
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user

@pytest_asyncio.fixture
async def admin_token(admin_user) -> str:
    return create_access_token(admin_user.id, admin_user.role)

@pytest_asyncio.fixture
async def admin_client(client, admin_token) -> AsyncClient:
    client.headers["Authorization"] = f"Bearer {admin_token}"
    return client
```

Pattern: `<role>_user` → `<role>_token` → `<role>_client`. Tests can then ask for `admin_client`, `hr_client`, `employee_client` directly.

## Anatomy of a good API test

```python
# tests/test_employees.py
import pytest

@pytest.mark.asyncio
class TestCreateEmployee:
    async def test_admin_can_create(self, admin_client, session):
        payload = {
            "user_id": 1,
            "employee_code": "EMP001",
            "joined_on": "2026-01-15",
        }
        r = await admin_client.post("/api/v1/employees", json=payload)
        assert r.status_code == 201
        body = r.json()
        assert body["employee_code"] == "EMP001"
        assert "id" in body
        assert "hashed_password" not in body  # no leakage

    async def test_employee_role_forbidden(self, employee_client):
        r = await employee_client.post("/api/v1/employees", json={...})
        assert r.status_code == 403

    async def test_no_token_unauthorized(self, client):
        r = await client.post("/api/v1/employees", json={...})
        assert r.status_code == 401

    async def test_duplicate_code_conflict(self, admin_client, session):
        await admin_client.post("/api/v1/employees", json={"employee_code": "EMP001", ...})
        r = await admin_client.post("/api/v1/employees", json={"employee_code": "EMP001", ...})
        assert r.status_code == 409

    async def test_invalid_payload_validation(self, admin_client):
        r = await admin_client.post("/api/v1/employees", json={"employee_code": ""})
        assert r.status_code == 422
        assert "employee_code" in r.text
```

**Minimum test set per endpoint:**
1. Happy path with the lowest authorized role.
2. 401 — no token.
3. 403 — authenticated but wrong role / wrong owner.
4. 422 — invalid input (missing required field).
5. 404 — resource missing (for GET/PUT/DELETE on `/{id}`).
6. 409 — uniqueness / state conflict (where applicable).
7. Edge cases specific to the endpoint (date ranges, file sizes, etc.).

## Unit tests for services

When a service has non-trivial logic, test it directly without going through HTTP:

```python
# tests/test_leave_service.py
@pytest.mark.asyncio
async def test_apply_for_leave_deducts_balance(session, employee_factory, leave_balance_factory):
    emp = await employee_factory(session, balance=10)
    leave_in = LeaveCreate(start_date=date(2026, 6, 1), end_date=date(2026, 6, 3))

    leave = await leave_service.apply_for_leave(session, employee_id=emp.id, leave_in=leave_in)

    assert leave.status == LeaveStatus.pending
    balance = await leave_balance_crud.get_for_employee(session, emp.id)
    assert balance.remaining == 7

@pytest.mark.asyncio
async def test_apply_for_leave_exceeds_balance_raises(session, employee_factory):
    emp = await employee_factory(session, balance=2)
    leave_in = LeaveCreate(start_date=date(2026, 6, 1), end_date=date(2026, 6, 10))

    with pytest.raises(BusinessRuleViolation, match="balance"):
        await leave_service.apply_for_leave(session, employee_id=emp.id, leave_in=leave_in)
```

## Mocking external services

Don't hit real SMTP / S3 / external APIs in tests. Mock at the boundary:

```python
# tests/test_password_reset.py
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_password_reset_sends_email(client, regular_user):
    with patch("app.services.auth_service.send_email", new=AsyncMock()) as mock_send:
        r = await client.post("/api/v1/auth/forgot-password", json={"email": regular_user.email})
        assert r.status_code == 202
        mock_send.assert_awaited_once()
        args = mock_send.await_args
        assert regular_user.email in args.kwargs["to"]
        assert "reset" in args.kwargs["subject"].lower()
```

**Mock the function you call, not its dependencies.** Patch `app.services.auth_service.send_email`, not `smtplib.SMTP`.

## Data factories

For non-trivial test data, hand-rolled async factories beat factory-boy:

```python
# tests/factories.py
async def employee_factory(session, **overrides) -> Employee:
    defaults = dict(
        employee_code=f"EMP{random.randint(1000, 9999)}",
        user_id=(await user_factory(session)).id,
        joined_on=date.today(),
        is_active=True,
    )
    defaults.update(overrides)
    emp = Employee(**defaults)
    session.add(emp)
    await session.commit()
    await session.refresh(emp)
    return emp
```

Register as a fixture:

```python
@pytest_asyncio.fixture
async def employee_factory(session):
    async def _make(**kw):
        return await _employee_factory_impl(session, **kw)
    return _make
```

Usage: `emp = await employee_factory(employee_code="EMP123")`.

## Edge cases worth a test

- **Pagination boundaries:** `skip=0 limit=1`, `skip=999999`, `limit=0` (should 422).
- **Date ranges:** end before start, same-day, ranges spanning DST / year boundary, future dates where business rule says past-only.
- **String fields at max length:** `"x" * 255` should pass, `"x" * 256` should 422.
- **Concurrent writes:** rarely test, but if you have a counter or balance, write at least one test that confirms two parallel updates don't race (use `select_for_update` or optimistic locking).
- **File upload:** wrong MIME, oversized, empty, file claiming to be PDF but isn't.
- **JWT:** expired token (401), tampered token (401), refresh token sent to protected endpoint (401), access token sent to refresh endpoint (401).
- **Role escalation attempts:** employee tries to PATCH their own role to admin (must 403 or be silently dropped by schema).

## Running tests

```bash
# Whole suite
pytest

# One file
pytest tests/test_employees.py -v

# One test, with output
pytest tests/test_employees.py::TestCreateEmployee::test_admin_can_create -v -s

# With coverage
pytest --cov=app --cov-report=term-missing --cov-report=html

# Stop on first failure, drop into debugger
pytest -x --pdb
```

`pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
    "slow: long-running tests (deselect with -m 'not slow')",
]
```

`asyncio_mode = "auto"` means you don't need `@pytest.mark.asyncio` on every test — saves boilerplate.

## CI checklist

In CI, run:
1. `ruff check .` — lint, fast.
2. `mypy app` — type check (optional but recommended).
3. `pytest --cov=app --cov-fail-under=80` — tests + coverage gate.
4. `alembic upgrade head` on a fresh DB — migrations apply cleanly.

A failing migration in CI is one of the few absolute red lines — never merge.

## Coverage targets

- **Routers**: 100% — every status code path.
- **Services**: 90%+ — every business rule branch.
- **CRUD**: 70%+ — the generic base is tested once; custom queries each get one test.
- **Utils**: 100% for pure functions, less critical for thin wrappers.

**Don't chase 100% overall.** Coverage of generated code, `__init__.py`, and Alembic `versions/` is meaningless. Configure exclusions in `.coveragerc`.

## What NOT to test

- Pydantic validation per se — Pydantic is already tested. Test that YOUR endpoint returns 422 for bad input, don't test that `EmailStr` rejects `"foo"`.
- SQLAlchemy itself — don't write tests proving "session.add then session.commit persists the row". Test YOUR query logic.
- Framework integration code — `FastAPI()` initialization, middleware order. Trust the framework.
