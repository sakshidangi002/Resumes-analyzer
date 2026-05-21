# Architecture & Layering

This file is the authoritative reference for **how to structure a FastAPI codebase** so it stays maintainable past 50 endpoints. Read it before scaffolding a new project or proposing a non-trivial refactor.

## The four-layer rule

```
HTTP request
    │
    ▼
[router]         ◄── thin: parse input, call service, return response_model
    │
    ▼
[service]        ◄── business logic, transactions, multi-model orchestration
    │
    ▼
[crud]           ◄── single-model DB operations (get, list, create, update, delete)
    │
    ▼
[model]          ◄── SQLModel table=True, relationships, constraints
```

**Rules:**

- Routers may import from `services` and `schemas`. NEVER from `crud` or `models` directly.
- Services may import from `crud`, other services, `models`, and `schemas`.
- CRUD may import from `models` and `schemas`. NEVER from services or routers.
- Models import only from `db.base` and other models.

Violating this creates circular imports and makes layers untestable.

## When to skip a layer

For genuinely trivial endpoints (e.g. `GET /health`, `GET /version`), put logic directly in the router. **Don't create empty service functions** just to satisfy the layering — that's cargo-cult architecture. The four-layer rule is for endpoints that touch the DB.

## Folder responsibilities

### `app/core/`

Cross-cutting concerns. Should be importable from anywhere.

- `config.py` — `Settings(BaseSettings)` from `pydantic-settings`. Reads `.env`. **One settings object per app, cached with `@lru_cache`.**
- `security.py` — JWT encode/decode, password hash/verify. No HTTP, no DB.
- `logging.py` — `logging.dictConfig(...)` at startup. Structured (JSON) in prod, pretty in dev.
- `exceptions.py` — custom exceptions (`NotFoundError`, `PermissionDeniedError`, `BusinessRuleViolation`) + their FastAPI handlers.

### `app/db/`

- `session.py` — async engine, `async_sessionmaker`, `get_session` async dependency.
- `base.py` — `from app.models import *` so Alembic sees all tables in `SQLModel.metadata`.

### `app/models/`

SQLModel classes with `table=True`. ONE model per file unless models are tightly coupled (e.g. polymorphic).

```python
# app/models/employee.py
from datetime import date
from sqlmodel import SQLModel, Field, Relationship

class Employee(SQLModel, table=True):
    __tablename__ = "employees"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", unique=True, index=True)
    employee_code: str = Field(unique=True, index=True, max_length=20)
    department_id: int | None = Field(default=None, foreign_key="departments.id", index=True)
    joined_on: date
    is_active: bool = Field(default=True, index=True)

    user: "User" = Relationship(back_populates="employee")
    department: "Department | None" = Relationship(back_populates="employees")
```

**Always set:** `index=True` on FK columns and frequently-filtered columns, `unique=True` on natural keys, `max_length` on strings, `nullable` via `| None` in the type hint.

### `app/schemas/`

SQLModel classes WITHOUT `table=True`, or plain Pydantic models for non-DB shapes. Always 3+ variants per resource:

```python
# app/schemas/employee.py
from datetime import date
from sqlmodel import SQLModel

class EmployeeBase(SQLModel):
    employee_code: str
    department_id: int | None = None
    joined_on: date

class EmployeeCreate(EmployeeBase):
    user_id: int

class EmployeeUpdate(SQLModel):
    department_id: int | None = None
    is_active: bool | None = None

class EmployeeRead(EmployeeBase):
    id: int
    is_active: bool

class EmployeeReadDetailed(EmployeeRead):
    department: "DepartmentRead | None" = None
    user: "UserRead"
```

**Why separate Read/Create/Update:** different fields are allowed/required in each context. `id` is read-only; `user_id` is set on create only; `is_active` is updatable but shouldn't appear on create.

### `app/crud/`

Single-model DB ops. Use a generic base for the common 80%:

```python
# app/crud/base.py — see templates/crud_base.py for full code
class CRUDBase(Generic[ModelT, CreateT, UpdateT]):
    async def get(self, session: AsyncSession, id: int) -> ModelT | None: ...
    async def get_multi(self, session, *, skip=0, limit=100, **filters) -> list[ModelT]: ...
    async def create(self, session, *, obj_in: CreateT) -> ModelT: ...
    async def update(self, session, *, db_obj: ModelT, obj_in: UpdateT) -> ModelT: ...
    async def delete(self, session, *, id: int) -> ModelT | None: ...
```

Custom queries get their own method on the model-specific CRUD:

```python
# app/crud/employee.py
class CRUDEmployee(CRUDBase[Employee, EmployeeCreate, EmployeeUpdate]):
    async def get_by_code(self, session, code: str) -> Employee | None:
        stmt = select(Employee).where(Employee.employee_code == code)
        return (await session.exec(stmt)).first()

    async def list_by_department(self, session, dept_id: int) -> list[Employee]:
        stmt = select(Employee).where(
            Employee.department_id == dept_id,
            Employee.is_active == True,
        ).options(selectinload(Employee.user))
        return (await session.exec(stmt)).all()

employee_crud = CRUDEmployee(Employee)
```

### `app/services/`

Business logic. Multiple CRUD calls, transactions, validation that can't live in Pydantic (e.g. "leave balance can't go negative").

```python
# app/services/leave_service.py
class LeaveService:
    async def apply_for_leave(
        self,
        session: AsyncSession,
        *,
        employee_id: int,
        leave_in: LeaveCreate,
    ) -> Leave:
        balance = await leave_balance_crud.get_for_employee(session, employee_id)
        days = (leave_in.end_date - leave_in.start_date).days + 1
        if days > balance.remaining:
            raise BusinessRuleViolation(
                f"Requested {days} days but balance is {balance.remaining}"
            )

        leave = await leave_crud.create(
            session,
            obj_in=leave_in.model_copy(update={"employee_id": employee_id, "status": "pending"}),
        )
        await leave_balance_crud.deduct(session, employee_id, days)
        await session.commit()
        return leave

leave_service = LeaveService()
```

**Service rules:**
- Accept `AsyncSession` as a parameter. Don't create your own session.
- Commit at the boundary, after all writes succeed. One transaction per service call.
- Raise domain exceptions (`BusinessRuleViolation`, `NotFoundError`), not `HTTPException`. Translate to HTTP in the exception handler.
- Never accept the FastAPI `Request` object. Stay HTTP-free for testability.

### `app/api/v1/`

Routers. One file per resource. All routers mounted under `/api/v1` for clean versioning.

```python
# app/api/v1/employees.py
router = APIRouter(prefix="/employees", tags=["employees"])

@router.get("", response_model=Page[EmployeeRead])
async def list_employees(
    session: AsyncSession = Depends(get_session),
    _: User = Depends(require_role("admin", "hr")),
    pagination: PaginationParams = Depends(),
    department_id: int | None = None,
):
    return await employee_service.list_paginated(
        session, pagination=pagination, department_id=department_id
    )

@router.post("", response_model=EmployeeRead, status_code=201)
async def create_employee(
    payload: EmployeeCreate,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(require_role("admin", "hr")),
):
    return await employee_service.create(session, payload)
```

### `app/api/deps.py`

ALL reusable dependencies live here. Don't define dependencies inline in routers.

- `get_session()` — yields `AsyncSession`, rolls back on exception.
- `get_current_user()` — decodes JWT, fetches user, raises 401.
- `require_role(*roles)` — depends on `get_current_user`, raises 403.
- `PaginationParams` — `skip`, `limit`, `sort_by`, `sort_dir`.

## API versioning

- Mount all routes under `/api/v1/...`. Even if you never ship v2, the prefix is free insurance.
- `app.include_router(api_router_v1, prefix="/api/v1")`.
- When breaking changes come: copy `app/api/v1/` → `app/api/v2/`, keep v1 running, deprecate over 2 releases.
- NEVER version individual endpoints (`/users/v2/create`). Version the whole surface.

## Configuration

```python
# app/core/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = "FastAPI Backend"
    ENVIRONMENT: str = "development"  # development | staging | production
    DEBUG: bool = False

    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    CORS_ORIGINS: list[str] = ["http://localhost:5173"]

    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 10

    SMTP_HOST: str | None = None
    SMTP_PORT: int = 587
    SMTP_USER: str | None = None
    SMTP_PASSWORD: str | None = None

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

**Why `@lru_cache`:** Settings is read once at startup. Caching avoids re-parsing `.env` on every request and allows overriding in tests via `app.dependency_overrides`.

## Logging

```python
# app/core/logging.py
import logging
import sys
from app.core.config import get_settings

def configure_logging() -> None:
    settings = get_settings()
    level = logging.DEBUG if settings.DEBUG else logging.INFO

    if settings.ENVIRONMENT == "production":
        fmt = '{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
    else:
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)
    logging.getLogger("uvicorn.access").handlers.clear()  # avoid double logs
```

Use `logger = logging.getLogger(__name__)` in every module. Never `print()`.

## Exception handling

Three custom exceptions cover 90% of needs:

```python
# app/core/exceptions.py
class AppException(Exception):
    status_code: int = 500
    detail: str = "Internal server error"

class NotFoundError(AppException):
    status_code = 404
    def __init__(self, resource: str, id: int | str):
        self.detail = f"{resource} with id={id} not found"

class PermissionDeniedError(AppException):
    status_code = 403
    detail = "Permission denied"

class BusinessRuleViolation(AppException):
    status_code = 409
    def __init__(self, message: str):
        self.detail = message
```

Register one global handler in `main.py`:

```python
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
```

This lets services raise domain exceptions without knowing about HTTP.

## Middleware order (matters!)

```python
app.add_middleware(CORSMiddleware, ...)         # outermost
app.add_middleware(RequestIDMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1000)
# auth lives in dependencies, not middleware — gives per-route opt-out
```

Auth as a dependency (not middleware) lets you mark specific routes public without elaborate skip-lists.

## Dependency injection patterns

**Good — composable:**

```python
async def get_current_user(token: str = Depends(oauth2_scheme), session = Depends(get_session)) -> User: ...
def require_role(*roles: str):
    async def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise PermissionDeniedError()
        return user
    return checker
```

**Bad — closures over module-level mutables, hardcoded role lists:**

```python
ADMIN_USERS = []  # never do this
async def is_admin(token = Depends(...)):
    if user not in ADMIN_USERS: ...
```

## Lifespan over startup/shutdown events

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    await ping_db()
    yield
    await engine.dispose()

app = FastAPI(lifespan=lifespan, title=settings.PROJECT_NAME)
```

`@app.on_event("startup")` is deprecated. Always use `lifespan`.
