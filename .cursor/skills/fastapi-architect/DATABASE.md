# Database — SQLModel + PostgreSQL + Async + Alembic

Authoritative reference for DB layer work. Read before writing models, queries, migrations, or debugging slow endpoints.

## Engine & session (async)

```python
# app/db/session.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from app.core.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.DATABASE_URL,                     # postgresql+asyncpg://user:pass@host/db
    echo=settings.DEBUG,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,                        # detect dead connections
    pool_recycle=3600,                         # recycle hourly (avoid stale conns behind LB)
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)

async def get_session():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        # commit happens in service, not here
```

**Key choices:**

- **`asyncpg` driver** (not `psycopg`): `postgresql+asyncpg://...`. Faster, true async.
- **`expire_on_commit=False`**: keeps objects usable after commit (needed for returning them in response).
- **`autoflush=False`**: prevents surprise flushes; you flush explicitly when needed.
- **`pool_pre_ping=True`**: essential behind any load balancer / connection pooler (PgBouncer).
- **`pool_recycle=3600`**: AWS RDS and many providers kill idle connections silently.

## SQLModel patterns

### Table model

```python
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, DateTime, func

class User(SQLModel, table=True):
    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True, max_length=255)
    hashed_password: str = Field(max_length=255)
    full_name: str = Field(max_length=100)
    role: str = Field(default="employee", index=True, max_length=20)
    is_active: bool = Field(default=True, index=True)

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    )
    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),
            onupdate=func.now(),
            nullable=False,
        )
    )

    employee: "Employee | None" = Relationship(back_populates="user")
```

**Always:**
- Timestamps via `server_default=func.now()` — DB handles the time, not Python.
- `DateTime(timezone=True)` — store UTC, convert in the app.
- `index=True` on any column you filter or sort by.
- `max_length` on every string (asserts schema in Postgres, helps query planner).

### Relationships

```python
class Employee(SQLModel, table=True):
    __tablename__ = "employees"
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", unique=True, index=True)
    department_id: int | None = Field(default=None, foreign_key="departments.id", index=True)

    user: User = Relationship(back_populates="employee")
    department: "Department | None" = Relationship(back_populates="employees")
    attendances: list["Attendance"] = Relationship(
        back_populates="employee",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
```

**Forward references** (`"Department | None"`) avoid circular imports. Always quote when the related class is defined later or in another module.

### Enums

Use Python `Enum` + SQLAlchemy `Enum` column. NEVER store raw strings for status fields:

```python
from enum import Enum as PyEnum
from sqlalchemy import Column, Enum

class LeaveStatus(str, PyEnum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    cancelled = "cancelled"

class Leave(SQLModel, table=True):
    status: LeaveStatus = Field(
        sa_column=Column(Enum(LeaveStatus, name="leave_status"), nullable=False, index=True)
    )
```

Adding an enum value later requires an Alembic migration with `ALTER TYPE ... ADD VALUE`.

## Queries

### Use `select()` always; never `session.query()` in async

```python
from sqlmodel import select

stmt = select(Employee).where(Employee.is_active == True)
result = await session.exec(stmt)
employees = result.all()                    # list[Employee]

# Single result:
stmt = select(Employee).where(Employee.id == emp_id)
employee = (await session.exec(stmt)).first()
```

### Eager loading (kill N+1 queries)

```python
from sqlalchemy.orm import selectinload, joinedload

# selectinload — separate query per relationship (best for one-to-many)
stmt = (
    select(Employee)
    .options(selectinload(Employee.attendances), selectinload(Employee.user))
    .where(Employee.is_active == True)
)

# joinedload — JOIN (best for many-to-one, when you need the field always)
stmt = select(Employee).options(joinedload(Employee.department))
```

**Rule of thumb:**
- One-to-many / many-to-many → `selectinload` (avoids cartesian explosion).
- Many-to-one / one-to-one → `joinedload`.

If you write a loop like `for emp in employees: print(emp.department.name)` without eager loading, you have N+1. Always check.

### Pagination

```python
# app/api/deps.py
from fastapi import Query
from pydantic import BaseModel

class PaginationParams(BaseModel):
    skip: int = 0
    limit: int = 50

    @classmethod
    def from_query(cls,
        skip: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
    ):
        return cls(skip=skip, limit=limit)

# Page response
from typing import Generic, TypeVar
T = TypeVar("T")

class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    skip: int
    limit: int
```

```python
# service
async def list_paginated(session, *, pagination: PaginationParams, **filters) -> Page[EmployeeRead]:
    base_stmt = select(Employee)
    for k, v in filters.items():
        if v is not None:
            base_stmt = base_stmt.where(getattr(Employee, k) == v)

    count_stmt = select(func.count()).select_from(base_stmt.subquery())
    total = (await session.exec(count_stmt)).one()

    items_stmt = base_stmt.offset(pagination.skip).limit(pagination.limit)
    items = (await session.exec(items_stmt)).all()
    return Page(items=items, total=total, skip=pagination.skip, limit=pagination.limit)
```

For very large tables (>1M rows), prefer **keyset pagination** (`WHERE id > last_seen_id`) over `OFFSET`. OFFSET gets quadratically slower past ~10k.

### Filtering & sorting

Don't take raw column names from the client unguarded — that's an injection vector for ORM metadata. Whitelist:

```python
SORTABLE_FIELDS = {"id", "created_at", "employee_code", "joined_on"}

def apply_sort(stmt, sort_by: str, sort_dir: str):
    if sort_by not in SORTABLE_FIELDS:
        raise BusinessRuleViolation(f"Cannot sort by {sort_by}")
    col = getattr(Employee, sort_by)
    return stmt.order_by(col.desc() if sort_dir == "desc" else col.asc())
```

### Transactions

A service method is one transaction. Don't sprinkle `commit()` across CRUD methods:

```python
async def transfer_employee(session, *, emp_id: int, new_dept_id: int):
    employee = await employee_crud.get(session, emp_id)
    if not employee:
        raise NotFoundError("Employee", emp_id)

    await employee_crud.update(session, db_obj=employee, obj_in={"department_id": new_dept_id})
    await audit_crud.log(session, action="dept_transfer", target_id=emp_id, meta={"new_dept_id": new_dept_id})

    await session.commit()   # ONE commit for both writes
    await session.refresh(employee)
    return employee
```

If anything raises before `commit()`, `get_session()` calls `rollback()` and both writes are discarded. That's the whole point.

### Bulk operations

```python
# Bulk insert
session.add_all([Attendance(...) for row in rows])
await session.commit()

# Bulk update — use SQL update statement, not Python loop
from sqlmodel import update
stmt = update(Employee).where(Employee.department_id == old_id).values(department_id=new_id)
await session.exec(stmt)
await session.commit()
```

Looping `await crud.update(...)` for 10,000 rows is the #1 cause of "the import endpoint times out".

## Alembic migrations

### Setup

```bash
alembic init alembic
```

Edit `alembic/env.py`:

```python
from app.core.config import get_settings
from app.db.base import SQLModel  # imports all models as side-effect
import app.models  # noqa: F401  - ensures all models are registered

config = context.config
config.set_main_option("sqlalchemy.url", get_settings().DATABASE_URL.replace("+asyncpg", ""))
target_metadata = SQLModel.metadata
```

Alembic itself is sync — use the sync URL (`postgresql://...`, not `postgresql+asyncpg://...`).

### Creating a migration

```bash
alembic revision --autogenerate -m "add leave_balance table"
```

**ALWAYS open the generated file and check:**

- Unintended drops of indexes / constraints — autogenerate sometimes misses them and recreates differently.
- Missing `server_default` for new NOT NULL columns on existing tables (will fail on rows that already exist).
- Type changes that need data migration (e.g. `String(20)` → `String(10)` will truncate or fail).

### Safe column changes

```python
# Adding a NOT NULL column to an existing table — 3-step:
# Migration 1: add as nullable with default
op.add_column("users", sa.Column("phone", sa.String(20), nullable=True))
op.execute("UPDATE users SET phone = '' WHERE phone IS NULL")
# Migration 2 (next deploy): make NOT NULL
op.alter_column("users", "phone", nullable=False)
```

### Destructive operations require confirmation

Before generating a migration with `op.drop_column`, `op.drop_table`, or `op.alter_column(type_=...)` that narrows a type — **stop and ask the user**. State the data loss risk explicitly.

### Migration testing

After every migration:

```bash
alembic upgrade head     # apply
alembic downgrade -1     # roll back
alembic upgrade head     # re-apply — both directions must work
```

If `downgrade` raises, fix the migration before merging.

## Query optimization checklist

When an endpoint feels slow:

1. **Set `echo=True` on the engine.** Count the SQL statements per request. Should be < 5 for most endpoints.
2. **Look for N+1.** Loop printing related fields without `selectinload` / `joinedload` → fix the query.
3. **Check indexes.** `EXPLAIN ANALYZE` your slow query. If you see `Seq Scan` on a large table with a WHERE clause, add an index.
4. **Check `LIMIT` placement.** `select().options(selectinload(...)).limit(10)` is fine; `select().limit(10).options(selectinload(...))` may load all related rows. Always put `.limit()` last.
5. **Connection pool exhaustion.** `pool_size=10, max_overflow=20` = 30 concurrent. Under load, monitor with `SELECT count(*) FROM pg_stat_activity`. Tune up if hitting the ceiling; tune down if connections idle.
6. **Avoid `COUNT(*)` on large tables in pagination.** Cache it, use estimates (`pg_class.reltuples`), or return "load more" cursors without total counts.
7. **Don't return entire rows when you need one column.** `select(User.email)` returns scalars, faster than fetching the whole model.

## Common pitfalls

- **Sync session in async handler** → silent blocking, kills throughput. If you see `from sqlalchemy.orm import Session` next to `async def`, that's a bug.
- **Forgetting `expire_on_commit=False`** → accessing `obj.field` after commit raises `MissingGreenlet` / triggers a new query. Set the flag.
- **Updating without `refresh()`** → return value missing DB-side defaults / server-side timestamps. Call `await session.refresh(obj)` after commit if you return it.
- **Multi-statement transactions across `await`s with external calls** → if you `await http_client.post(...)` mid-transaction, you hold a connection during the network call. Move external calls outside the transaction.
- **`UNIQUE` constraint as validation** → don't catch `IntegrityError` to mean "duplicate". Check existence first in the service, then create. The catch is a safety net, not the primary check.

## When to introduce a read replica / cache

Don't, until you have measured. Single Postgres on modest hardware handles 10k req/s of simple reads. Most slowness is N+1 or missing indexes, not lack of caching.

Reach for Redis when:
- You have a hot read (>100 rps) that's expensive to compute (joins, aggregates).
- You need rate limiting / session blacklist.
- You need a queue (Celery / RQ / arq) — Redis is the broker.

Reach for a read replica when:
- Reads dominate (>10:1 vs writes) AND single primary is CPU-bound.
- Reports / dashboards can tolerate seconds of replication lag.
