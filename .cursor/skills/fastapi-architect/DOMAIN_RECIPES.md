# Domain Recipes

Concrete patterns for the recurring domains in HRMS / resume / employee management systems. Each recipe is a starting structure — models, key business rules, endpoints. Adapt to the specific project's needs.

## Attendance system

### Models

```python
class Attendance(SQLModel, table=True):
    __tablename__ = "attendances"
    id: int | None = Field(default=None, primary_key=True)
    employee_id: int = Field(foreign_key="employees.id", index=True)
    date: date = Field(index=True)
    check_in: datetime | None = None
    check_out: datetime | None = None
    status: AttendanceStatus = Field(default=AttendanceStatus.present, index=True)
    notes: str | None = Field(default=None, max_length=500)

    __table_args__ = (UniqueConstraint("employee_id", "date", name="uq_attendance_emp_date"),)
```

### Key rules
- **One row per (employee, date)** — enforced by unique constraint, not application logic.
- Check-out must be after check-in.
- Future-dated attendance is rejected.
- HR / admin can mark on behalf of an employee with `marked_by_user_id`.

### Endpoints
```
POST /api/v1/attendance/check-in              # employee self-service
POST /api/v1/attendance/check-out
GET  /api/v1/attendance/me?from=...&to=...   # own history
GET  /api/v1/attendance/employees/{id}       # HR/admin/manager view
POST /api/v1/attendance/bulk                  # HR bulk upload
GET  /api/v1/attendance/reports/monthly      # aggregated
```

### Common pitfall
Don't store `check_in` as a date — store as `datetime` UTC. The "attendance date" is a separate column derived at check-in time (in the employee's timezone).

---

## Leave management

### Models

```python
class LeaveType(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, max_length=50)   # "casual", "sick", "earned"
    days_per_year: int

class LeaveBalance(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    employee_id: int = Field(foreign_key="employees.id", index=True)
    leave_type_id: int = Field(foreign_key="leavetypes.id")
    year: int = Field(index=True)
    used: int = Field(default=0)
    __table_args__ = (UniqueConstraint("employee_id", "leave_type_id", "year"),)

class Leave(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    employee_id: int = Field(foreign_key="employees.id", index=True)
    leave_type_id: int = Field(foreign_key="leavetypes.id")
    start_date: date
    end_date: date
    status: LeaveStatus = Field(default=LeaveStatus.pending, index=True)
    reason: str = Field(max_length=500)
    approver_id: int | None = Field(default=None, foreign_key="users.id")
    approved_at: datetime | None = None
```

### Key rules
- Balance check is in the service, not the schema (needs DB read).
- Approving a leave is a separate endpoint — does NOT happen on create.
- Cancellation BEFORE start date refunds the balance; cancellation AFTER doesn't.
- Overlapping leaves rejected.
- Approver must not be the same as the requester.

### Endpoints
```
GET  /api/v1/leaves/balance/me
POST /api/v1/leaves                          # apply (employee)
GET  /api/v1/leaves/me
GET  /api/v1/leaves/pending                  # manager queue
POST /api/v1/leaves/{id}/approve             # manager
POST /api/v1/leaves/{id}/reject              # manager
POST /api/v1/leaves/{id}/cancel              # employee
```

---

## Employee management

### Models

```python
class Department(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, max_length=100)
    head_id: int | None = Field(default=None, foreign_key="users.id")

class Employee(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", unique=True)
    employee_code: str = Field(unique=True, index=True, max_length=20)
    department_id: int | None = Field(default=None, foreign_key="departments.id", index=True)
    manager_id: int | None = Field(default=None, foreign_key="employees.id")  # self-ref
    designation: str = Field(max_length=100)
    joined_on: date
    is_active: bool = Field(default=True, index=True)
    # ... contact, address, emergency contact, etc.
```

### Key rules
- Employee is a 1:1 with User. Creating an employee also creates a User (in one transaction) OR links to an existing pending invite.
- Soft delete via `is_active=False`. Never hard delete — payroll records, attendance, audit chains reference these rows.
- `manager_id` enables org chart traversal — write a CTE query for "all reports under manager X".
- Self-referencing FK: `manager_id` references same table.

### Endpoints
```
GET    /api/v1/employees                # paginated, filterable by dept/role/status
POST   /api/v1/employees                # admin/hr only
GET    /api/v1/employees/{id}
PATCH  /api/v1/employees/{id}
DELETE /api/v1/employees/{id}           # soft delete only
GET    /api/v1/employees/{id}/reports   # direct reports
GET    /api/v1/employees/me             # current user
PATCH  /api/v1/employees/me             # limited field set (phone, address, not role/salary)
```

### Common pitfall
Don't let employees PATCH their own `salary`, `role`, `department_id`, `is_active`. Have a separate `EmployeeSelfUpdate` schema with only the safe fields. NEVER share schemas between admin-write and self-write.

---

## Resume analyzer APIs

### Models

```python
class Resume(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    candidate_name: str = Field(max_length=200)
    email: str = Field(index=True, max_length=255)
    phone: str | None = Field(default=None, max_length=20)
    file_path: str = Field(max_length=500)  # UUID-prefixed; see AUTH.md upload section
    file_hash: str = Field(unique=True, index=True, max_length=64)  # sha256 dedup
    parsed_text: str | None = None
    skills: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    experience_years: float | None = None
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    uploaded_by_id: int = Field(foreign_key="users.id")
    status: ResumeStatus = Field(default=ResumeStatus.uploaded, index=True)
```

### Key flow
1. **Upload** — validate file (see AUTH.md), compute hash, dedupe, store, create row with `status=uploaded`.
2. **Parse** — background task (FastAPI `BackgroundTasks` for light parsing, Celery/arq for heavy). Updates `parsed_text`, `skills`, `experience_years`, `status=parsed`.
3. **Match** — when given a job description, rank resumes by skill overlap + semantic similarity. Returns top-K with scores.

### Endpoints
```
POST /api/v1/resumes/upload                  # multipart/form-data
GET  /api/v1/resumes                         # admin/hr, paginated, filter by skills
GET  /api/v1/resumes/{id}
GET  /api/v1/resumes/{id}/download           # serve file with auth check
POST /api/v1/resumes/match                   # { job_description, top_k } → ranked list
DELETE /api/v1/resumes/{id}                  # soft delete
```

### Critical security
- **NEVER** serve `/uploads/<filename>` directly via static files — that bypasses auth. Route through an endpoint that checks the JWT.
- Dedupe by content hash, not filename. Same resume uploaded twice with different filenames → same row.
- If using LLM / embeddings, NEVER send resume content to a third-party API without explicit consent flag on the candidate record.

---

## File uploads (general)

See [AUTH.md](AUTH.md) § File upload security for the full pattern. Recap:

1. Validate `content_type` + extension + magic bytes + size.
2. UUID-prefix stored name. Never trust `file.filename` for the filesystem.
3. Auth-gated serving — `/uploads/{id}` checks ownership before streaming.
4. Heavy files (>5MB) — stream rather than `await file.read()` all at once.
5. Production: scan with ClamAV before letting the file be downloaded by others.

### Streaming download

```python
from fastapi.responses import StreamingResponse

@router.get("/uploads/{file_id}")
async def download(file_id: int, user: User = Depends(get_current_user)):
    record = await upload_crud.get(session, file_id)
    if not record or not _user_can_access(user, record):
        raise NotFoundError("Upload", file_id)

    def iterfile():
        with open(record.file_path, "rb") as f:
            yield from iter(lambda: f.read(8192), b"")

    return StreamingResponse(
        iterfile(),
        media_type=record.content_type,
        headers={"Content-Disposition": f'attachment; filename="{record.original_name}"'},
    )
```

---

## Email integration

### Setup

```python
# app/services/email_service.py
import smtplib
from email.message import EmailMessage
from fastapi.concurrency import run_in_threadpool

class EmailService:
    def __init__(self, settings: Settings):
        self.s = settings

    async def send(self, *, to: str, subject: str, body: str, html: str | None = None):
        if not self.s.SMTP_HOST:
            logger.warning("SMTP not configured; would send to %s: %s", to, subject)
            return
        await run_in_threadpool(self._send_sync, to, subject, body, html)

    def _send_sync(self, to: str, subject: str, body: str, html: str | None):
        msg = EmailMessage()
        msg["From"] = self.s.SMTP_FROM
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        if html:
            msg.add_alternative(html, subtype="html")
        with smtplib.SMTP(self.s.SMTP_HOST, self.s.SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(self.s.SMTP_USER, self.s.SMTP_PASSWORD)
            smtp.send_message(msg)
```

### Key rules
- **Never send email from the request thread synchronously** — use `run_in_threadpool` for simple cases, Celery/arq for high volume.
- **Use templates** — Jinja2 for HTML, separate file per email type. Don't concatenate strings in service code.
- **Idempotency** — log every send with a unique key (e.g. `"password_reset_user_42"`). Don't double-send.
- **Mock in tests** — never hit real SMTP in CI.

---

## Dashboard / admin analytics APIs

### Patterns

Dashboard endpoints return **aggregated** data, not lists. Three common shapes:

```python
class DashboardKPI(BaseModel):
    label: str
    value: int | float
    trend: float | None = None     # % change vs previous period
    icon: str | None = None

class TimeSeriesPoint(BaseModel):
    timestamp: datetime
    value: float

class DashboardOverview(BaseModel):
    kpis: list[DashboardKPI]
    series: dict[str, list[TimeSeriesPoint]]
    last_updated: datetime
```

### Endpoints

```
GET /api/v1/dashboard/overview               # all KPIs in one response
GET /api/v1/dashboard/attendance/weekly
GET /api/v1/dashboard/leaves/by-department
GET /api/v1/dashboard/employees/headcount-trend
```

### Performance rules
- **Aggregate in SQL**, not in Python. `SELECT COUNT(*), department_id FROM employees GROUP BY department_id`, not `for emp in all_employees: counts[emp.dept] += 1`.
- **Cache hot dashboards** with TTL (Redis or in-memory `cachetools`). 60s TTL is invisible to users but slashes load.
- **Date-range parameters** — accept `from` / `to` (ISO 8601). Default to last 30 days. Cap at 1 year to prevent giant scans.
- **One endpoint per widget** is fine, but if a dashboard has > 6 widgets, build a single `/overview` endpoint that returns them all — saves N round trips.

### Permissions
- Admin / HR see all departments.
- Manager sees their department(s) only.
- Employee sees only their own data (rarely the case for full dashboards).

Bake this into the service via `_filter_by_visibility(user)`, not into each query manually.

---

## Audit logging (for HR / employee data)

Most HRMS systems need an audit trail (who changed what, when):

```python
class AuditLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    actor_id: int = Field(foreign_key="users.id", index=True)
    action: str = Field(max_length=50, index=True)   # "employee.update", "leave.approve"
    target_table: str = Field(max_length=50)
    target_id: int = Field(index=True)
    diff: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
```

Write logs in the **service layer**, in the same transaction as the mutation. If the mutation rolls back, the log rolls back too — no orphan records.

```python
async def update_employee(session, *, actor: User, employee_id: int, update: EmployeeUpdate):
    emp = await employee_crud.get(session, employee_id)
    old = emp.model_dump()
    updated = await employee_crud.update(session, db_obj=emp, obj_in=update)
    new = updated.model_dump()
    diff = {k: {"old": old[k], "new": new[k]} for k in new if old.get(k) != new.get(k)}
    await audit_crud.create(session, AuditLog(actor_id=actor.id, action="employee.update", target_table="employees", target_id=employee_id, diff=diff))
    await session.commit()
    return updated
```

Audit log is **append-only**. Never UPDATE or DELETE rows. If a record needs to be "corrected", append a new compensating entry.
