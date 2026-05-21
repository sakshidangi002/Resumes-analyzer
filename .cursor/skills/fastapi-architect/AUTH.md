# Authentication, Authorization & Security

JWT auth, RBAC, password hashing, file upload security, and the everyday security best practices for FastAPI backends.

## Password hashing — pick ONE library

Two options, mutually exclusive. **Do not install both** (will conflict with each other and with some transitive deps).

### Option A — Native `bcrypt` (recommended for new projects)

```python
# app/core/security.py
import bcrypt

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())
```

`requirements.txt`:
```
bcrypt>=4.1.0,<5
```

### Option B — `passlib[bcrypt]` (legacy projects)

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)
```

Pin `passlib==1.7.4` and `bcrypt==4.0.1` — newer bcrypt removes `__about__` and breaks passlib.

**Rule:** never mix. If you see both in `requirements.txt`, that's a bug — remove one.

## JWT — encode, decode, refresh

```python
# app/core/security.py
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from app.core.config import get_settings

settings = get_settings()

def _create_token(subject: str, expires_delta: timedelta, token_type: str, extra: dict | None = None) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(subject),
        "iat": now,
        "exp": now + expires_delta,
        "type": token_type,
        **(extra or {}),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def create_access_token(user_id: int, role: str) -> str:
    return _create_token(
        subject=user_id,
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        token_type="access",
        extra={"role": role},
    )

def create_refresh_token(user_id: int) -> str:
    return _create_token(
        subject=user_id,
        expires_delta=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        token_type="refresh",
    )

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError as e:
        raise PermissionDeniedError(f"Invalid token: {e}")
```

**Always include:**
- `sub` — the user id (as string, per RFC 7519).
- `iat`, `exp` — issue + expiry.
- `type` — `"access"` or `"refresh"`. Refresh tokens must NOT be accepted at protected endpoints; access tokens must NOT be accepted at `/auth/refresh`. Validate this in dependencies.
- `role` — included for fast RBAC check without DB lookup (with the trade-off that role changes require new login or short token TTL).

**Never include:** password hash, full user object, email (PII), or any data > 1KB total. Tokens are sent on every request.

**Algorithm:** `HS256` is fine for monoliths. Use `RS256` if multiple services verify tokens (one signs, others verify with public key — share key safely).

### Refresh flow

```
POST /auth/login        → access + refresh (refresh stored httpOnly cookie OR returned for SPA)
GET  /protected         → Authorization: Bearer <access>
POST /auth/refresh      → body: refresh token → new access (+ optionally new refresh, rotated)
POST /auth/logout       → invalidate refresh token (DB or Redis blacklist)
```

**Token rotation:** every refresh issues a new refresh token AND invalidates the old one. If an old refresh token is reused (replay attack), invalidate the user's entire session.

## Auth dependencies

```python
# app/api/deps.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_session),
) -> User:
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong token type")
    user_id = int(payload["sub"])
    user = await user_crud.get(session, user_id)
    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found or inactive")
    return user

def require_role(*allowed: str):
    async def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Requires one of: {', '.join(allowed)}",
            )
        return user
    return checker
```

### RBAC roles

Standard set: `admin`, `hr`, `manager`, `employee`. Stored as a single column on the user (cheap, simple). When permissions get complex (>5 roles, per-resource perms), switch to:
- `Role` table + `UserRole` join (many-to-many),
- `Permission` table + `RolePermission` join,
- Caching the user's effective permissions in the JWT or Redis.

But — **do not start there**. 90% of HR / attendance / resume systems are fine with a single role column.

### Usage in routers

```python
# Public
@router.post("/auth/login")
async def login(...): ...

# Authenticated, any role
@router.get("/me", response_model=UserRead)
async def me(user: User = Depends(get_current_user)): return user

# Admin or HR only
@router.post("/employees", response_model=EmployeeRead, status_code=201)
async def create_employee(
    payload: EmployeeCreate,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(require_role("admin", "hr")),
): ...

# Resource-owner check
@router.get("/payslips/{payslip_id}", response_model=PayslipRead)
async def get_payslip(
    payslip_id: int,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
):
    payslip = await payslip_crud.get(session, payslip_id)
    if not payslip:
        raise NotFoundError("Payslip", payslip_id)
    if user.role not in ("admin", "hr") and payslip.employee.user_id != user.id:
        raise PermissionDeniedError()
    return payslip
```

**Resource-owner checks live in the service or route**, not in a generic dependency — they need the loaded resource to compare.

## CORS

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # explicit list, NEVER ["*"] with credentials
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
```

`allow_origins=["*"]` with `allow_credentials=True` is silently ignored by browsers and is a common cause of "CORS works in curl but not in Chrome".

## Rate limiting

Don't roll your own. Use `slowapi` (Redis or in-memory):

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@router.post("/auth/login")
@limiter.limit("5/minute")
async def login(request: Request, ...): ...
```

**Always rate-limit:** `/auth/login`, `/auth/forgot-password`, `/auth/refresh`. Failed attempts are how credential-stuffing works.

## File upload security

```python
# app/api/v1/uploads.py
import uuid
from pathlib import Path
from fastapi import UploadFile, File, HTTPException

ALLOWED_MIME = {"application/pdf", "image/jpeg", "image/png"}
ALLOWED_EXT = {".pdf", ".jpg", ".jpeg", ".png"}
MAX_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

@router.post("/uploads/resume", response_model=UploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(415, f"Unsupported type: {file.content_type}")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"Unsupported extension: {ext}")

    contents = await file.read()
    if len(contents) > MAX_SIZE_BYTES:
        raise HTTPException(413, "File too large")
    if not _looks_like_pdf(contents) and ext == ".pdf":
        raise HTTPException(415, "Not a real PDF")

    safe_name = f"{uuid.uuid4()}{ext}"
    dest = Path(settings.UPLOAD_DIR) / safe_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(contents)
    return UploadResponse(stored_name=safe_name, size=len(contents))

def _looks_like_pdf(b: bytes) -> bool:
    return b.startswith(b"%PDF-")
```

**Always:**
1. Validate `content_type` AND extension AND magic bytes (defense in depth).
2. Enforce size limit (also at reverse proxy: nginx `client_max_body_size`).
3. UUID-prefix the stored name; **never** trust `file.filename` for the filesystem.
4. Store under a path the web server CANNOT serve directly as PHP/scripts. Postgres or S3 is safer than `/var/www/uploads`.
5. For PDFs: scan with a virus engine in production (ClamAV) if the file is shared with other users.

**Never:**
- Trust client-supplied `content_type` alone — set by JS, can lie.
- Use `os.path.join(UPLOAD_DIR, file.filename)` — path traversal (`../../etc/passwd`).
- Serve uploads from a URL that includes the user's id without auth check.

## Other security best practices

### Input validation = Pydantic, not regex sprinkled in handlers

```python
from pydantic import EmailStr, Field, field_validator

class UserCreate(SQLModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=1, max_length=100)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isdigit() for c in v) or not any(c.isalpha() for c in v):
            raise ValueError("Password must contain letters and digits")
        return v
```

### SQL injection — non-issue if you use SQLModel correctly

```python
# Safe — parameterized
stmt = select(User).where(User.email == email)

# UNSAFE — DON'T
stmt = text(f"SELECT * FROM users WHERE email = '{email}'")
```

If you must use raw SQL, use bound parameters: `text("SELECT * FROM users WHERE email = :email").bindparams(email=email)`.

### Secrets in environment, never in code

```bash
# .env (gitignored)
SECRET_KEY=<openssl rand -hex 32>
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
SMTP_PASSWORD=...
```

```bash
# .env.example (committed)
SECRET_KEY=
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/myapp
SMTP_PASSWORD=
```

In production, use the cloud provider's secrets manager (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault). Inject via env var on startup.

### Security headers (production)

```python
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if settings.ENVIRONMENT == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

### What you DO NOT need (for most projects)

- API keys + JWTs together. Pick one per use case (JWT for users, API key for service-to-service).
- Encrypting fields at rest in the DB — Postgres TDE or disk encryption is enough for 99% of compliance.
- Custom CSRF tokens if you use JWT in `Authorization` header (CSRF is a cookie attack; pure header tokens don't have it).

## Login endpoint — reference implementation

```python
# app/api/v1/auth.py
@router.post("/auth/login", response_model=TokenPair)
async def login(
    payload: LoginRequest,
    session: AsyncSession = Depends(get_session),
):
    user = await user_crud.get_by_email(session, payload.email)
    if not user or not verify_password(payload.password, user.hashed_password):
        # Same error message + same timing for "no user" vs "bad password"
        raise HTTPException(401, "Invalid credentials")
    if not user.is_active:
        raise HTTPException(403, "Account disabled")

    return TokenPair(
        access_token=create_access_token(user.id, user.role),
        refresh_token=create_refresh_token(user.id),
        token_type="bearer",
    )
```

**Note:** identical error for "user not found" and "wrong password" to prevent username enumeration. Constant-time string compare is built into bcrypt's `checkpw`.
