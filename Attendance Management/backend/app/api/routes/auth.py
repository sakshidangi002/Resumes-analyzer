"""Login, signup (bootstrap), and JWT token.
Bearer token: login returns access_token; frontend sends it as Authorization: Bearer <token>.
"""
import secrets
import string
import os
import threading
import time
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.core.security import verify_password, get_password_hash, create_access_token
from app.models import User
from app.models.employee import Employee
from app.models.user import Role, user_roles
from app.schemas.auth import LoginRequest, SignupRequest, Token, ForgotPasswordRequest, ChangePasswordRequest
from app.api.deps import get_current_user_for_password_change, is_employment_status_blocked, require_roles
from app.core.net import client_ip
from app.services.audit_service import log_audit
from app.schemas.user import UserWithRoles

router = APIRouter(prefix="/auth", tags=["auth"])


def _set_auth_cookie(response: Response, token: str, request: Request | None = None) -> None:
    response.set_cookie(
        "access_token",
        token,
        httponly=True,
        secure=bool(request and request.url.scheme == "https"),
        samesite="lax",
        max_age=60 * 60 * 8,
        path="/",
    )

# Login protection: two independent counters over a 15-minute window.
#
#   account:<username>  -- max 5. Catches guessing against ONE account, no
#                          matter how many source addresses it comes from.
#   ip:<client_ip>      -- max 50. A blunt backstop against one host spraying
#                          many accounts. Deliberately far above the account
#                          limit: a shared office NAT puts every employee behind
#                          ONE address, so a low IP ceiling locks out the whole
#                          office. It must never be the limit that trips first
#                          in normal use.
#
# The account key must NOT include the IP -- an IP-scoped account key can never
# reach its limit before the (lower-scoped) IP key does, which makes it dead
# code and leaves per-account guessing effectively unlimited across addresses.
#
# Process-local on purpose for the current single-worker deployment; move to
# Redis before running multiple workers or instances.
_LOGIN_MAX_ATTEMPTS = int(os.getenv("LOGIN_RATE_LIMIT_MAX", "5"))
_LOGIN_MAX_ATTEMPTS_PER_IP = int(os.getenv("LOGIN_RATE_LIMIT_MAX_PER_IP", "50"))
_LOGIN_WINDOW_SECONDS = int(os.getenv("LOGIN_RATE_LIMIT_WINDOW_SECONDS", "900"))
_LOGIN_FAILURES: dict[str, list[float]] = {}
_LOGIN_RATE_LOCK = threading.Lock()

# Honour X-Forwarded-For ONLY when the deployment actually runs behind a proxy
# we control. Off by default because the header is client-supplied: trusting it
# on a directly-exposed server lets an attacker forge a fresh IP per request and
# walk straight past the IP counter.
#
# Conversely, leaving this OFF *behind* a reverse proxy makes request.client.host
# the proxy's address for every user, so the whole system shares one IP bucket.
# The high per-IP ceiling keeps that from being an outage, but set
# TRUST_PROXY_HEADERS=1 (and make the proxy overwrite, not append, XFF) so the
# backstop tracks real clients.
# Client-address resolution is shared with the middleware limiter in main.py --
# see app/core/net.py. Keeping one implementation is what stops the two
# limiters disagreeing about who the caller is when behind a proxy.
_client_ip = client_ip


def _login_keys(request: Request, username: str) -> tuple[str, str]:
    """Return (account_key, ip_key). Order matters -- see _KEY_LIMITS."""
    normalized_username = (username or "").strip().lower()[:200]
    return f"account:{normalized_username}", f"ip:{_client_ip(request)}"


def _limit_for(key: str) -> int:
    return _LOGIN_MAX_ATTEMPTS_PER_IP if key.startswith("ip:") else _LOGIN_MAX_ATTEMPTS


def _prune_login_failures(now: float) -> None:
    cutoff = now - _LOGIN_WINDOW_SECONDS
    expired = []
    for key, attempts in _LOGIN_FAILURES.items():
        kept = [stamp for stamp in attempts if stamp > cutoff]
        if kept:
            _LOGIN_FAILURES[key] = kept
        else:
            expired.append(key)
    for key in expired:
        _LOGIN_FAILURES.pop(key, None)


def _check_login_rate_limit(keys: tuple[str, str]) -> int | None:
    now = time.monotonic()
    with _LOGIN_RATE_LOCK:
        _prune_login_failures(now)
        # Each key carries its own ceiling (account: strict, IP: loose backstop).
        retry_after = 0
        for key in keys:
            attempts = _LOGIN_FAILURES.get(key, [])
            if len(attempts) >= _limit_for(key):
                retry_after = max(retry_after, int(max(1, _LOGIN_WINDOW_SECONDS - (now - attempts[0]))))
        return retry_after or None


def _record_login_failure(keys: tuple[str, str]) -> None:
    now = time.monotonic()
    with _LOGIN_RATE_LOCK:
        _prune_login_failures(now)
        for key in keys:
            _LOGIN_FAILURES.setdefault(key, []).append(now)


def _clear_login_failures(keys: tuple[str, str]) -> None:
    """Reset the ACCOUNT counter after a genuine login. The IP counter is left
    alone on purpose: clearing it would let an attacker who holds one valid
    credential reset the backstop at will (4 guesses, 1 real login, repeat)."""
    account_key, _ip_key = keys
    with _LOGIN_RATE_LOCK:
        _LOGIN_FAILURES.pop(account_key, None)


def _temporary_password() -> str:
    alphabet = string.ascii_letters + string.digits
    return "Softwiz@" + "".join(secrets.choice(alphabet) for _ in range(10))


def _auto_link_user_to_employee(db: Session, user: User) -> Employee | None:
    """Link an account during login; read endpoints must remain read-only."""
    if user.employee_id:
        return db.query(Employee).filter(Employee.id == user.employee_id).first()
    candidate_email = (user.official_email or "").strip() or (user.username or "").strip()
    if "@" not in candidate_email:
        return None
    emp = db.query(Employee).filter(Employee.official_email == candidate_email).first()
    if not emp:
        return None
    user.employee_id = emp.id
    db.add(user)
    db.commit()
    db.refresh(user)
    return emp


@router.get("/can-signup")
def can_signup(db: Session = Depends(get_db)):
    """Allow signup only when no users exist (first-time setup)."""
    count = db.query(User).count()
    return {"allowed": count == 0}


@router.post("/signup", response_model=Token)
def signup(data: SignupRequest, response: Response, db: Session = Depends(get_db)):
    """Create the first user (Admin) when the database is empty. Otherwise returns 403."""
    if db.query(User).count() > 0:
        raise HTTPException(status_code=403, detail="Signup disabled. Use login or create users as Admin.")
    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(status_code=400, detail="Username already taken.")
    # Ensure Admin role exists
    admin_role = db.query(Role).filter(Role.name == "Admin").first()
    if not admin_role:
        for name in ["Admin", "HR", "Manager", "Employee"]:
            db.add(Role(name=name, description=name))
        db.flush()
        admin_role = db.query(Role).filter(Role.name == "Admin").first()
    user = User(
        username=data.username,
        password_hash=get_password_hash(data.password),
        official_email=data.official_email,
        is_active=True,
    )
    db.add(user)
    db.flush()
    db.execute(user_roles.insert().values(user_id=user.id, role_id=admin_role.id))
    db.commit()
    db.refresh(user)
    role_names = [r.name for r in user.roles]
    log_audit(db, user.id, "SIGNUP", "User", str(user.id), "Bootstrap admin account created")
    access_token = create_access_token(
        subject=user.id,
        extra_claims={
            "roles": role_names,
            "employee_id": user.employee_id,
            "pwd_change": False,
        },
    )
    # Fetch employee data if linked
    employee_code = None
    designation = None
    if user.employee_id:
        emp = db.query(Employee).filter(Employee.id == user.employee_id).first()
        if emp:
            employee_code = emp.employee_code
            if emp.designation:
                designation = emp.designation.title

    token_response = Token(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        username=user.username,
        roles=role_names,
        employee_id=user.employee_id,
        employee_code=employee_code,
        designation=designation,
        must_change_password=False,
    )
    _set_auth_cookie(response, access_token)
    return token_response


@router.post("/login", response_model=Token)
def login(data: LoginRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    keys = _login_keys(request, data.username)
    retry_after = _check_login_rate_limit(keys)
    if retry_after:
        # Log the lockout, not every failed attempt: a row per failure would let
        # an attacker drive unbounded DB writes from unauthenticated requests.
        log_audit(
            db, None, "LOGIN_RATE_LIMITED", "User", None,
            f"Login throttled for '{(data.username or '')[:100]}' from {_client_ip(request)}",
            ip_address=_client_ip(request),
        )
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts. Please try again later.",
            headers={"Retry-After": str(retry_after)},
        )

    user = db.query(User).filter(User.username == data.username).first()
    if not user or not verify_password(data.password, user.password_hash):
        _record_login_failure(keys)
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not user.is_active:
        _record_login_failure(keys)
        raise HTTPException(status_code=401, detail="User inactive")

    try:
        _auto_link_user_to_employee(db, user)
    except Exception:
        # Linking is a convenience and must never prevent a valid login.
        db.rollback()

    # Fetch employee data if linked
    employee_code = None
    designation = None
    emp = None
    if user.employee_id:
        emp = db.query(Employee).filter(Employee.id == user.employee_id).first()
        if emp:
            # Resigned / Terminated employees cannot log in.
            if is_employment_status_blocked(emp.employment_status):
                _record_login_failure(keys)
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Access revoked: this account is linked to an employee "
                        f"marked '{emp.employment_status}'. Contact HR if this is "
                        f"a mistake."
                    ),
                )
            employee_code = emp.employee_code
            if emp.designation:
                designation = emp.designation.title

    role_names = [r.name for r in user.roles]
    _clear_login_failures(keys)
    access_token = create_access_token(
        subject=user.id,
        extra_claims={
            "roles": role_names,
            "employee_id": user.employee_id,
            # Carried in the token so the Resume Analyzer gate in main.py can
            # enforce the forced password change without a DB round-trip on
            # every /resume* request. Re-issued by /auth/change-password.
            "pwd_change": bool(user.must_change_password),
        },
    )

    token_response = Token(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        username=user.username,
        roles=role_names,
        employee_id=user.employee_id,
        employee_code=employee_code,
        designation=designation,
        must_change_password=bool(user.must_change_password),
    )
    _set_auth_cookie(response, access_token, request)
    log_audit(db, user.id, "LOGIN_SUCCESS", "User", str(user.id), "Successful login", _client_ip(request))
    return token_response


@router.post("/logout", status_code=204)
def logout(response: Response):
    response.delete_cookie("access_token", path="/")
    return Response(status_code=204)


@router.post("/forgot-password")
def forgot_password(
    data: ForgotPasswordRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin"])),
):
    """Generate a one-time temporary password. ADMIN ONLY.

    This was previously unauthenticated, which let anyone reset any account --
    including Admin -- to a constant password that the response disclosed. It is
    now an administrative action: an Admin resets the password and tells the
    person, which is how it was already being used in practice.

    A self-service reset must not live here; it needs an emailed one-time token.
    """
    user = db.query(User).filter(User.username == data.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    temporary_password = _temporary_password()
    user.password_hash = get_password_hash(temporary_password)
    user.must_change_password = True
    db.commit()
    log_audit(db, current_user.id, "PASSWORD_RESET", "User", str(user.id), f"Temporary password generated for {user.username}")
    return {
        "detail": "Temporary password generated. The user must change it after login.",
        "temporary_password": temporary_password,
    }


@router.post("/change-password")
def change_password(
    data: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_password_change),
    response: Response = None,
):
    if len(data.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters.")
    if data.current_password == data.new_password:
        raise HTTPException(status_code=400, detail="New password must be different from the temporary password.")

    # Re-load the user through THIS request's session.
    #
    # `current_user` belongs to a DIFFERENT Session: it is resolved via
    # get_current_user_for_password_change -> get_db_session, while `db` comes
    # from get_db. FastAPI caches dependencies per callable, and those are two
    # distinct callables, so they are two distinct sessions. Assigning to
    # `current_user` and then calling `db.commit()` commits a session with no
    # pending changes -- the write was silently dropped and the user kept their
    # temporary password despite a "success" response.
    user = db.query(User).filter(User.id == current_user.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if not verify_password(data.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    user.password_hash = get_password_hash(data.new_password)
    user.must_change_password = False
    db.commit()
    log_audit(db, user.id, "PASSWORD_CHANGED", "User", str(user.id), "Password changed")

    # Issue a replacement token. The old one still carries pwd_change=true, and
    # the Resume Analyzer gate reads that claim -- without a swap the user would
    # stay locked out of /resume* until their next login.
    role_names = [r.name for r in user.roles]
    access_token = create_access_token(
        subject=user.id,
        extra_claims={
            "roles": role_names,
            "employee_id": user.employee_id,
            "pwd_change": False,
        },
    )
    result = {
        "detail": "Password changed successfully.",
        "access_token": access_token,
        "token_type": "bearer",
    }
    if response is not None:
        _set_auth_cookie(response, access_token)
    return result


@router.get("/me", response_model=UserWithRoles)
def me(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_password_change),
):
    """Return the current authenticated user with roles (read-only)."""
    # Fetch employee data if linked
    employee_code = None
    designation = None
    if current_user.employee_id:
        emp = db.query(Employee).filter(Employee.id == current_user.employee_id).first()
        if emp:
            employee_code = emp.employee_code
            if emp.designation:
                designation = emp.designation.title

    return UserWithRoles(
        id=current_user.id,
        username=current_user.username,
        official_email=current_user.official_email,
        is_active=current_user.is_active,
        employee_id=current_user.employee_id,
        employee_code=employee_code,
        designation=designation,
        created_at=current_user.created_at,
        roles=[r.name for r in current_user.roles],
    )
