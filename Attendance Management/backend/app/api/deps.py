"""Dependencies: get_db, get_current_user, role-based access."""
from typing import Generator, List
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.core.security import decode_access_token, decode_media_token
from app.models import User
from app.models.employee import Employee, EmploymentStatus

security = HTTPBearer(auto_error=False)


# Statuses that revoke ALL access (login, API calls, websockets, etc.).
# Admin / HR can still manage these records from their own (active) account.
BLOCKED_EMPLOYMENT_STATUSES = {
    EmploymentStatus.RESIGNED.value,
    EmploymentStatus.TERMINATED.value,
}


def is_employment_status_blocked(value: str | None) -> bool:
    return bool(value) and value in BLOCKED_EMPLOYMENT_STATUSES


def get_db_session() -> Generator[Session, None, None]:
    yield from get_db()


def _get_current_user(
    db: Session = Depends(get_db_session),
    request: Request = None,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> User:
    token = credentials.credentials if credentials else ((request.cookies.get("access_token", "") if request else ""))
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=401, detail="User inactive")

    # Resigned / Terminated employees lose access immediately, even if their
    # User row is still marked active and a JWT was issued earlier.
    if user.employee_id:
        emp_status = (
            db.query(Employee.employment_status)
            .filter(Employee.id == user.employee_id)
            .scalar()
        )
        if is_employment_status_blocked(emp_status):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access revoked: employee is {emp_status}.",
            )
    return user


def get_current_user(
    db: Session = Depends(get_db_session),
    request: Request = None,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> User:
    user = _get_current_user(db=db, request=request, credentials=credentials)
    if user.must_change_password:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Password change required before accessing HRMS.",
            headers={"X-Password-Change-Required": "true"},
        )
    return user


def get_current_user_for_password_change(
    db: Session = Depends(get_db_session),
    request: Request = None,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> User:
    """Authenticated user dependency that remains available during forced change."""
    return _get_current_user(db=db, request=request, credentials=credentials)


CAMERA_MEDIA_ROLES = {"Admin", "HR"}


def require_media_access(request: Request) -> dict:
    """Authenticate a camera media request via `?t=<media token>`.

    Used by the MJPEG stream and JPEG preview endpoints, which browsers load
    through <img src="..."> and therefore cannot authenticate with a header.
    Mint the token from POST /api/cameras/media-token.
    """
    token = request.query_params.get("t", "").strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing media token. Request one from /api/cameras/media-token.",
        )
    payload = decode_media_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired media token",
        )
    roles = {str(r) for r in (payload.get("roles") or [])}
    if not (roles & CAMERA_MEDIA_ROLES):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )
    return payload


def require_media_or_bearer(
    request: Request,
    db: Session = Depends(get_db_session),
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    """Accept EITHER a normal Bearer session OR a `?t=` media token.

    Endpoints such as /cameras/{id}/preview are fetched both by regular API
    callers (header) and by <img> tags (query token).
    """
    if credentials:
        # Keyword arguments, NOT positional. `get_current_user` takes
        # (db, request, credentials) — passing credentials positionally binds it
        # to `request`, leaves `credentials` holding its unresolved Depends()
        # default, and the first attribute access blows up with a 500. Calling
        # a dependency as a plain function means no defaults get resolved, so
        # every argument it actually needs has to be supplied by name.
        user = get_current_user(db=db, request=request, credentials=credentials)
        role_names = {r.name for r in user.roles}
        if not (role_names & CAMERA_MEDIA_ROLES):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user
    return require_media_access(request)


def require_roles(allowed_roles: List[str]):
    """Dependency factory: require current user to have one of the given roles."""
    def role_check(current_user: User = Depends(get_current_user)) -> User:
        role_names = [r.name for r in current_user.roles]
        if not any(r in role_names for r in allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user
    return role_check


# Shortcuts for common role checks
RequireAdmin = Depends(require_roles(["Admin"]))
RequireHR = Depends(require_roles(["Admin", "HR"]))
RequireManager = Depends(require_roles(["Admin", "HR", "Manager"]))
RequireEmployee = Depends(require_roles(["Admin", "HR", "Manager", "Employee"]))
