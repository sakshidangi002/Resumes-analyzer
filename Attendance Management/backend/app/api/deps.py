"""Dependencies: get_db, get_current_user, role-based access."""
from typing import Generator, List
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyCookie
from sqlalchemy.orm import Session
from jose import JWTError, jwt

from app.db.session import SessionLocal, get_db
from app.core.config import get_settings
from app.core.security import decode_access_token
from app.models import User
from app.models.user import Role
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


def get_current_user(
    db: Session = Depends(get_db_session),
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> User:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
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
