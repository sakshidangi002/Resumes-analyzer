"""User management (Admin/HR)."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.core.security import get_password_hash
from app.models import User
from app.models.user import Role, user_roles
from app.schemas.user import UserCreate, UserUpdate, UserResponse, UserWithRoles
from app.api.deps import get_current_user, require_roles

router = APIRouter()


@router.get("", response_model=list[UserWithRoles])
def list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    users = db.query(User).all()
    return [
        UserWithRoles(
            id=u.id,
            username=u.username,
            official_email=u.official_email,
            is_active=u.is_active,
            employee_id=u.employee_id,
            created_at=u.created_at,
            roles=[r.name for r in u.roles],
        )
        for u in users
    ]


@router.post("", response_model=UserResponse)
def create_user(
    data: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin", "HR"])),
):
    """
    - Admin can create any user/role combination.
    - HR can only create Employee users (role_names must be ['Employee'] or empty / Employee-only).
    """
    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")

    role_names = data.role_names or []
    # Enforce HR restriction: cannot create Admin/HR/Manager users
    if "HR" in [r.name for r in current_user.roles] and not any(r.name == "Admin" for r in current_user.roles):
        forbidden = {"Admin", "HR", "Manager"}
        if any(r in forbidden for r in role_names):
            raise HTTPException(status_code=403, detail="HR can only create Employee logins")

    user = User(
        username=data.username,
        password_hash=get_password_hash(data.password),
        official_email=data.official_email,
        employee_id=data.employee_id,
        is_active=True,
    )
    db.add(user)
    db.flush()
    for role_name in role_names:
        role = db.query(Role).filter(Role.name == role_name).first()
        if role:
            db.execute(user_roles.insert().values(user_id=user.id, role_id=role.id))
    db.commit()
    db.refresh(user)
    return user


@router.get("/{user_id}", response_model=UserWithRoles)
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin"])),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserWithRoles(
        id=user.id,
        username=user.username,
        official_email=user.official_email,
        is_active=user.is_active,
        employee_id=user.employee_id,
        created_at=user.created_at,
        roles=[r.name for r in user.roles],
    )


@router.patch("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    data: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin"])),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if data.password is not None:
        user.password_hash = get_password_hash(data.password)
    if data.official_email is not None:
        user.official_email = data.official_email
    if data.employee_id is not None:
        user.employee_id = data.employee_id
    if data.is_active is not None:
        user.is_active = data.is_active
    if data.role_names is not None:
        db.execute(delete(user_roles).where(user_roles.c.user_id == user.id))
        for role_name in data.role_names:
            role = db.query(Role).filter(Role.name == role_name).first()
            if role:
                db.execute(user_roles.insert().values(user_id=user.id, role_id=role.id))
    db.commit()
    db.refresh(user)
    return user


@router.delete("/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(["Admin"])),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    db.execute(delete(user_roles).where(user_roles.c.user_id == user.id))
    db.delete(user)
    db.commit()
    return {"message": "User deleted"}
