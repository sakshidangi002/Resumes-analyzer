"""Login, signup (bootstrap), and JWT token.
Bearer token: login returns access_token; frontend sends it as Authorization: Bearer <token>.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.core.security import verify_password, get_password_hash, create_access_token
from app.models import User
from app.models.employee import Employee
from app.models.user import Role, user_roles
from app.schemas.auth import LoginRequest, SignupRequest, Token, ForgotPasswordRequest
from app.api.deps import get_current_user
from app.schemas.user import UserWithRoles

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/can-signup")
def can_signup(db: Session = Depends(get_db)):
    """Allow signup only when no users exist (first-time setup)."""
    count = db.query(User).count()
    return {"allowed": count == 0}


@router.post("/signup", response_model=Token)
def signup(data: SignupRequest, db: Session = Depends(get_db)):
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
    access_token = create_access_token(
        subject=user.id,
        extra_claims={"roles": role_names, "employee_id": user.employee_id},
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

    return Token(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        username=user.username,
        roles=role_names,
        employee_id=user.employee_id,
        employee_code=employee_code,
        designation=designation,
    )


@router.post("/login", response_model=Token)
def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data.username).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not user.is_active:
        raise HTTPException(status_code=401, detail="User inactive")
    role_names = [r.name for r in user.roles]
    access_token = create_access_token(
        subject=user.id,
        extra_claims={"roles": role_names, "employee_id": user.employee_id},
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

    return Token(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        username=user.username,
        roles=role_names,
        employee_id=user.employee_id,
        employee_code=employee_code,
        designation=designation,
    )


@router.post("/forgot-password")
def forgot_password(data: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Reset password to a default value."""
    user = db.query(User).filter(User.username == data.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    
    user.password_hash = get_password_hash("Softwiz@123")
    db.commit()
    return {"detail": "Password has been successfully reset to: Softwiz@123"}


@router.get("/me", response_model=UserWithRoles)
def me(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Return the current authenticated user with roles.

    We also *try* to auto-link the user to an Employee by official email so that
    HR/Admin personal views (My Leave, My Payroll, etc.) work without manual linking.

    This auto-linking must never break authentication: any error here should be
    swallowed so that /auth/me continues to work and the frontend does not get a
    500 (which would immediately log the user out).
    """
    if not current_user.employee_id:
        try:
            candidate_email = (current_user.official_email or "").strip() or (current_user.username or "").strip()
            if "@" in candidate_email:
                emp = db.query(Employee).filter(Employee.official_email == candidate_email).first()
                if emp:
                    current_user.employee_id = emp.id
                    db.add(current_user)
                    db.commit()
                    db.refresh(current_user)
        except Exception:
            # If anything goes wrong while trying to auto-link, roll back and continue.
            # It's better to return a valid user without employee_id than to raise 500.
            db.rollback()
    
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
