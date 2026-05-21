"""Create in-app notifications for users."""
from sqlalchemy.orm import Session
from app.models import User, AppNotification
from app.models.user import Role, user_roles


def create_notification(
    db: Session,
    user_id: int,
    title: str,
    body: str | None = None,
    kind: str = "GENERAL",
    link_path: str | None = None,
) -> AppNotification:
    n = AppNotification(
        user_id=user_id,
        title=title,
        body=body,
        kind=kind,
        link_path=link_path,
    )
    db.add(n)
    db.commit()
    db.refresh(n)
    return n


def notify_user_for_employee(
    db: Session,
    employee_id: int,
    title: str,
    body: str | None = None,
    kind: str = "GENERAL",
    link_path: str | None = None,
) -> None:
    u = db.query(User).filter(User.employee_id == employee_id).first()
    if not u:
        return
    create_notification(db, u.id, title, body, kind, link_path)


def notify_users_with_roles(
    db: Session,
    role_names: list[str],
    title: str,
    body: str | None = None,
    kind: str = "GENERAL",
    link_path: str | None = None,
) -> None:
    q = (
        db.query(User)
        .join(user_roles, User.id == user_roles.c.user_id)
        .join(Role, Role.id == user_roles.c.role_id)
        .filter(Role.name.in_(role_names))
        .distinct()
    )
    users = q.all()
    if not users:
        return
    for u in users:
        db.add(
            AppNotification(
                user_id=u.id,
                title=title,
                body=body,
                kind=kind,
                link_path=link_path,
            )
        )
    db.commit()
