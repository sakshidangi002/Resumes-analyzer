"""
Seed default data: roles, admin user, company config, current financial year, leave types.
Run from backend dir: python -m scripts.seed
Requires DATABASE_URL / .env and tables created (alembic upgrade head).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datetime import date
from app.db.session import SessionLocal
from app.models import Role, User, CompanyConfig, FinancialYear, LeaveType
from app.core.security import get_password_hash


def seed():
    db = SessionLocal()
    try:
        # Roles
        for name in ["Admin", "HR", "Manager", "Employee"]:
            if db.query(Role).filter(Role.name == name).first():
                continue
            db.add(Role(name=name, description=name))
        db.commit()

        # Admin user (if not exists)
        if not db.query(User).filter(User.username == "admin").first():
            admin = User(
                username="admin",
                password_hash=get_password_hash("admin123"),
                official_email="admin@company.com",
                is_active=True,
            )
            db.add(admin)
            db.flush()
            admin_role = db.query(Role).filter(Role.name == "Admin").first()
            if admin_role:
                from app.models.user import user_roles
            db.execute(user_roles.insert().values(user_id=admin.id, role_id=admin_role.id))
        db.commit()

        # Company config
        if not db.query(CompanyConfig).first():
            db.add(CompanyConfig(
                name="Default Company",
                default_working_days_per_month=30,
                weekly_off_days="SAT,SUN",
                grace_time_minutes=15,
                half_day_threshold_hours=4,
            ))
        db.commit()

        # Current financial year (e.g. 2024-25)
        start = date(2024, 4, 1)
        end = date(2025, 3, 31)
        if not db.query(FinancialYear).filter(FinancialYear.start_date == start).first():
            db.add(FinancialYear(start_date=start, end_date=end, is_current=True, name="2024-25"))
        db.commit()

        # Leave types
        for code, name, paid, half in [("PL", "Paid Leave", True, True), ("UL", "Unpaid Leave (LOP)", False, True), ("HD", "Half Day", True, True)]:
            if db.query(LeaveType).filter(LeaveType.code == code).first():
                continue
            db.add(LeaveType(code=code, name=name, is_paid=paid, allow_half_day=half))
        db.commit()
        print("Seed completed.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
