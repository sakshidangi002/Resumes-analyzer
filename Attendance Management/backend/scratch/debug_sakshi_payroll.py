
import os
from datetime import date
from sqlalchemy import create_url
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import AttendanceRecord, Employee, LeaveRequest, LeaveType
from app.db.base_class import Base

# Load .env manually if needed or assume environment is set
# For this script, we'll construct the URL from the .env I saw earlier
# POSTGRES_USER=postgres
# POSTGRES_PASSWORD=d5x8JGZ%40CH5td7X
# POSTGRES_HOST=43.205.127.72
# POSTGRES_PORT=5432
# POSTGRES_DB=Attendance_system

url = "postgresql://postgres:d5x8JGZ%40CH5td7X@43.205.127.72:5432/Attendance_system"
engine = create_engine(url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

emp = db.query(Employee).filter(Employee.first_name.like("Sakshi%")).first()
if not emp:
    print("Employee not found")
else:
    print(f"Employee: {emp.id} - {emp.first_name} {emp.last_name}")
    records = db.query(AttendanceRecord).filter(
        AttendanceRecord.employee_id == emp.id,
        AttendanceRecord.date >= date(2026, 3, 1),
        AttendanceRecord.date <= date(2026, 3, 31)
    ).order_by(AttendanceRecord.date).all()
    
    print("Attendance Records (March 2026):")
    for r in records:
        print(f"  {r.date}: {r.status} (In: {r.sign_in_time}, Out: {r.sign_out_time})")
        
    leaves = db.query(LeaveRequest, LeaveType).join(LeaveType).filter(
        LeaveRequest.employee_id == emp.id,
        LeaveRequest.status == "APPROVED",
        LeaveRequest.start_date <= date(2026, 3, 31),
        LeaveRequest.end_date >= date(2026, 3, 1)
    ).all()
    
    print("\nApproved Leaves (March 2026):")
    for req, lt in leaves:
        print(f"  {req.start_date} to {req.end_date}: {lt.name} (Paid: {lt.is_paid})")

db.close()
