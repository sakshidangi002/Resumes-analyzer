
import os
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import LeaveRequest, LeaveType, Employee, AttendanceRecord

url = "postgresql://postgres:d5x8JGZ%40CH5td7X@43.205.127.72:5432/Attendance_system"
engine = create_engine(url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

target_date = date(2026, 3, 30)
emp = db.query(Employee).filter(Employee.first_name.like("Sakshi%")).first()

if not emp:
    print("Employee not found")
else:
    print(f"Checking records for {emp.first_name} {emp.last_name} (ID: {emp.id}) on {target_date}")
    
    # 1. Check Attendance Record
    att = db.query(AttendanceRecord).filter(
        AttendanceRecord.employee_id == emp.id,
        AttendanceRecord.date == target_date
    ).first()
    if att:
        print(f"Attendance Record: Status={att.status}, In={att.sign_in_time}, Out={att.sign_out_time}")
    else:
        print("No Attendance Record found for this date.")
        
    # 2. Check Leave Requests for this date
    leaves = db.query(LeaveRequest, LeaveType).join(LeaveType).filter(
        LeaveRequest.employee_id == emp.id,
        LeaveRequest.start_date <= target_date,
        LeaveRequest.end_date >= target_date
    ).all()
    
    if not leaves:
        print("No Leave Request found for this date.")
    else:
        for req, lt in leaves:
            print(f"Leave Request: ID={req.id}, Type={lt.name} (Code={lt.code}, Paid={lt.is_paid}), Status={req.status}, Period={req.start_date} to {req.end_date}")

db.close()
