
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.models import LeaveRequest, LeaveAllocation, AttendanceRecord
from scripts.seed import seed

url = "postgresql://postgres:d5x8JGZ%40CH5td7X@43.205.127.72:5432/Attendance_system"
engine = create_engine(url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

try:
    print("Resetting Leave Data...")
    
    # 1. Delete all Leave Requests
    db.query(LeaveRequest).delete()
    print("Deleted all Leave Requests.")
    
    # 2. Delete all Leave Allocations
    db.query(LeaveAllocation).delete()
    print("Deleted all Leave Allocations.")
    
    # 3. Reset attendance statuses that were marked as ON_LEAVE back to ABSENT
    # (Since the leaves they were based on are now gone)
    db.query(AttendanceRecord).filter(AttendanceRecord.status == "ON_LEAVE").update({"status": "ABSENT"})
    print("Reset ON_LEAVE attendance records to ABSENT.")
    
    db.commit()
    print("Database changes committed.")
    
    # 4. Re-run seed to restore default types and allocations
    print("Re-running seed to restore defaults...")
    seed()
    
    print("\nSUCCESS: Leave data has been reset to a clean state.")
    print("You can now go to 'Leave Allocations' to set quotas and 'Apply Leave' to enter new requests.")

except Exception as e:
    db.rollback()
    print(f"ERROR: {e}")
finally:
    db.close()
