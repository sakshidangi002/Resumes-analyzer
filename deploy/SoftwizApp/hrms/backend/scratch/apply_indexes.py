import sys
from pathlib import Path

# Add backend dir to path so we can import app modules
backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir))

from app.db.session import SessionLocal
from sqlalchemy import text

def apply_indexes():
    db = SessionLocal()
    queries = [
        "CREATE INDEX IF NOT EXISTS ix_attendance_records_date ON attendance_records (date);",
        "CREATE INDEX IF NOT EXISTS ix_attendance_records_employee_id ON attendance_records (employee_id);",
        "CREATE INDEX IF NOT EXISTS ix_leave_requests_employee_id ON leave_requests (employee_id);",
        "CREATE INDEX IF NOT EXISTS ix_leave_requests_start_date ON leave_requests (start_date);",
        "CREATE INDEX IF NOT EXISTS ix_leave_requests_end_date ON leave_requests (end_date);",
        "CREATE INDEX IF NOT EXISTS ix_leave_allocations_employee_id ON leave_allocations (employee_id);",
        "CREATE INDEX IF NOT EXISTS ix_leave_allocations_financial_year_id ON leave_allocations (financial_year_id);"
    ]
    try:
        for q in queries:
            print(f"Running: {q}")
            db.execute(text(q))
        db.commit()
        print("Indexes applied successfully.")
    except Exception as e:
        print(f"Error applying indexes: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    apply_indexes()
