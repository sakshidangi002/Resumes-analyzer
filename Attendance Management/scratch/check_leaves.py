
import sqlite3
import os

db_path = "backend/attendance.db"
if not os.path.exists(db_path):
    print(f"DB not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("--- Attendance Records (PAID_LEAVE / ON_LEAVE) for May 2026 ---")
cursor.execute("SELECT employee_id, date, status FROM attendance_records WHERE status IN ('PAID_LEAVE', 'ON_LEAVE') AND date LIKE '2026-05%'")
rows = cursor.fetchall()
for r in rows:
    print(r)

print("\n--- Approved Leave Requests ---")
cursor.execute("SELECT id, employee_id, start_date, end_date, status FROM leave_requests WHERE status = 'APPROVED'")
rows = cursor.fetchall()
for r in rows:
    print(r)

conn.close()
