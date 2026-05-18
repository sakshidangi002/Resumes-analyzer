
import os
import psycopg2
from urllib.parse import unquote

# Password d5x8JGZ%40CH5td7X decoded is d5x8JGZ@CH5td7X
password = unquote("d5x8JGZ%40CH5td7X")

def check_db():
    try:
        conn = psycopg2.connect(
            host="43.205.127.72",
            port=5432,
            user="postgres",
            password=password,
            database="Attendance_system"
        )
        cur = conn.cursor()
        
        print("--- Attendance Records with Leave Status (May 2026) ---")
        cur.execute("SELECT employee_id, date, status FROM attendance_records WHERE status IN ('PAID_LEAVE', 'ON_LEAVE') AND date >= '2026-05-01' AND date <= '2026-05-31'")
        rows = cur.fetchall()
        for r in rows:
            print(r)
            
        print("\n--- All Leave Requests for May 2026 ---")
        cur.execute("SELECT id, employee_id, start_date, end_date, status FROM leave_requests WHERE start_date <= '2026-05-31' AND end_date >= '2026-05-01'")
        rows = cur.fetchall()
        for r in rows:
            print(r)
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_db()
