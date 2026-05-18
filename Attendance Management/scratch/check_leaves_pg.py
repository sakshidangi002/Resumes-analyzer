
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv("backend/.env")

def check_db():
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DB")
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
