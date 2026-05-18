import sqlite3
import os

db_path = r"c:\Attendance Management\backend\attendance.db"

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if priority column exists
    cursor.execute("PRAGMA table_info(onboarding_tasks)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if "priority" not in columns:
        print("Adding priority column...")
        cursor.execute("ALTER TABLE onboarding_tasks ADD COLUMN priority VARCHAR(20) DEFAULT 'Medium'")
    
    if "due_date" not in columns:
        print("Adding due_date column...")
        cursor.execute("ALTER TABLE onboarding_tasks ADD COLUMN due_date DATETIME")
    
    conn.commit()
    conn.close()
    print("Database schema updated successfully.")
else:
    print("Database file not found.")
