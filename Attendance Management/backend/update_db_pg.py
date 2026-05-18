import sys
import os

# Add the current directory to sys.path to import 'app'
os.chdir(r"c:\Attendance Management\backend")
sys.path.append(os.getcwd())

from app.db.session import engine
from sqlalchemy import text

print("Connecting to database...")
with engine.connect() as conn:
    print("Checking/Adding priority column...")
    try:
        conn.execute(text("ALTER TABLE onboarding_tasks ADD COLUMN priority VARCHAR(20) DEFAULT 'Medium'"))
        print("Priority column added.")
    except Exception as e:
        print(f"Priority column might already exist or error: {e}")
    
    print("Checking/Adding due_date column...")
    try:
        conn.execute(text("ALTER TABLE onboarding_tasks ADD COLUMN due_date TIMESTAMP WITHOUT TIME ZONE"))
        print("Due_date column added.")
    except Exception as e:
        print(f"Due_date column might already exist or error: {e}")
    
    conn.commit()
print("Database schema update attempt finished.")
