import sys
import os

# Add the current directory to sys.path to import 'app'
os.chdir(r"c:\sakshi folder\application\Resume analyzer\Attendance Management\backend")
sys.path.append(os.getcwd())

from app.db.session import engine
from sqlalchemy import text, inspect

print("Connecting to database...")
inspector = inspect(engine)
columns = [col["name"] for col in inspector.get_columns("attendance_events")]
print("Current columns in attendance_events:", columns)

with engine.connect() as conn:
    if "attendance_date" not in columns:
        print("Adding attendance_date column to attendance_events...")
        try:
            conn.execute(text("ALTER TABLE attendance_events ADD COLUMN attendance_date DATE"))
            print("attendance_date column added.")
        except Exception as e:
            print(f"Error adding column: {e}")
    else:
        print("attendance_date column already exists.")

    print("Updating event_type values from IN/OUT to CHECK_IN/CHECK_OUT...")
    try:
        r1 = conn.execute(text("UPDATE attendance_events SET event_type = 'CHECK_IN' WHERE event_type = 'IN'"))
        r2 = conn.execute(text("UPDATE attendance_events SET event_type = 'CHECK_OUT' WHERE event_type = 'OUT'"))
        print(f"Updated event types. IN->CHECK_IN: {r1.rowcount}, OUT->CHECK_OUT: {r2.rowcount}")
    except Exception as e:
        print(f"Error updating event types: {e}")

    # Backfill attendance_date for existing events using event_time
    print("Backfilling attendance_date for existing events...")
    try:
        r3 = conn.execute(text("UPDATE attendance_events SET attendance_date = CAST(event_time AS DATE) WHERE attendance_date IS NULL"))
        print(f"Backfilled {r3.rowcount} rows.")
    except Exception as e:
        print(f"Error backfilling: {e}")

    # Alter attendance_date to be NOT NULL if wanted, but let's keep it nullable or set NOT NULL now
    try:
        conn.execute(text("ALTER TABLE attendance_events ALTER COLUMN attendance_date SET NOT NULL"))
        print("Set attendance_date to NOT NULL.")
    except Exception as e:
        print(f"Error setting NOT NULL: {e}")

    conn.commit()
print("Finished db check.")
