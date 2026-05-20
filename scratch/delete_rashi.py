from sqlalchemy import create_engine, text

def main():
    engine = create_engine('postgresql://postgres:d5x8JGZ%40CH5td7X@43.205.127.72:5432/Attendance_system')
    with engine.connect() as conn:
        res = conn.execute(text("DELETE FROM resumes WHERE name ILIKE '%Rashi%'"))
        conn.commit()
        print("Successfully deleted old Rashi Thakur records!")

if __name__ == "__main__":
    main()
