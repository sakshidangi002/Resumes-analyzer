import os
from sqlalchemy import create_engine, text

def main():
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:d5x8JGZ%40CH5td7X@43.205.127.72:5432/Attendance_system")
    engine = create_engine(db_url)
    with engine.connect() as conn:
        res = conn.execute(text("SELECT name, deleted_at FROM resumes"))
        for row in res.fetchall():
            print(row)

if __name__ == "__main__":
    main()
