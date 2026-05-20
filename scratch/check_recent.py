from sqlalchemy import create_engine, text

def main():
    engine = create_engine('postgresql://postgres:d5x8JGZ%40CH5td7X@43.205.127.72:5432/Attendance_system')
    with engine.connect() as conn:
        res = conn.execute(text("SELECT name, skills, source_file, created_at FROM resumes ORDER BY created_at DESC LIMIT 5"))
        for row in res.fetchall():
            print("--------------------------------------------------")
            print("Name:", row[0])
            print("Skills:", row[1])
            print("Source File:", row[2])
            print("Created At:", row[3])

if __name__ == "__main__":
    main()
