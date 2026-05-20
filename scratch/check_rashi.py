from sqlalchemy import create_engine, text

def main():
    engine = create_engine('postgresql://postgres:d5x8JGZ%40CH5td7X@43.205.127.72:5432/Attendance_system')
    with engine.connect() as conn:
        res = conn.execute(text("SELECT name, skills, key_skills, primary_skills, other_skills, source_file, created_at FROM resumes WHERE name ILIKE '%Rashi%' ORDER BY created_at DESC LIMIT 1"))
        row = res.fetchone()
        if not row:
            print("No candidate named Rashi found!")
            return
        
        print("Name:", row[0])
        print("Skills in DB:", row[1])
        print("Key Skills in DB:", row[2])
        print("Primary Skills in DB:", row[3])
        print("Other Skills in DB:", row[4])
        print("Source File:", row[5])
        print("Created At:", row[6])

if __name__ == "__main__":
    main()
