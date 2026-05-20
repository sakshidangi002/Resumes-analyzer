from sqlalchemy import create_engine, text

def main():
    engine = create_engine('postgresql://postgres:d5x8JGZ%40CH5td7X@43.205.127.72:5432/Attendance_system')
    with engine.connect() as conn:
        res = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='users'"))
        cols = [r[0] for r in res.fetchall()]
        print("Columns in users:", cols)
        
        # print some users to see
        res_users = conn.execute(text(f"SELECT {', '.join(cols)} FROM users"))
        for u in res_users.fetchall():
            print(u)

if __name__ == "__main__":
    main()
