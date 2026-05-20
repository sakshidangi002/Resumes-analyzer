from sqlalchemy import create_engine, text
import os

DATABASE_URL = "postgresql://postgres:d5x8JGZ%40CH5td7X@43.205.127.72:5432/Attendance_system"
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    print("USERS:")
    users = conn.execute(text("SELECT id, username, employee_id, is_active FROM users")).fetchall()
    for u in users:
        print(u)
        
    print("\nROLES:")
    roles = conn.execute(text("SELECT id, name FROM roles")).fetchall()
    for r in roles:
        print(r)
        
    print("\nUSER ROLES:")
    ur = conn.execute(text("SELECT user_id, role_id FROM user_roles")).fetchall()
    for row in ur:
        print(row)
