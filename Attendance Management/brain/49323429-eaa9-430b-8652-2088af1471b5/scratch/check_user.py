from app.db.session import SessionLocal
from app.models import User

db = SessionLocal()
user = db.query(User).filter(User.username == 'saloni').first()
if user:
    print(f"User found: {user.username}")
    print(f"Active: {user.is_active}")
    print(f"Official Email: {user.official_email}")
else:
    print("User 'saloni' not found.")
db.close()
