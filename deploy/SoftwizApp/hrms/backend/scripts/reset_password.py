"""
Reset (or create) a user's password in the database.

Run from backend dir:
  python -m scripts.reset_password --username admin --password admin123

Uses the same hashing as the app (bcrypt via passlib).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.security import get_password_hash  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.models import User  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--username", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--create-if-missing", action="store_true", default=True)
    args = p.parse_args()

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == args.username).first()
        if not user:
            if not args.create_if_missing:
                print(f"User '{args.username}' not found.")
                return 2
            user = User(username=args.username, password_hash=get_password_hash(args.password), is_active=True)
            db.add(user)
            db.commit()
            print(f"Created user '{args.username}' with new password.")
            return 0

        user.password_hash = get_password_hash(args.password)
        db.add(user)
        db.commit()
        print(f"Password reset for user '{args.username}'.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

