"""One-time backfill: encrypt face embeddings that are still stored as plaintext.

WHY THIS IS NEEDED
------------------
`EncryptedBinary` (app/core/encrypted_types.py) encrypts on WRITE and reads
legacy plaintext transparently, so switching the column type protected new
enrolments but left every existing row exactly as it was. Adding the column type
therefore closed the issue in code while the biometric data on disk stayed in
the clear. This script rewrites those rows so the encryption actually applies.

IDEMPOTENT: rows already carrying the SWENC1: marker are skipped, so it is safe
to run repeatedly, and safe to run again after new enrolments.

USAGE
-----
    cd "Attendance Management/backend"
    python scripts/encrypt_existing_embeddings.py            # report only
    python scripts/encrypt_existing_embeddings.py --apply    # perform the write

Run the report first. Take a database backup before --apply: these rows are
biometric enrolments, and if EMBEDDING_ENCRYPTION_KEY / SECRET_KEY is not stable
between runs the ciphertext cannot be recovered (see config.py).
"""
import argparse
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import text  # noqa: E402

from app.core.encrypted_types import EncryptedBinary  # noqa: E402
from app.db.session import engine  # noqa: E402

PREFIX = EncryptedBinary.prefix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually write the encrypted values (default: report only)",
    )
    args = parser.parse_args()

    codec = EncryptedBinary()
    # Deliberately raw SQL, NOT the ORM: selecting through the mapped column
    # would run process_result_value and hand back decrypted bytes, making
    # encrypted and plaintext rows indistinguishable. We need the stored form.
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, embedding FROM employees WHERE embedding IS NOT NULL")
        ).fetchall()

        plaintext = [(r[0], bytes(r[1])) for r in rows if not bytes(r[1]).startswith(PREFIX)]
        already = len(rows) - len(plaintext)

        print(f"employees with an embedding : {len(rows)}")
        print(f"  already encrypted         : {already}")
        print(f"  still plaintext           : {len(plaintext)}")

        if not plaintext:
            print("\nNothing to do — every stored embedding is encrypted.")
            return 0

        if not args.apply:
            print("\nReport only. Re-run with --apply to encrypt these rows.")
            print("Take a database backup first.")
            return 0

        if not os.getenv("SECRET_KEY") and not os.getenv("EMBEDDING_ENCRYPTION_KEY"):
            print(
                "\nNeither SECRET_KEY nor EMBEDDING_ENCRYPTION_KEY is set in the "
                "environment; the key would come from .env. That is fine for a "
                "normal deployment — just make sure it is the SAME key the "
                "application will run with, or these rows become unreadable."
            )

        converted = 0
        for employee_id, raw in plaintext:
            encrypted = codec.process_bind_param(raw, None)
            # Verify BEFORE committing that what we are about to store decrypts
            # back to the original. A silent mistake here destroys enrolments.
            if codec.process_result_value(encrypted, None) != raw:
                conn.rollback()
                print(f"\nABORTED at employee {employee_id}: round-trip check failed.")
                print("No changes were committed.")
                return 1
            conn.execute(
                text("UPDATE employees SET embedding = :blob WHERE id = :id"),
                {"blob": encrypted, "id": employee_id},
            )
            converted += 1

        conn.commit()
        print(f"\nEncrypted {converted} embedding(s).")

    # Re-read independently to confirm the committed state.
    with engine.connect() as conn:
        remaining = [
            r for r in conn.execute(
                text("SELECT embedding FROM employees WHERE embedding IS NOT NULL")
            ).fetchall()
            if not bytes(r[0]).startswith(PREFIX)
        ]
    print(f"Verification: {len(remaining)} plaintext embedding(s) remain.")
    return 0 if not remaining else 1


if __name__ == "__main__":
    raise SystemExit(main())
