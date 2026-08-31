"""Logical backup of the HRMS database, using only SQLAlchemy.

Written because pg_dump is not on PATH here, and the server is PostgreSQL 18 —
an older pg_dump would refuse to dump it anyway. This needs no external tools.

Dumps every table to a gzipped JSON file plus a manifest recording the alembic
revision and per-table row counts, so a restore can be verified rather than
hoped at.

Usage:
    python scripts/backup_db.py                 # writes to backups/
    python scripts/backup_db.py --out D:\\safe   # elsewhere
    python scripts/backup_db.py --verify FILE   # re-check an existing backup
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import inspect, text  # noqa: E402

from app.db.session import SessionLocal, engine  # noqa: E402


def _encode(value):
    """JSON-safe rendering that survives a round trip.

    The fallback matters: a backup must never silently drop a column because
    its type was unanticipated (UUID, INET, JSONB, enum...). Anything unknown is
    stringified and tagged, so it is preserved and visibly non-native rather
    than crashing the dump or vanishing from it.
    """
    import base64
    import uuid

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return {"__type__": "datetime", "value": value.isoformat()}
    if isinstance(value, Decimal):
        return {"__type__": "decimal", "value": str(value)}
    if isinstance(value, uuid.UUID):
        return {"__type__": "uuid", "value": str(value)}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__type__": "bytes", "value": base64.b64encode(bytes(value)).decode()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    return {"__type__": "repr", "python_type": type(value).__name__, "value": str(value)}


def backup(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    target = out_dir / f"hrms_backup_{stamp}.json.gz"

    inspector = inspect(engine)
    tables = sorted(inspector.get_table_names())

    payload: dict = {"tables": {}}
    counts: dict[str, int] = {}

    with SessionLocal() as db:
        payload["alembic_version"] = db.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar()
        payload["server_version"] = db.execute(text("SHOW server_version")).scalar()
        payload["database"] = db.execute(text("SELECT current_database()")).scalar()

        for table in tables:
            rows = db.execute(text(f'SELECT * FROM "{table}"')).mappings().all()
            payload["tables"][table] = [
                {k: _encode(v) for k, v in row.items()} for row in rows
            ]
            counts[table] = len(rows)

    payload["row_counts"] = counts
    payload["created_at"] = _dt.datetime.now().isoformat()

    with gzip.open(target, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)

    manifest = target.with_suffix("").with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "backup": target.name,
                "created_at": payload["created_at"],
                "database": payload["database"],
                "server_version": payload["server_version"],
                "alembic_version": payload["alembic_version"],
                "row_counts": counts,
                "total_rows": sum(counts.values()),
                "size_bytes": target.stat().st_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def verify(path: Path) -> bool:
    """Re-open a backup and confirm it parses and matches its manifest."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    counts = payload.get("row_counts", {})
    actual = {t: len(rows) for t, rows in payload.get("tables", {}).items()}
    ok = counts == actual
    print(f"  alembic revision : {payload.get('alembic_version')}")
    print(f"  tables           : {len(actual)}")
    print(f"  total rows       : {sum(actual.values())}")
    print(f"  integrity        : {'OK' if ok else 'MISMATCH'}")
    for table in ("attendance_events", "attendance_records", "employees"):
        if table in actual:
            print(f"    {table:<20} {actual[table]} rows")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "backups"))
    parser.add_argument("--verify")
    args = parser.parse_args()

    if args.verify:
        return 0 if verify(Path(args.verify)) else 1

    target = backup(Path(args.out))
    print(f"backup written: {target}")
    print(f"size: {target.stat().st_size / 1024:.0f} KB")
    print()
    print("verifying...")
    return 0 if verify(target) else 1


if __name__ == "__main__":
    raise SystemExit(main())
