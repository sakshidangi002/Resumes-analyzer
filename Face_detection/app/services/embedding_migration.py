from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from app.db import blob_to_embedding, db_cursor, embedding_to_blob, get_app_meta, set_app_meta
from app.services.face_service import EMBEDDING_MODEL_VERSION, extract_face_embeddings
from app.services.match import cosine_similarity


def _load_face_embedding(image_path: str):
    path = Path(image_path)
    if not path.exists():
        return None

    try:
        image = Image.open(path).convert("RGB")
        faces = extract_face_embeddings(image)
        if not faces:
            return None
        return faces[0]["embedding"]
    except Exception:
        return None


def embedding_migration_needed() -> bool:
    stored_version = get_app_meta("embedding_model")
    if stored_version != EMBEDDING_MODEL_VERSION:
        return True

    with db_cursor() as (_, cur):
        rows = cur.execute(
            """
            SELECT employee_id, embedding
            FROM face_samples
            ORDER BY employee_id, id
            LIMIT 6
            """
        ).fetchall()

    if len(rows) < 2:
        return False

    embeddings = [blob_to_embedding(row["embedding"]) for row in rows]
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarities.append(cosine_similarity(embeddings[i], embeddings[j]))

    if not similarities:
        return False

    return sum(similarities) / len(similarities) > 0.999


def rebuild_embeddings_from_samples() -> dict:
    with db_cursor() as (_, cur):
        samples = cur.execute(
            """
            SELECT id, employee_id, image_path
            FROM face_samples
            ORDER BY employee_id, id
            """
        ).fetchall()

    by_employee: dict[int, list] = defaultdict(list)
    updated_samples = 0

    for sample in samples:
        embedding = _load_face_embedding(sample["image_path"])
        if embedding is None:
            continue

        by_employee[int(sample["employee_id"])].append(embedding)
        with db_cursor() as (_, cur):
            cur.execute(
                "UPDATE face_samples SET embedding = ? WHERE id = ?",
                (embedding_to_blob(embedding), sample["id"]),
            )
        updated_samples += 1

    updated_employees = 0
    with db_cursor() as (_, cur):
        employee_rows = cur.execute(
            "SELECT id, name FROM employees ORDER BY id"
        ).fetchall()

    for employee in employee_rows:
        embs = by_employee.get(int(employee["id"]), [])
        if not embs:
            continue

        mean_embedding = np.mean(np.stack(embs), axis=0).astype(np.float32)
        with db_cursor() as (_, cur):
            sample_count = cur.execute(
                "SELECT COUNT(*) AS count FROM face_samples WHERE employee_id = ?",
                (employee["id"],),
            ).fetchone()["count"]
            cur.execute(
                """
                UPDATE employees
                SET embedding = ?, sample_count = ?
                WHERE id = ?
                """,
                (embedding_to_blob(mean_embedding), sample_count, employee["id"]),
            )
        updated_employees += 1

    set_app_meta("embedding_model", EMBEDDING_MODEL_VERSION)

    return {
        "updated_samples": updated_samples,
        "updated_employees": updated_employees,
    }
