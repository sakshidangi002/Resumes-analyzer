from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.db import init_db
from app.routers import attendance, cameras, employees, health, pages, recognition
from app.services.camera_service import camera_manager
from app.services.embedding_cache import warm_embedding_cache
from app.services.embedding_migration import (
    embedding_migration_needed,
    rebuild_embeddings_from_samples,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    if embedding_migration_needed():
        result = rebuild_embeddings_from_samples()
        logger.info("Embedding migration completed: %s", result)
    cached = warm_embedding_cache()
    logger.info("Loaded %s employee embedding candidates into cache", cached)
    camera_manager.sync_from_db()
    yield
    camera_manager.stop_all()


app = FastAPI(title="Attendance System", version="1.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(pages.router)
app.include_router(health.router)
app.include_router(employees.router)
app.include_router(recognition.router)
app.include_router(attendance.router)
app.include_router(cameras.router)
