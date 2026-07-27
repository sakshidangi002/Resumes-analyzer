"""Check if cameras exist in the database"""
from app.db.session import SessionLocal
from app.models.camera import CameraConfig

with SessionLocal() as db:
    cameras = db.query(CameraConfig).all()
    print(f"Total cameras in database: {len(cameras)}")
    for cam in cameras:
        print(f"  ID: {cam.id}, Name: {cam.name}, Source URL: {cam.source_url}, Enabled: {cam.enabled}")
