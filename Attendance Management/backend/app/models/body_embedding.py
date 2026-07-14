"""Body (appearance) embeddings for cross-camera person Re-Identification.

One row = "on <day>, on <camera>, employee X looked like <embedding>".

Appearance is CLOTHING-based, so a row is only meaningful for the day it was
captured — rows are keyed by `day` and the Identity Manager only ever loads
today's. Old rows are harmless history and can be purged at will.

Embeddings are captured OPPORTUNISTICALLY: whenever ArcFace positively
recognises a face on a camera, that person's body crop from THAT camera is
embedded and stored. This deliberately keeps the gallery in the same viewpoint
as the camera that will later have to match it (a standing check-in view and a
ceiling-mounted seated view look nothing alike to a ReID model).
"""
from sqlalchemy import Column, Integer, String, Date, DateTime, ForeignKey, LargeBinary, Index
from sqlalchemy.orm import relationship
from app.db.base_class import Base

from app.core.datetime_utils import get_ist_now


class BodyEmbedding(Base):
    __tablename__ = "body_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False, index=True)
    camera_id = Column(String(50), nullable=False, index=True)
    day = Column(Date, nullable=False, index=True)
    embedding = Column(LargeBinary, nullable=False)   # float32[512], L2-normalised
    score = Column(Integer, nullable=True)            # face-match score ×100 (quality of the enrolment)
    created_at = Column(DateTime, default=get_ist_now)

    employee = relationship("Employee", backref="body_embeddings")


Index("ix_body_embeddings_day_camera", BodyEmbedding.day, BodyEmbedding.camera_id)
