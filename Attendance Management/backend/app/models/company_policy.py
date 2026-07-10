"""Company policies with version history.

Append-only: each published version of a named policy is a new row. Publishing a
new version never overwrites the old one, so previous policies stay viewable.
The "current" policy for a name is simply its highest-version row.
"""
from sqlalchemy import Column, Integer, String, Text, Date, DateTime, ForeignKey
from app.db.base_class import Base

from app.core.datetime_utils import get_ist_now


class CompanyPolicy(Base):
    __tablename__ = "company_policies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(150), nullable=False, index=True)  # policy identity; groups versions
    title = Column(String(200), nullable=True)              # heading for this version
    category = Column(String(80), nullable=True)
    content = Column(Text, nullable=True)                   # typed body (optional if a file is attached)
    effective_date = Column(Date, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    attachment_path = Column(String(500), nullable=True)    # server-side stored path
    attachment_name = Column(String(255), nullable=True)    # original filename (for download)
    published_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    published_by_name = Column(String(150), nullable=True)
    created_at = Column(DateTime, default=get_ist_now)
