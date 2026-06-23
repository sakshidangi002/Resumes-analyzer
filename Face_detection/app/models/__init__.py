from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

from .employee import Employee
from .camera import Camera
from .event import AttendanceEvent
from .daily import DailyAttendance

