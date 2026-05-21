from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class UserCreate(BaseModel):
    username: str
    password: str
    official_email: Optional[str] = None
    employee_id: Optional[int] = None
    role_names: List[str] = []


class UserUpdate(BaseModel):
    password: Optional[str] = None
    official_email: Optional[str] = None
    employee_id: Optional[int] = None
    is_active: Optional[bool] = None
    role_names: Optional[List[str]] = None


class UserResponse(BaseModel):
    id: int
    username: str
    official_email: Optional[str] = None
    is_active: bool
    employee_id: Optional[int] = None
    employee_code: Optional[str] = None
    designation: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class UserWithRoles(UserResponse):
    roles: List[str] = []
