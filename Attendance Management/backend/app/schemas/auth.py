from pydantic import BaseModel
from typing import Optional, List


class LoginRequest(BaseModel):
    username: str
    password: str


class SignupRequest(BaseModel):
    username: str
    password: str
    official_email: Optional[str] = None


class ForgotPasswordRequest(BaseModel):
    username: str


class TokenPayload(BaseModel):
    sub: str
    exp: int
    roles: Optional[List[str]] = None
    employee_id: Optional[int] = None


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    roles: List[str]
    employee_id: Optional[int] = None
    employee_code: Optional[str] = None
    designation: Optional[str] = None
