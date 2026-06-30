from fastapi import APIRouter
from app.api.routes import (
    auth,
    employees,
    attendance,
    leave,
    payroll,
    letters,
    reports,
    calendar,
    users,
    company,
    activity,
    onboarding,
    dsr,
    dsr_reminder,
    push,
    recognition,
    cameras,
)

api_router = APIRouter()
api_router.include_router(auth.router, prefix="")
api_router.include_router(company.router, prefix="")
api_router.include_router(users.router, prefix="/users")
api_router.include_router(employees.router, prefix="/employees", tags=["employees"])
api_router.include_router(attendance.router, prefix="/attendance", tags=["attendance"])
api_router.include_router(leave.router, prefix="/leave", tags=["leave"])
api_router.include_router(payroll.router, prefix="/payroll", tags=["payroll"])
api_router.include_router(letters.router, prefix="/letters", tags=["letters"])
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_router.include_router(calendar.router, prefix="/calendar", tags=["calendar"])
api_router.include_router(activity.router, prefix="/activity", tags=["activity"])
api_router.include_router(onboarding.router, prefix="/onboarding", tags=["onboarding"])
api_router.include_router(dsr.router, prefix="/dsr", tags=["dsr"])
api_router.include_router(
    dsr_reminder.router, prefix="/dsr-reminder", tags=["dsr-reminder"]
)
api_router.include_router(push.router, prefix="/push", tags=["push"])
api_router.include_router(recognition.router, prefix="", tags=["recognition"])
api_router.include_router(cameras.router, prefix="", tags=["cameras"])
