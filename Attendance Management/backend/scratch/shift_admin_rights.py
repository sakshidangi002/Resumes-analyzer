
from app.db.session import SessionLocal
from app.models import User, AttendanceCorrectionRequest, LeaveRequest, LetterInstance, LetterReply, AuditLog, AppNotification

def shift_rights(from_username: str, to_username: str):
    db = SessionLocal()
    try:
        from_user = db.query(User).filter(User.username == from_username).first()
        to_user = db.query(User).filter(User.username == to_username).first()

        if not from_user:
            print(f"Error: User '{from_username}' not found.")
            return
        if not to_user:
            print(f"Error: User '{to_username}' not found.")
            return

        print(f"Shifting records from {from_username} (ID: {from_user.id}) to {to_username} (ID: {to_user.id})...")

        # 1. Attendance Corrections
        db.query(AttendanceCorrectionRequest).filter(AttendanceCorrectionRequest.approver_id == from_user.id).update(
            {"approver_id": to_user.id}, synchronize_session=False
        )
        # 2. Leave Requests
        db.query(LeaveRequest).filter(LeaveRequest.manager_approver_id == from_user.id).update(
            {"manager_approver_id": to_user.id}, synchronize_session=False
        )
        db.query(LeaveRequest).filter(LeaveRequest.hr_approver_id == from_user.id).update(
            {"hr_approver_id": to_user.id}, synchronize_session=False
        )
        # 3. Letters
        db.query(LetterInstance).filter(LetterInstance.generated_by_user_id == from_user.id).update(
            {"generated_by_user_id": to_user.id}, synchronize_session=False
        )
        db.query(LetterReply).filter(LetterReply.author_user_id == from_user.id).update(
            {"author_user_id": to_user.id}, synchronize_session=False
        )
        # 4. Audit Logs
        db.query(AuditLog).filter(AuditLog.user_id == from_user.id).update(
            {"user_id": to_user.id}, synchronize_session=False
        )
        
        # 5. Notifications - Just delete them for 'abc'
        cnt_notif = db.query(AppNotification).filter(AppNotification.user_id == from_user.id).delete()
        print(f"- Deleted {cnt_notif} old notifications for abc.")

        db.commit()
        print("Success: All rights shifted and cleanup complete.")
    except Exception as e:
        db.rollback()
        print(f"Error during shift: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    shift_rights("abc", "admin")
