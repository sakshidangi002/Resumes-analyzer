# Implementation Plan - Refactor Attendance System (Summary vs Events)

Refactor the biometric attendance system to store raw CHECK_IN and CHECK_OUT events in a dedicated `attendance_events` table while preserving a simple, high-level summary in the `attendance_records` table. 

## User Review Required

> [!IMPORTANT]
> - **Database Schema Migration:** The `attendance_events` table has been successfully migrated on the remote database. We added `attendance_date` (Date) and backfilled all existing events, converting legacy `IN` -> `CHECK_IN` and `OUT` -> `CHECK_OUT`.
> - **Database Column Synonyms:** To avoid breaking any existing database queries or reports, the `attendance_records` table columns (`sign_in_time` and `sign_out_time`) are mapped in Python using SQLAlchemy `synonyms` as `first_check_in` and `last_check_out` respectively.

## Proposed Changes

### Database & Models

#### [MODIFY] [attendance.py](file:///c:/sakshi%20folder/application/Resume%20analyzer/Attendance%20Management/backend/app/models/attendance.py)
- Modify `AttendanceEvent` to include the `attendance_date` field (type `Date`, indexable, not null).
- Explicitly update comment/choices for `event_type` to `CHECK_IN` and `CHECK_OUT`.
- Add `first_check_in = synonym("sign_in_time")` and `last_check_out = synonym("sign_out_time")` to `AttendanceRecord` to ensure Python property-level compatibility with the new terms.

### Pydantic Schemas

#### [MODIFY] [attendance.py](file:///c:/sakshi%20folder/application/Resume%20analyzer/Attendance%20Management/backend/app/schemas/attendance.py)
- Update `AttendanceEventResponse` to include `attendance_date`.
- Add `first_check_in` and `last_check_out` as optional `time` fields to `AttendanceRecordResponse`, `AttendanceDetailsResponse`, and `DailyAttendanceReportRow`. Preserve `sign_in_time` and `sign_out_time` for backward compatibility.
- Ensure that during validation/serialization, `first_check_in` reads from `sign_in_time` (or synonym) and `last_check_out` from `sign_out_time` (or synonym).

### Services (Backend Business Logic)

#### [MODIFY] [attendance_event_service.py](file:///c:/sakshi%20folder/application/Resume%20analyzer/Attendance%20Management/backend/app/services/attendance_event_service.py)
- Update `determine_next_event_type` to return `"CHECK_IN"` and `"CHECK_OUT"` (with legacy support for `"IN"` and `"OUT"`).
- Update `calculate_intervals_from_events` to compute work/break intervals using `CHECK_IN` and `CHECK_OUT` strings (with backward compatibility for `IN`/`OUT`).
- Update `add_attendance_event` to set `attendance_date` when creating new event records and translate event types if needed.
- Update `recalculate_attendance_summary` to write to `rec.sign_in_time` and `rec.sign_out_time` (via the synonyms `first_check_in` and `last_check_out`).

### API Endpoints

#### [MODIFY] [attendance.py](file:///c:/sakshi%20folder/application/Resume%20analyzer/Attendance%20Management/backend/app/api/routes/attendance.py)
- Update the `/events` GET route (`list_attendance_events`) to make the `date` query parameter optional. If omitted, return all events for the specified employee sorted by `event_time` descending.

### Frontend Integration

#### [MODIFY] [FaceDetection.tsx](file:///c:/sakshi%20folder/application/Resume%20analyzer/Attendance%20Management/frontend/src/pages/FaceDetection.tsx)
- Support both legacy (`IN`/`OUT`) and new (`CHECK_IN`/`CHECK_OUT`) statuses in face recognition results, display labels, and cache handling.

#### [MODIFY] [Attendance.tsx](file:///c:/sakshi%20folder/application/Resume%20analyzer/Attendance%20Management/frontend/src/pages/Attendance.tsx)
- Rename table headers from "First In" to "First Check-In" and "Last Out" to "Last Check-Out".
- Remove the "Required Time" column from the Daily Attendance table to keep the UI simple.
- Adjust table rendering for 7 columns (Member, First Check-In, Last Check-Out, Working Hours, Break Time, Status, Actions).

#### [MODIFY] [EmployeeProfile.tsx](file:///c:/sakshi%20folder/application/Resume%20analyzer/Attendance%20Management/frontend/src/pages/EmployeeProfile.tsx)
- Add a new tab `attendance_details` labeled "Attendance Details".
- Implement the "Attendance Timeline" UI card showing daily stats (First Check-In, Last Check-Out, Working Hours, Break Time) for a selected date (with a date picker defaulting to today).
- Render a scrollable table displaying the complete list of check-in/check-out events (columns: Date, Event Time, Event Type, Source) fetched from the backend.

## Verification Plan

### Automated Tests
- Run backend server and test endpoints:
  - Check `GET /api/attendance/events?employee_id=X` returns complete history.
  - Check `GET /api/attendance/details?employee_id=X&date=YYYY-MM-DD` returns correct timeline.
  - Run face recognition simulation.

### Manual Verification
- Test webcam scanning to confirm alternating events (`CHECK_IN` -> `CHECK_OUT` -> `CHECK_IN`) are registered with 60-second cooldown protection.
- Open `/attendance` and check the simplified daily summary table.
- Open an employee profile and check the new **Attendance Details** tab, verify the timeline stats, and inspect the chronological history log.
