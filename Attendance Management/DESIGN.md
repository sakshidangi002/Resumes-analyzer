# Attendance & HRMS – Design Notes

## Leave policy (financial year)

- **Leave cycle**: April–March (Indian financial year).
- **No carry-forward**: Leave balance does **not** carry forward to the next financial year. Any unused leaves at the end of the FY expire.
- **Allocation**: Default annual leave is allocated per employee at the start of each FY; no carried balance from the previous year.

## Technology choices

- **Database**: **PostgreSQL** (connection via `postgresql+psycopg2` or `asyncpg`).
- **Email**: **Simple SMTP** (configurable host, port, user, password, TLS). No third-party email API required.

## Out of scope (confirmed)

- Carry-forward of leaves to next FY
- Overtime, WFH, leave encashment, bonus/incentives, multiple salary cycles, digital signatures, mobile app, biometric/AI features
