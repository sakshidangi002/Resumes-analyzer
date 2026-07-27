"""Merge duplicate attendance rows and enforce one record per employee per day

Revision ID: 026_unique_attendance_per_day
Revises: 025_add_body_embeddings
Create Date: 2026-07-14

Concurrent camera workers could each insert their own attendance_records row for
the same employee/day (check-then-insert with no DB constraint), which inflated
present-day counts on the dashboard (e.g. 11 present days in a 10-working-day
month -> 110%). This migration collapses the existing duplicates into the
earliest row of each group and adds the unique constraint that prevents new ones.
"""
from alembic import op
import sqlalchemy as sa


revision = '026_unique_attendance_per_day'
down_revision = '025_add_body_embeddings'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()

    # Keep the lowest id per (employee_id, date); fold the other rows' data into
    # it: earliest sign-in, latest sign-out, largest recorded work/break hours,
    # and the "most present" status.
    conn.execute(sa.text("""
        WITH keeper AS (
            SELECT employee_id, date, MIN(id) AS keep_id
            FROM attendance_records
            GROUP BY employee_id, date
            HAVING COUNT(*) > 1
        ),
        merged AS (
            SELECT k.keep_id,
                   MIN(r.sign_in_time)      AS sign_in_time,
                   MAX(r.sign_out_time)     AS sign_out_time,
                   MAX(r.total_work_hours)  AS total_work_hours,
                   MAX(r.total_break_hours) AS total_break_hours,
                   BOOL_OR(r.is_late)       AS is_late,
                   BOOL_OR(r.is_early_exit) AS is_early_exit,
                   MIN(CASE r.status
                         WHEN 'PRESENT'    THEN 1
                         WHEN 'HALF_DAY'   THEN 2
                         WHEN 'SHORT'      THEN 3
                         WHEN 'PAID_LEAVE' THEN 4
                         WHEN 'ON_LEAVE'   THEN 5
                         WHEN 'HOLIDAY'    THEN 6
                         WHEN 'WEEKLY_OFF' THEN 7
                         ELSE 8
                       END)                 AS status_rank
            FROM keeper k
            JOIN attendance_records r
              ON r.employee_id = k.employee_id AND r.date = k.date
            GROUP BY k.keep_id
        )
        UPDATE attendance_records r
        SET sign_in_time      = m.sign_in_time,
            sign_out_time     = m.sign_out_time,
            total_work_hours  = m.total_work_hours,
            total_break_hours = m.total_break_hours,
            is_late           = m.is_late,
            is_early_exit     = m.is_early_exit,
            status = CASE m.status_rank
                       WHEN 1 THEN 'PRESENT'
                       WHEN 2 THEN 'HALF_DAY'
                       WHEN 3 THEN 'SHORT'
                       WHEN 4 THEN 'PAID_LEAVE'
                       WHEN 5 THEN 'ON_LEAVE'
                       WHEN 6 THEN 'HOLIDAY'
                       WHEN 7 THEN 'WEEKLY_OFF'
                       ELSE 'ABSENT'
                     END
        FROM merged m
        WHERE r.id = m.keep_id
    """))

    # Re-point events that referenced a losing row, then drop the losers.
    conn.execute(sa.text("""
        WITH keeper AS (
            SELECT employee_id, date, MIN(id) AS keep_id
            FROM attendance_records
            GROUP BY employee_id, date
        )
        UPDATE attendance_events e
        SET attendance_record_id = k.keep_id
        FROM attendance_records r
        JOIN keeper k ON k.employee_id = r.employee_id AND k.date = r.date
        WHERE e.attendance_record_id = r.id AND r.id <> k.keep_id
    """))

    conn.execute(sa.text("""
        DELETE FROM attendance_records r
        USING (
            SELECT employee_id, date, MIN(id) AS keep_id
            FROM attendance_records
            GROUP BY employee_id, date
        ) k
        WHERE r.employee_id = k.employee_id
          AND r.date = k.date
          AND r.id <> k.keep_id
    """))

    op.create_unique_constraint(
        "uq_attendance_employee_date", "attendance_records", ["employee_id", "date"]
    )


def downgrade():
    op.drop_constraint("uq_attendance_employee_date", "attendance_records", type_="unique")
