-- READ-ONLY. Run this against the production DB FIRST (nothing is modified).
-- It shows any duplicate daily attendance rows that migration 020 will merge.
--
--   psql "<DATABASE_URL>" -f scripts/inspect_attendance_duplicates.sql
--
-- If the first query returns 0 rows, there are no duplicates and the dedupe
-- step of the migration is a no-op (it still adds the unique constraint/index).

-- 1) Duplicate (employee_id, date) groups, with how many rows each has.
SELECT employee_id, date, COUNT(*) AS row_count, MIN(id) AS keep_id,
       ARRAY_AGG(id ORDER BY id) AS all_ids
FROM attendance_records
GROUP BY employee_id, date
HAVING COUNT(*) > 1
ORDER BY row_count DESC, employee_id, date;

-- 2) How many attendance_events are attached to the rows that would be removed
--    (these get re-pointed to the kept row by the migration — none are lost).
WITH ranked AS (
    SELECT id, MIN(id) OVER (PARTITION BY employee_id, date) AS keep_id
    FROM attendance_records
)
SELECT COUNT(*) AS events_to_repoint
FROM attendance_events e
JOIN ranked r ON e.attendance_record_id = r.id
WHERE r.id <> r.keep_id;
