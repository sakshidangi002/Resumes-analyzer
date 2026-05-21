"""Ensure one payslip per employee per period.

- Deduplicate existing rows (keep newest generated_at / id)
- Add unique constraint on (employee_id, payroll_period_id)
"""

from alembic import op


revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Remove duplicates before adding constraint (Postgres).
    # Keep the newest row for each (employee_id, payroll_period_id) pair.
    op.execute(
        """
        DELETE FROM payslips p
        USING payslips p2
        WHERE p.employee_id = p2.employee_id
          AND p.payroll_period_id = p2.payroll_period_id
          AND (
            p.generated_at < p2.generated_at
            OR (p.generated_at = p2.generated_at AND p.id < p2.id)
          );
        """
    )
    op.create_unique_constraint(
        "uq_payslips_employee_period",
        "payslips",
        ["employee_id", "payroll_period_id"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_payslips_employee_period", "payslips", type_="unique")

