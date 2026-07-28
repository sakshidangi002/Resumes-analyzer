"""Which pay/leave rules apply to a given person.

One definition, imported by payroll and leave alike, so the two can never drift
into disagreeing about who is on a fixed salary — a disagreement that would show
up as a wrong payslip rather than an error.
"""


def is_fixed_salary_staff(employee) -> bool:
    """True for NON-EMPLOYEE staff: housekeeping, security, drivers, and so on.

    These people work variable hours with no fixed daily target. Their salary is
    a flat monthly amount:

      * attendance is recorded for presence and reporting, but never reduces pay
        (no LOP, whatever the hours);
      * no leave-account involvement — short days must not be charged against
        Short/Paid Leave buffers, which they typically have no allocation for;
      * salary ADVANCES are still deducted; that is the only deduction.

    `staff_type` is the discriminator: "Employee" (or unset, for older rows
    written before the column existed) means the normal attendance-linked payroll
    applies; anything else means fixed salary.
    """
    staff_type = (getattr(employee, "staff_type", None) or "Employee").strip().lower()
    return staff_type != "employee"
