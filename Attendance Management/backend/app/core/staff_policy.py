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


# The daily target assumed for someone who has never had one set. Only ever
# applied to a NULL, never to a zero -- see below.
DEFAULT_EXPECTED_HOURS = 9.0


def expected_daily_hours(employee):
    """This person's daily hour target, or None if they have no fixed hours.

    Zero is a REAL value here, not a missing one. `create_support_staff` already
    writes `expected_working_hours=0.0` with the comment "no work-hour
    expectation for support staff", so the intent has always been that 0 means
    "no fixed daily target". Every reader defeated it by writing
    `float(emp.expected_working_hours or 9.0)` -- and `0.0 or 9.0` is `9.0`, so
    a person on no fixed hours was silently measured against a nine-hour day.

    Returning None rather than 0.0 forces each caller to say what it does when
    there is no target, instead of dividing by it or subtracting from it:

        expected = expected_daily_hours(emp)
        if expected is None:
            ...            # no target to measure against
    """
    raw = getattr(employee, "expected_working_hours", None)
    if raw is None:
        return DEFAULT_EXPECTED_HOURS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_EXPECTED_HOURS
    # Negative is not meaningful either; treat it the same as zero rather than
    # letting it flow into a half-day threshold as a negative number.
    return None if value <= 0 else value
