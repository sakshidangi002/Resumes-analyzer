"""Turning "last month", "in August", "today" into concrete dates.

Deterministic and regex-based. A 1.5B model gets date arithmetic wrong often
enough to produce confidently incorrect attendance figures, which is worse than
asking which month was meant.

Every "now" goes through `get_ist_now()`. The HRMS treats IST as the business
timezone throughout, and "yesterday" resolved in UTC is wrong for five and a
half hours of every day.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

from app.core.datetime_utils import get_ist_now

#: Floor for a cumulative range. Earlier than any HRMS record, and a concrete
#: date so every query stays a plain BETWEEN rather than a special case.
EPOCH = date(2000, 1, 1)

MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

_TODAY = re.compile(r"\b(today|right now|currently|at the moment|now)\b", re.I)
_YESTERDAY = re.compile(r"\byesterday\b", re.I)
_TOMORROW = re.compile(r"\btomorrow\b", re.I)
_THIS_WEEK = re.compile(r"\b(this|current) week\b", re.I)
_LAST_WEEK = re.compile(r"\blast week\b", re.I)
_LAST_MONTH = re.compile(r"\b(last|previous|prev) month\b", re.I)
_THIS_MONTH = re.compile(r"\b(this|current) month\b", re.I)
_MONTHS_AGO = re.compile(r"\b(\d{1,2}) months? ago\b", re.I)
_THIS_YEAR = re.compile(r"\b(this|current) year\b", re.I)
_LAST_YEAR = re.compile(r"\blast year\b", re.I)
#: "to date", "till now", "from the beginning" — everything on record, not a
#: month. Checked before month names so "till 31 August" is read as a cumulative
#: range ending in August, not as the month of August on its own.
_CUMULATIVE = re.compile(
    r"(?:to date|till date|till now|until now|upto now|up to now|so far"
    r"|all time|all-time|cumulative|lifetime|ever paid|total paid"
    r"|from (?:the )?(?:start|starting|beginning|inception)"
    r"|since (?:the )?(?:start|beginning|inception))",
    re.I,
)
#: "till 31 august", "up to 15 March" — a cumulative total with an explicit end.
_UNTIL_DATE = re.compile(
    r"(?:till|until|upto|up\s+to|through|to)\s+"
    r"(\d{1,2})\s*(?:st|nd|rd|th)?\s*([A-Za-z]{3,9})",
    re.I,
)

_NUMERIC = re.compile(r"\b(0?[1-9]|1[0-2])[/-](20\d{2})\b")
_ISO_MONTH = re.compile(r"\b(20\d{2})-(0?[1-9]|1[0-2])\b")
_BARE_YEAR = re.compile(r"\b(20\d{2})\b")


@dataclass(frozen=True)
class Period:
    """The stretch of time a question is about.

    Always carries a month and year so month-shaped lookups have something to
    work with, plus `day` when the question named a specific date. A skill that
    reports a single day reads `day`; one that reports a month ignores it.
    """

    month: int
    year: int
    label: str
    #: Set only when a specific day was named ("today", "yesterday").
    day: date | None = None
    #: True when nothing was stated and the current month was assumed.
    assumed: bool = False
    #: A day-shaped question, so answering with a monthly total would be wrong.
    is_day: bool = False
    #: Everything on record up to the end of the range — "to date", "till 31
    #: August". Skills that aggregate must widen their filter rather than pin a
    #: single month: "total payroll paid to date" answered for the current month
    #: alone reports a confident zero for a company that has been paying salaries
    #: for years.
    cumulative: bool = False
    #: Start/end for range lookups; equals the month bounds unless a week was named.
    start: date | None = None
    end: date | None = None

    @property
    def month_name(self) -> str:
        return MONTH_NAMES[self.month - 1]

    @property
    def month_label(self) -> str:
        return f"{self.month_name} {self.year}"

    def bounds(self) -> tuple[date, date]:
        """The date range this period covers."""
        if self.start and self.end:
            return self.start, self.end
        month_end = date(
            self.year, self.month, calendar.monthrange(self.year, self.month)[1]
        )
        if self.cumulative:
            return EPOCH, month_end
        return date(self.year, self.month, 1), month_end

    def covers_month(self, month: int, year: int) -> bool:
        """Does this period include the given payroll month?

        Payroll is stored per month rather than per date, so aggregating over a
        cumulative range means "every period up to and including this one",
        which a date BETWEEN cannot express.
        """
        if not self.cumulative:
            return month == self.month and year == self.year
        return (year, month) <= (self.year, self.month)

    def on(self) -> date:
        """The single day this period is about - the named day, or today."""
        return self.day or get_ist_now().date()


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    index = (year * 12 + (month - 1)) + delta
    return index // 12, index % 12 + 1


def _day(when: date, label: str) -> Period:
    return Period(
        month=when.month, year=when.year, label=label, day=when,
        is_day=True, start=when, end=when,
    )


def parse(text: str, today: date | None = None) -> Period:
    """Resolve the period a question refers to. Never fails; defaults to this month."""
    today = today or get_ist_now().date()
    low = text.lower()

    if _TODAY.search(low):
        return _day(today, "today")
    if _YESTERDAY.search(low):
        return _day(today - timedelta(days=1), "yesterday")
    if _TOMORROW.search(low):
        return _day(today + timedelta(days=1), "tomorrow")

    # Weeks run Monday-Sunday, matching how the attendance reports read.
    if _THIS_WEEK.search(low):
        start = today - timedelta(days=today.weekday())
        return Period(today.month, today.year, "this week",
                      start=start, end=start + timedelta(days=6))
    if _LAST_WEEK.search(low):
        start = today - timedelta(days=today.weekday() + 7)
        return Period(start.month, start.year, "last week",
                      start=start, end=start + timedelta(days=6))

    # Cumulative, before month names: "till 31 August" is a range that ends in
    # August, not the month of August by itself.
    until = _UNTIL_DATE.search(low)
    if _CUMULATIVE.search(low) or until:
        end_month, end_year, label = today.month, today.year, "to date"
        if until:
            month_number = MONTHS.get(until.group(2).lower())
            if month_number:
                # A bare month means its most recent occurrence, as elsewhere.
                end_year = today.year if month_number <= today.month else today.year - 1
                end_month = month_number
                label = f"up to {MONTH_NAMES[month_number - 1]} {end_year}"
            elif not _CUMULATIVE.search(low):
                # "to" followed by something that isn't a month is not a period
                # at all ("paid to employees") — fall through to normal parsing.
                until = None
        if until or _CUMULATIVE.search(low):
            return Period(end_month, end_year, label, cumulative=True)

    if _LAST_MONTH.search(low):
        year, month = _shift_month(today.year, today.month, -1)
        return Period(month, year, "last month")
    if _THIS_MONTH.search(low):
        return Period(today.month, today.year, "this month")

    ago = _MONTHS_AGO.search(low)
    if ago:
        year, month = _shift_month(today.year, today.month, -int(ago.group(1)))
        return Period(month, year, f"{ago.group(1)} months ago")

    for name, number in MONTHS.items():
        if re.search(rf"(?<![a-z]){name}(?![a-z])", low):
            year_match = re.search(rf"(?<![a-z]){name}(?![a-z])[^\d]{{0,6}}(20\d{{2}})", low)
            if year_match:
                year = int(year_match.group(1))
            else:
                # A bare month name means its most recent occurrence, not a
                # future one: "attendance in December" asked in March means
                # last December.
                year = today.year if number <= today.month else today.year - 1
            return Period(number, year, f"{MONTH_NAMES[number - 1]} {year}")

    numeric = _NUMERIC.search(low)
    if numeric:
        month, year = int(numeric.group(1)), int(numeric.group(2))
        return Period(month, year, f"{MONTH_NAMES[month - 1]} {year}")

    iso = _ISO_MONTH.search(low)
    if iso:
        month, year = int(iso.group(2)), int(iso.group(1))
        return Period(month, year, f"{MONTH_NAMES[month - 1]} {year}")

    if _LAST_YEAR.search(low):
        return Period(today.month, today.year - 1, str(today.year - 1),
                      start=date(today.year - 1, 1, 1), end=date(today.year - 1, 12, 31))
    if _THIS_YEAR.search(low):
        return Period(today.month, today.year, str(today.year),
                      start=date(today.year, 1, 1), end=date(today.year, 12, 31))

    bare_year = _BARE_YEAR.search(low)
    if bare_year:
        year = int(bare_year.group(1))
        return Period(today.month, year, str(year),
                      start=date(year, 1, 1), end=date(year, 12, 31))

    return Period(today.month, today.year, "this month", assumed=True)


def in_words(label: str) -> str:
    """"in August 2026", but "today" and "this month" without the preposition."""
    bare = ("this ", "last ", "next ", "today", "yesterday", "tomorrow")
    return label if label.lower().startswith(bare) else f"in {label}"
