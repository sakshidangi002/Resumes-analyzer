
from datetime import datetime, timedelta, timezone

def get_ist_now() -> datetime:
    """
    Returns the current datetime in Indian Standard Time (IST) as a timezone-naive datetime.
    IST is UTC + 5:30.

    Storing timezone-naive datetimes representing local time ensures that database 
    engines (like PostgreSQL) store the local hour exactly as-is without offset conversion.
    """
    return datetime.utcnow() + timedelta(hours=5, minutes=30)
