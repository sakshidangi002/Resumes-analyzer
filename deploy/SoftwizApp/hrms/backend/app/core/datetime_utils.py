
from datetime import datetime, timedelta, timezone

def get_ist_now():
    """
    Returns the current datetime in Indian Standard Time (IST).
    IST is UTC + 5:30.
    """
    # Using UTC as a base and adding 5.5 hours
    return datetime.utcnow() + timedelta(hours=5, minutes=30)
