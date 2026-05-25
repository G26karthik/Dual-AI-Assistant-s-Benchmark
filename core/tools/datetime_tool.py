from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def get_datetime(timezone: str = "UTC") -> str:
    """Return current date/time in the requested timezone."""
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        return f"Invalid timezone '{timezone}'. Example: UTC, Asia/Kolkata."

    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S (%A) %Z")
