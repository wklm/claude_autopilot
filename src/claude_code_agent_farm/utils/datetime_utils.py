"""Utilities for consistent timezone-aware datetime handling.

This module provides helper functions to ensure all datetime objects
in the codebase are timezone-aware, preventing comparison errors between
naive and aware datetimes.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

# Default timezone for the application
UTC = ZoneInfo("UTC")


def now_utc() -> datetime:
    """Get the current time as a timezone-aware datetime in UTC.
    
    Returns:
        Timezone-aware datetime object in UTC
    """
    return datetime.now(UTC)


def make_aware(dt: datetime, timezone: ZoneInfo | None = None) -> datetime:
    """Convert a naive datetime to timezone-aware.
    
    Args:
        dt: Datetime object (naive or aware)
        timezone: Target timezone (defaults to UTC)
        
    Returns:
        Timezone-aware datetime object
        
    Note:
        If the datetime is already timezone-aware, it will be converted
        to the specified timezone. If naive, it's assumed to be in the
        target timezone.
    """
    if timezone is None:
        timezone = UTC
        
    if dt.tzinfo is None:
        # Naive datetime - assume it's in the target timezone
        return dt.replace(tzinfo=timezone)
    else:
        # Already aware - convert to target timezone
        return dt.astimezone(timezone)


def ensure_aware(dt: datetime | None) -> datetime | None:
    """Ensure a datetime is timezone-aware, converting if necessary.
    
    Args:
        dt: Datetime object or None
        
    Returns:
        Timezone-aware datetime or None if input is None
    """
    if dt is None:
        return None
    return make_aware(dt)


def is_aware(dt: datetime) -> bool:
    """Check if a datetime object is timezone-aware.
    
    Args:
        dt: Datetime object to check
        
    Returns:
        True if timezone-aware, False if naive
    """
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None
