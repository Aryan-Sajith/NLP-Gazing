"""Timestamp parsing and conversion utilities"""

from datetime import datetime, timezone
from typing import Optional


def parse_utc_timestamp(timestamp_str: str) -> Optional[int]:
    """
    Parse UTC timestamp to Unix milliseconds.
    
    The query_timestamp field in query_logs_table.csv is in UTC.
    We explicitly mark it as UTC to ensure correct conversion regardless
    of the system timezone where this script runs.
    
    Args:
        timestamp_str: Timestamp in format 'YYYY-MM-DD HH:MM:SS'
    
    Returns:
        Unix timestamp in milliseconds, or None if parsing fails
    """
    try:
        dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        dt_utc = dt.replace(tzinfo=timezone.utc)
        return int(dt_utc.timestamp() * 1000)
    except Exception as e:
        print(f"Error parsing timestamp {timestamp_str}: {e}")
        return None
