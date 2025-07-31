"""Utility for parsing time information from Claude usage limit messages.

Refactored to use Pydantic models for strongly-typed results.
"""

import re
from datetime import datetime, time, timedelta
from typing import Optional, Tuple, List
from zoneinfo import ZoneInfo

from claude_code_agent_farm.models.time import (
    ParsedTime,
    TimePatternType,
    TimezoneMapping,
    UsageLimitTimeInfo,
)


class UsageLimitTimeParser:
    """Parse usage limit retry times from Claude messages.
    
    This parser returns structured Pydantic models instead of raw datetime objects,
    providing better type safety and validation.
    """
    
    # Timezone mappings as Pydantic models
    TIMEZONE_MAPPINGS: List[TimezoneMapping] = [
        TimezoneMapping(abbreviation="PST", iana_name="America/Los_Angeles", utc_offset="-08:00", is_dst=False),
        TimezoneMapping(abbreviation="PDT", iana_name="America/Los_Angeles", utc_offset="-07:00", is_dst=True),
        TimezoneMapping(abbreviation="EST", iana_name="America/New_York", utc_offset="-05:00", is_dst=False),
        TimezoneMapping(abbreviation="EDT", iana_name="America/New_York", utc_offset="-04:00", is_dst=True),
        TimezoneMapping(abbreviation="CST", iana_name="America/Chicago", utc_offset="-06:00", is_dst=False),
        TimezoneMapping(abbreviation="CDT", iana_name="America/Chicago", utc_offset="-05:00", is_dst=True),
        TimezoneMapping(abbreviation="MST", iana_name="America/Denver", utc_offset="-07:00", is_dst=False),
        TimezoneMapping(abbreviation="MDT", iana_name="America/Denver", utc_offset="-06:00", is_dst=True),
        TimezoneMapping(abbreviation="UTC", iana_name="UTC", utc_offset="+00:00", is_dst=False),
        TimezoneMapping(abbreviation="GMT", iana_name="UTC", utc_offset="+00:00", is_dst=False),
    ]
    
    # Quick lookup dict for timezone mappings
    TIMEZONE_MAP = {tz.abbreviation: tz.iana_name for tz in TIMEZONE_MAPPINGS}
    
    # Patterns for different time formats Claude might use
    TIME_PATTERNS = [
        # "Try again at 3:00 PM PST"
        r'(?:try again|available|retry) (?:at|after) (\d{1,2}):(\d{2})\s*(AM|PM)?\s*([A-Z]{3,4})?',
        # "Usage resets at midnight UTC"
        r'(?:resets?|available) at (midnight|noon)\s*([A-Z]{3,4})?',
        # "Available in 2 hours"
        r'(?:available|try again) in (\d+)\s*(hours?|minutes?)',
        # "Usage limit reached until 15:00"
        r'(?:until|after) (\d{1,2}):(\d{2})',
        # ISO format: "2024-01-15T15:00:00Z"
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2}))',
    ]
    
    def parse_usage_limit_message(self, message: str) -> UsageLimitTimeInfo:
        """Parse a usage limit message and return structured time information.
        
        Args:
            message: The message from Claude containing usage limit info
            
        Returns:
            UsageLimitTimeInfo with parsed results and retry time
        """
        parse_results: List[ParsedTime] = []
        best_result: Optional[ParsedTime] = None
        
        # Try each pattern
        for pattern_idx, pattern in enumerate(self.TIME_PATTERNS):
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                parsed = self._parse_match(match, pattern, pattern_idx, message)
                if parsed:
                    parse_results.append(parsed)
                    # Use first successful parse as best result
                    if not best_result and parsed.parsed_datetime:
                        best_result = parsed
        
        # Calculate retry time and wait duration from best result
        retry_time: Optional[datetime] = None
        wait_duration: Optional[timedelta] = None
        
        if best_result and best_result.parsed_datetime:
            retry_time = best_result.parsed_datetime
            
            # Calculate wait duration
            now = datetime.now(retry_time.tzinfo or ZoneInfo('UTC'))
            wait_duration = retry_time - now
            
            # If negative, assume tomorrow
            if wait_duration.total_seconds() < 0:
                wait_duration += timedelta(days=1)
                retry_time += timedelta(days=1)
        
        # If no patterns matched, create default result
        if not retry_time:
            return UsageLimitTimeInfo.create_default(message, wait_hours=1)
        
        return UsageLimitTimeInfo(
            message=message,
            parse_results=parse_results,
            retry_time=retry_time,
            wait_duration=wait_duration,
            timezone_used=best_result.timezone or "UTC" if best_result else "UTC",
        )
    
    def _parse_match(self, match: re.Match, pattern: str, pattern_idx: int, original_text: str) -> Optional[ParsedTime]:
        """Parse a regex match based on the pattern type.
        
        Returns:
            ParsedTime model with structured parsing results
        """
        now = datetime.now(ZoneInfo('UTC'))
        
        # Determine pattern type
        pattern_type = TimePatternType.UNKNOWN
        if 'AM|PM' in pattern:
            pattern_type = TimePatternType.TIME_WITH_MERIDIEM
        elif 'midnight|noon' in pattern:
            pattern_type = TimePatternType.SPECIAL_TIME
        elif 'hours?|minutes?' in pattern:
            pattern_type = TimePatternType.RELATIVE_TIME
        elif r'\d{4}-\d{2}-\d{2}T' in pattern:
            pattern_type = TimePatternType.ISO_FORMAT
        elif r'(\d{1,2}):(\d{2})' in pattern:
            pattern_type = TimePatternType.ABSOLUTE_TIME
        
        # Pattern: "Try again at 3:00 PM PST"
        if pattern_type == TimePatternType.TIME_WITH_MERIDIEM:
            hour = int(match.group(1))
            minute = int(match.group(2))
            meridiem = match.group(3) or 'AM'
            tz_abbr = match.group(4) or 'UTC'
            
            # Convert to 24-hour format
            display_hour = hour
            if meridiem.upper() == 'PM' and hour != 12:
                hour += 12
            elif meridiem.upper() == 'AM' and hour == 12:
                hour = 0
                
            # Get timezone
            tz_name = self.TIMEZONE_MAP.get(tz_abbr.upper(), 'UTC')
            tz = ZoneInfo(tz_name)
            
            # Create datetime for today in the specified timezone
            target_time = datetime.now(tz).replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            
            # If the time has already passed today, assume tomorrow
            if target_time <= datetime.now(tz):
                target_time += timedelta(days=1)
                
            return ParsedTime(
                original_text=original_text,
                matched_pattern=pattern,
                pattern_type=pattern_type,
                hour=hour,
                minute=minute,
                meridiem=meridiem.upper(),
                timezone_str=tz_abbr,
                timezone=tz_name,
                parsed_datetime=target_time,
                confidence=0.95,
            )
            
        # Pattern: "resets at midnight UTC"
        elif pattern_type == TimePatternType.SPECIAL_TIME:
            time_word = match.group(1).lower()
            tz_abbr = match.group(2) or 'UTC'
            
            hour = 0 if time_word == 'midnight' else 12
            
            tz_name = self.TIMEZONE_MAP.get(tz_abbr.upper(), 'UTC')
            tz = ZoneInfo(tz_name)
            
            # Create datetime for today in the specified timezone
            target_time = datetime.now(tz).replace(
                hour=hour, minute=0, second=0, microsecond=0
            )
            
            # If the time has already passed today, assume tomorrow
            if target_time <= datetime.now(tz):
                target_time += timedelta(days=1)
                
            return ParsedTime(
                original_text=original_text,
                matched_pattern=pattern,
                pattern_type=pattern_type,
                hour=hour,
                minute=0,
                timezone_str=tz_abbr,
                timezone=tz_name,
                parsed_datetime=target_time,
                confidence=0.95,
            )
            
        # Pattern: "Available in 2 hours"
        elif pattern_type == TimePatternType.RELATIVE_TIME:
            amount = int(match.group(1))
            unit = match.group(2).lower()
            
            if 'hour' in unit:
                target_time = now + timedelta(hours=amount)
            else:  # minutes
                target_time = now + timedelta(minutes=amount)
                
            return ParsedTime(
                original_text=original_text,
                matched_pattern=pattern,
                pattern_type=pattern_type,
                relative_amount=amount,
                relative_unit=unit,
                parsed_datetime=target_time,
                timezone="UTC",
                confidence=0.9,
            )
                
        # Pattern: "until 15:00"
        elif pattern_type == TimePatternType.ABSOLUTE_TIME:
            hour = int(match.group(1))
            minute = int(match.group(2))
            
            # Assume local timezone
            local_tz = datetime.now().astimezone().tzinfo
            target_time = datetime.now(local_tz).replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            
            # If the time has already passed today, assume tomorrow
            if target_time <= datetime.now(local_tz):
                target_time += timedelta(days=1)
                
            return ParsedTime(
                original_text=original_text,
                matched_pattern=pattern,
                pattern_type=pattern_type,
                hour=hour,
                minute=minute,
                parsed_datetime=target_time,
                timezone=str(local_tz),
                confidence=0.85,
            )
            
        # ISO format
        elif pattern_type == TimePatternType.ISO_FORMAT:
            try:
                iso_string = match.group(1)
                target_time = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
                
                return ParsedTime(
                    original_text=original_text,
                    matched_pattern=pattern,
                    pattern_type=pattern_type,
                    parsed_datetime=target_time,
                    timezone=str(target_time.tzinfo) if target_time.tzinfo else "UTC",
                    confidence=1.0,
                )
            except ValueError:
                return None
                
        return None
    
    def get_wait_duration(self, message: str) -> Optional[timedelta]:
        """Get the duration to wait based on the usage limit message.
        
        This is a convenience method that wraps parse_usage_limit_message.
        
        Args:
            message: The message from Claude containing usage limit info
            
        Returns:
            Timedelta to wait, or None if not found
        """
        result = self.parse_usage_limit_message(message)
        return result.wait_duration if result.has_valid_time else None
    
    def format_retry_time(self, retry_time: datetime) -> str:
        """Format a retry time for display.
        
        This is a convenience method for backward compatibility.
        Prefer using UsageLimitTimeInfo.format_retry_time() directly.
        
        Args:
            retry_time: The datetime when retry is available
            
        Returns:
            Formatted string for display
        """
        info = UsageLimitTimeInfo(
            message="",
            retry_time=retry_time,
            wait_duration=timedelta(seconds=0),
        )
        return info.format_retry_time()
    
    def get_timezone_info(self, abbreviation: str) -> Optional[TimezoneMapping]:
        """Get timezone mapping information.
        
        Args:
            abbreviation: Timezone abbreviation (e.g., 'PST')
            
        Returns:
            TimezoneMapping model or None if not found
        """
        abbr_upper = abbreviation.upper()
        for tz_map in self.TIMEZONE_MAPPINGS:
            if tz_map.abbreviation == abbr_upper:
                return tz_map
        return None