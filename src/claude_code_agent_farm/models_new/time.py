"""Time-related Pydantic models for usage limit parsing."""

from datetime import datetime, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, computed_field, field_validator

from claude_code_agent_farm.models_new.base import SerializableModel


class TimePatternType(str, Enum):
    """Types of time patterns we can parse."""

    TIME_WITH_MERIDIEM = "time_with_meridiem"  # "3:00 PM PST"
    SPECIAL_TIME = "special_time"  # "midnight UTC"
    RELATIVE_TIME = "relative_time"  # "in 2 hours"
    ABSOLUTE_TIME = "absolute_time"  # "15:00"
    ISO_FORMAT = "iso_format"  # "2024-01-15T15:00:00Z"
    UNKNOWN = "unknown"


class TimezoneMapping(BaseModel):
    """Timezone abbreviation to IANA timezone mapping."""

    abbreviation: str = Field(..., description="Timezone abbreviation (e.g., PST)")
    iana_name: str = Field(..., description="IANA timezone name (e.g., America/Los_Angeles)")
    utc_offset: str | None = Field(default=None, description="UTC offset (e.g., -08:00)")
    is_dst: bool = Field(default=False, description="Whether this is DST variant")

    @field_validator("abbreviation")
    @classmethod
    def uppercase_abbreviation(cls, v: str) -> str:
        """Ensure abbreviation is uppercase."""
        return v.upper()

    model_config = {
        "json_schema_extra": {
            "example": {
                "abbreviation": "PST",
                "iana_name": "America/Los_Angeles",
                "utc_offset": "-08:00",
                "is_dst": False,
            },
        },
    }


class ParsedTime(SerializableModel):
    """Result of parsing a time from text."""

    # Original input
    original_text: str = Field(..., description="Original text that was parsed")
    matched_pattern: str = Field(..., description="The pattern that matched")
    pattern_type: TimePatternType = Field(..., description="Type of pattern matched")

    # Parsed components
    hour: int | None = Field(default=None, ge=0, le=23)
    minute: int | None = Field(default=None, ge=0, le=59)
    meridiem: str | None = Field(default=None, pattern=r"^(AM|PM|am|pm)$")
    timezone_str: str | None = Field(default=None, description="Timezone string from input")
    timezone: str | None = Field(default=None, description="Resolved IANA timezone")

    # Relative time components
    relative_amount: int | None = Field(default=None, ge=0)
    relative_unit: str | None = Field(default=None, pattern=r"^(hours?|minutes?|days?)$")

    # Result
    parsed_datetime: datetime | None = Field(default=None, description="The parsed datetime")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Parsing confidence")

    @computed_field
    @property
    def is_relative(self) -> bool:
        """Check if this is a relative time."""
        return self.pattern_type == TimePatternType.RELATIVE_TIME

    @computed_field
    @property
    def is_absolute(self) -> bool:
        """Check if this is an absolute time."""
        return self.pattern_type in (
            TimePatternType.TIME_WITH_MERIDIEM,
            TimePatternType.ABSOLUTE_TIME,
            TimePatternType.ISO_FORMAT,
            TimePatternType.SPECIAL_TIME,
        )

    def to_datetime(self, base_time: datetime | None = None) -> datetime | None:
        """Convert to datetime, using base_time for relative times."""
        if self.parsed_datetime:
            return self.parsed_datetime

        if self.is_relative and base_time and self.relative_unit and self.relative_amount is not None:
            if "hour" in self.relative_unit:
                return base_time + timedelta(hours=self.relative_amount)
            if "minute" in self.relative_unit:
                return base_time + timedelta(minutes=self.relative_amount)
            if "day" in self.relative_unit:
                return base_time + timedelta(days=self.relative_amount)

        return None


class UsageLimitTimeInfo(SerializableModel):
    """Complete usage limit time information."""

    # Input
    message: str = Field(..., description="The usage limit message")
    parse_results: list[ParsedTime] = Field(default_factory=list, description="All parsed times")

    # Best result
    retry_time: datetime | None = Field(default=None, description="When to retry")
    wait_duration: timedelta | None = Field(default=None, description="How long to wait")

    # Metadata
    parsed_at: datetime = Field(default_factory=datetime.now)
    timezone_used: str = Field(default="UTC", description="Timezone used for calculations")

    @computed_field
    @property
    def has_valid_time(self) -> bool:
        """Check if we have a valid retry time."""
        return self.retry_time is not None

    @computed_field
    @property
    def wait_seconds(self) -> float:
        """Get wait duration in seconds."""
        if not self.wait_duration:
            return 0.0
        return max(0.0, self.wait_duration.total_seconds())

    @computed_field
    @property
    def wait_minutes(self) -> float:
        """Get wait duration in minutes."""
        return self.wait_seconds / 60.0

    @computed_field
    @property
    def wait_hours(self) -> float:
        """Get wait duration in hours."""
        return self.wait_seconds / 3600.0

    @computed_field
    @property
    def should_wait(self) -> bool:
        """Check if we should wait based on retry time."""
        if not self.retry_time:
            return False
        return self.retry_time > datetime.now(self.retry_time.tzinfo)

    def format_retry_time(self) -> str:
        """Format the retry time for display."""
        if not self.retry_time:
            return "Unknown"

        # Ensure timezone aware
        if not self.retry_time.tzinfo:
            retry_time = self.retry_time.replace(tzinfo=ZoneInfo("UTC"))
        else:
            retry_time = self.retry_time

        # Convert to local time
        local_time = retry_time.astimezone()
        now = datetime.now().astimezone()

        # Format based on when it is
        if local_time.date() == now.date():
            return f"today at {local_time.strftime('%I:%M %p %Z')}"
        if local_time.date() == (now + timedelta(days=1)).date():
            return f"tomorrow at {local_time.strftime('%I:%M %p %Z')}"
        return local_time.strftime("%Y-%m-%d %I:%M %p %Z")

    def format_wait_duration(self) -> str:
        """Format the wait duration for display."""
        if not self.wait_duration or self.wait_seconds <= 0:
            return "No wait required"

        hours = int(self.wait_hours)
        minutes = int((self.wait_seconds % 3600) / 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")

        return " and ".join(parts) if parts else "Less than a minute"

    @classmethod
    def create_default(cls, message: str, wait_hours: int = 1) -> "UsageLimitTimeInfo":
        """Create with a default wait time when parsing fails."""
        now = datetime.now(ZoneInfo("UTC"))
        retry_time = now + timedelta(hours=wait_hours)

        return cls(
            message=message,
            retry_time=retry_time,
            wait_duration=timedelta(hours=wait_hours),
            timezone_used="UTC",
        )


class TimeRange(BaseModel):
    """Model for a time range."""

    start: datetime = Field(..., description="Start time")
    end: datetime = Field(..., description="End time")

    @field_validator("end")
    @classmethod
    def validate_end_after_start(cls, v: datetime, info) -> datetime:
        """Ensure end time is after start time."""
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError("End time must be after start time")
        return v

    @computed_field
    @property
    def duration(self) -> timedelta:
        """Get the duration of the range."""
        return self.end - self.start

    def contains(self, time: datetime) -> bool:
        """Check if a time is within this range."""
        return self.start <= time <= self.end

    def overlaps(self, other: "TimeRange") -> bool:
        """Check if this range overlaps with another."""
        return self.start < other.end and other.start < self.end
