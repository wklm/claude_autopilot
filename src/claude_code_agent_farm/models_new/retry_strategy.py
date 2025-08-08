"""Retry strategy models for handling usage limits with exponential backoff."""

import random
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from claude_code_agent_farm.utils import now_utc


class RetryStrategy(BaseModel):
    """Configuration for retry behavior with exponential backoff."""

    # Base configuration
    initial_delay_seconds: int = Field(default=60, ge=1, description="Initial retry delay in seconds")
    max_delay_seconds: int = Field(default=3600, ge=60, description="Maximum retry delay in seconds")
    backoff_factor: float = Field(default=2.0, ge=1.0, description="Exponential backoff multiplier")
    jitter_factor: float = Field(default=0.1, ge=0.0, le=1.0, description="Random jitter factor (0-1)")

    # Retry limits
    max_retry_attempts: int = Field(default=10, ge=1, description="Maximum number of retry attempts")
    retry_window_hours: int = Field(default=24, ge=1, description="Time window for retry attempts")

    # Current state
    current_attempt: int = Field(default=0, ge=0)
    last_retry_time: datetime | None = Field(default=None)
    retry_history: list[datetime] = Field(default_factory=list)

    def calculate_next_delay(self) -> int:
        """Calculate the next retry delay with exponential backoff and jitter."""
        if self.current_attempt == 0:
            base_delay = self.initial_delay_seconds
        else:
            base_delay = min(
                self.initial_delay_seconds * (self.backoff_factor**self.current_attempt), self.max_delay_seconds,
            )

        # Add random jitter to prevent thundering herd
        jitter_range = base_delay * self.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)

        return max(1, int(base_delay + jitter))

    def should_retry(self) -> bool:
        """Check if we should attempt another retry."""
        if self.current_attempt >= self.max_retry_attempts:
            return False

        # Check if we're within the retry window
        if self.retry_history:
            oldest_retry = self.retry_history[0]
            if now_utc() - oldest_retry > timedelta(hours=self.retry_window_hours):
                # Reset if outside window
                self.reset()

        return True

    def record_retry_attempt(self) -> None:
        """Record a retry attempt."""
        now = now_utc()
        self.current_attempt += 1
        self.last_retry_time = now
        self.retry_history.append(now)

        # Maintain window of retry history
        cutoff_time = now - timedelta(hours=self.retry_window_hours)
        self.retry_history = [t for t in self.retry_history if t > cutoff_time]

    def reset(self) -> None:
        """Reset retry state."""
        self.current_attempt = 0
        self.last_retry_time = None
        self.retry_history = []

    def get_next_retry_time(self) -> datetime:
        """Get the next retry time based on current state."""
        delay_seconds = self.calculate_next_delay()
        return now_utc() + timedelta(seconds=delay_seconds)


class UsageLimitRetryInfo(BaseModel):
    """Enhanced usage limit info with retry strategy."""

    message: str = Field(..., description="The usage limit message from Claude")
    detected_at: datetime = Field(default_factory=now_utc)

    # Parsed retry time from message (if available)
    parsed_retry_time: datetime | None = Field(default=None)

    # Calculated retry time using strategy
    calculated_retry_time: datetime | None = Field(default=None)

    # Retry strategy
    retry_strategy: RetryStrategy = Field(default_factory=RetryStrategy)

    # Fallback behavior
    use_fallback: bool = Field(default=False, description="Whether using fallback retry logic")
    fallback_reason: str | None = Field(default=None)

    @property
    def effective_retry_time(self) -> datetime:
        """Get the effective retry time (parsed or calculated)."""
        if self.parsed_retry_time and not self.use_fallback:
            return self.parsed_retry_time

        if self.calculated_retry_time:
            return self.calculated_retry_time

        # Final fallback: use strategy to calculate
        return self.retry_strategy.get_next_retry_time()

    def set_parsed_time(self, retry_time: datetime | None) -> None:
        """Set the parsed retry time and determine if fallback is needed."""
        self.parsed_retry_time = retry_time

        if retry_time is None:
            self.use_fallback = True
            self.fallback_reason = "Could not parse retry time from message"
            self.calculated_retry_time = self.retry_strategy.get_next_retry_time()
        elif retry_time < now_utc():
            self.use_fallback = True
            self.fallback_reason = "Parsed time is in the past"
            self.calculated_retry_time = self.retry_strategy.get_next_retry_time()
        else:
            self.use_fallback = False
            self.fallback_reason = None
