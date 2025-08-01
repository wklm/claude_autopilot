"""Watchdog timer for detecting hung agents."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WatchdogState(str, Enum):
    """Watchdog timer states."""

    ACTIVE = "active"
    TRIGGERED = "triggered"
    PAUSED = "paused"
    STOPPED = "stopped"


class ActivityType(str, Enum):
    """Types of agent activity."""

    PANE_OUTPUT = "pane_output"
    STATUS_CHANGE = "status_change"
    USER_INPUT = "user_input"
    TASK_PROGRESS = "task_progress"
    HEARTBEAT = "heartbeat"


class AgentActivity(BaseModel):
    """Record of agent activity."""

    activity_type: ActivityType
    timestamp: datetime = Field(default_factory=datetime.now)
    description: str
    details: dict[str, Any] = Field(default_factory=dict)


class WatchdogTimer(BaseModel):
    """Watchdog timer for detecting hung or unresponsive agents."""

    # Configuration
    timeout_seconds: int = Field(default=300, ge=30, description="Timeout before considering agent hung")
    grace_period_seconds: int = Field(default=60, ge=0, description="Grace period after restart")
    check_interval_seconds: int = Field(default=10, ge=1, description="How often to check for timeout")

    # State
    state: WatchdogState = Field(default=WatchdogState.STOPPED)
    last_activity: datetime | None = Field(default=None)
    last_check: datetime | None = Field(default=None)
    timer_started: datetime | None = Field(default=None)

    # Activity tracking
    activity_log: list[AgentActivity] = Field(default_factory=list)
    trigger_count: int = Field(default=0)
    last_trigger: datetime | None = Field(default=None)

    # Content tracking for detecting real changes
    last_pane_content_hash: str | None = Field(default=None)
    last_meaningful_change: datetime | None = Field(default=None)

    def start(self) -> None:
        """Start the watchdog timer."""
        self.state = WatchdogState.ACTIVE
        self.timer_started = datetime.now()
        self.last_activity = datetime.now()
        self.last_check = datetime.now()

    def stop(self) -> None:
        """Stop the watchdog timer."""
        self.state = WatchdogState.STOPPED
        self.timer_started = None

    def pause(self) -> None:
        """Pause the watchdog timer."""
        if self.state == WatchdogState.ACTIVE:
            self.state = WatchdogState.PAUSED

    def resume(self) -> None:
        """Resume the watchdog timer."""
        if self.state == WatchdogState.PAUSED:
            self.state = WatchdogState.ACTIVE
            self.last_activity = datetime.now()  # Reset to avoid immediate trigger

    def feed(self, activity_type: ActivityType, description: str, **details) -> None:
        """Feed the watchdog to indicate agent activity."""
        if self.state != WatchdogState.ACTIVE:
            return

        activity = AgentActivity(activity_type=activity_type, description=description, details=details)

        self.activity_log.append(activity)
        self.last_activity = activity.timestamp

        # Maintain log size
        if len(self.activity_log) > 1000:
            self.activity_log = self.activity_log[-500:]

    def check_timeout(self) -> tuple[bool, str | None]:
        """Check if the watchdog has timed out."""
        if self.state != WatchdogState.ACTIVE:
            return False, None

        now = datetime.now()
        self.last_check = now

        # Check if in grace period
        if self.timer_started and (now - self.timer_started).total_seconds() < self.grace_period_seconds:
            return False, None

        # Check for timeout
        if self.last_activity:
            time_since_activity = (now - self.last_activity).total_seconds()
            if time_since_activity > self.timeout_seconds:
                self.state = WatchdogState.TRIGGERED
                self.trigger_count += 1
                self.last_trigger = now
                return True, f"No activity for {int(time_since_activity)} seconds"

        return False, None

    def is_content_changed(self, content: str) -> bool:
        """Check if pane content has meaningfully changed."""
        import hashlib

        # Calculate hash of content
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # Check if content changed
        if self.last_pane_content_hash != content_hash:
            self.last_pane_content_hash = content_hash
            self.last_meaningful_change = datetime.now()
            return True

        return False

    def get_time_since_activity(self) -> timedelta | None:
        """Get time since last activity."""
        if self.last_activity:
            return datetime.now() - self.last_activity
        return None

    def get_activity_summary(self, last_n_minutes: int = 5) -> dict[ActivityType, int]:
        """Get summary of recent activity."""
        cutoff_time = datetime.now() - timedelta(minutes=last_n_minutes)
        recent_activities = [a for a in self.activity_log if a.timestamp > cutoff_time]

        summary = {}
        for activity in recent_activities:
            summary[activity.activity_type] = summary.get(activity.activity_type, 0) + 1

        return summary


class HungAgentDetector(BaseModel):
    """Detector for various types of hung agent scenarios."""

    # Detection thresholds
    no_output_threshold: int = Field(default=300, description="Seconds without output")
    stuck_pattern_threshold: int = Field(default=3, description="Repetitions to consider stuck")
    memory_growth_threshold: float = Field(default=0.5, description="Memory growth ratio threshold")

    # Pattern detection
    repeated_patterns: dict[str, int] = Field(default_factory=dict)
    last_memory_check: float | None = Field(default=None)

    def check_stuck_patterns(self, content: str) -> tuple[bool, str | None]:
        """Check for stuck/repeated patterns in output."""
        # Look for common stuck patterns
        stuck_indicators = [
            "Thinking...",
            "Processing...",
            "Loading...",
            "Waiting for",
            "Retrying",
        ]

        for pattern in stuck_indicators:
            if pattern in content:
                count = self.repeated_patterns.get(pattern, 0) + 1
                self.repeated_patterns[pattern] = count

                if count >= self.stuck_pattern_threshold:
                    return True, f"Stuck on '{pattern}' (repeated {count} times)"

        # Clear counts for patterns not in current content
        for pattern in list(self.repeated_patterns.keys()):
            if pattern not in content:
                del self.repeated_patterns[pattern]

        return False, None
