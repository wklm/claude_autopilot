"""Event models for Claude Flutter Agent."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

from claude_code_agent_farm.models_new.base import EventModel
from claude_code_agent_farm.models_new.time import UsageLimitTimeInfo


class EventType(str, Enum):
    """Types of events in the system."""

    # Agent lifecycle events
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_RESTARTED = "agent_restarted"
    AGENT_CRASHED = "agent_crashed"

    # Status change events
    STATUS_CHANGED = "status_changed"
    AGENT_IDLE = "agent_idle"
    AGENT_WORKING = "agent_working"
    AGENT_READY = "agent_ready"
    AGENT_ERROR = "agent_error"

    # Usage limit events
    USAGE_LIMIT_HIT = "usage_limit_hit"
    USAGE_LIMIT_WAIT_START = "usage_limit_wait_start"
    USAGE_LIMIT_WAIT_END = "usage_limit_wait_end"

    # Command events
    COMMAND_SENT = "command_sent"
    PROMPT_SENT = "prompt_sent"

    # System events
    MONITOR_STARTED = "monitor_started"
    MONITOR_STOPPED = "monitor_stopped"
    CONFIG_CHANGED = "config_changed"
    ERROR_OCCURRED = "error_occurred"


class AgentLifecycleEvent(EventModel):
    """Event for agent lifecycle changes."""

    event_type: EventType = Field(..., description="Type of lifecycle event")
    session_id: str = Field(..., description="Agent session ID")
    restart_count: int = Field(default=0, ge=0, description="Number of restarts")
    reason: str | None = Field(default=None, description="Reason for the event")

    @classmethod
    def started(cls, session_id: str) -> "AgentLifecycleEvent":
        """Create an agent started event."""
        return cls(
            event_type=EventType.AGENT_STARTED,
            session_id=session_id,
            source="monitor",
        )

    @classmethod
    def stopped(cls, session_id: str, reason: str = "User requested") -> "AgentLifecycleEvent":
        """Create an agent stopped event."""
        return cls(
            event_type=EventType.AGENT_STOPPED,
            session_id=session_id,
            reason=reason,
            source="monitor",
        )

    @classmethod
    def restarted(cls, session_id: str, restart_count: int, reason: str) -> "AgentLifecycleEvent":
        """Create an agent restarted event."""
        return cls(
            event_type=EventType.AGENT_RESTARTED,
            session_id=session_id,
            restart_count=restart_count,
            reason=reason,
            source="monitor",
        )


class StatusChangeEvent(EventModel):
    """Event for agent status changes."""

    event_type: Literal[EventType.STATUS_CHANGED] = EventType.STATUS_CHANGED
    previous_status: str = Field(..., description="Previous status")
    new_status: str = Field(..., description="New status")
    context_percentage: int | None = Field(default=None, ge=0, le=100)
    idle_seconds: float | None = Field(default=None, ge=0)

    @computed_field
    @property
    def is_error_transition(self) -> bool:
        """Check if this is a transition to error state."""
        return self.new_status == "error"

    @computed_field
    @property
    def is_recovery(self) -> bool:
        """Check if this is a recovery from error state."""
        return self.previous_status == "error" and self.new_status != "error"


class UsageLimitEvent(EventModel):
    """Event for usage limit occurrences."""

    event_type: EventType = Field(..., description="Usage limit event type")
    limit_info: UsageLimitTimeInfo = Field(..., description="Parsed usage limit information")
    wait_seconds: float = Field(default=0.0, ge=0, description="Seconds to wait")
    is_waiting: bool = Field(default=False, description="Whether currently waiting")

    @classmethod
    def hit(cls, limit_info: UsageLimitTimeInfo) -> "UsageLimitEvent":
        """Create a usage limit hit event."""
        return cls(
            event_type=EventType.USAGE_LIMIT_HIT,
            limit_info=limit_info,
            wait_seconds=limit_info.wait_seconds,
            source="monitor",
        )

    @classmethod
    def wait_start(cls, limit_info: UsageLimitTimeInfo) -> "UsageLimitEvent":
        """Create a wait start event."""
        return cls(
            event_type=EventType.USAGE_LIMIT_WAIT_START,
            limit_info=limit_info,
            wait_seconds=limit_info.wait_seconds,
            is_waiting=True,
            source="monitor",
        )

    @classmethod
    def wait_end(cls, limit_info: UsageLimitTimeInfo) -> "UsageLimitEvent":
        """Create a wait end event."""
        return cls(
            event_type=EventType.USAGE_LIMIT_WAIT_END,
            limit_info=limit_info,
            wait_seconds=0,
            is_waiting=False,
            source="monitor",
        )


class CommandEvent(EventModel):
    """Event for command executions."""

    event_type: EventType = Field(..., description="Command event type")
    command: str = Field(..., description="The command that was sent")
    target: str | None = Field(default=None, description="Target (e.g., tmux session)")
    success: bool = Field(default=True, description="Whether command succeeded")
    error_message: str | None = Field(default=None)

    @classmethod
    def prompt_sent(cls, prompt: str, session: str) -> "CommandEvent":
        """Create a prompt sent event."""
        # Truncate long prompts for events
        display_prompt = prompt[:200] + "..." if len(prompt) > 200 else prompt

        return cls(
            event_type=EventType.PROMPT_SENT,
            command=display_prompt,
            target=session,
            source="monitor",
        )

    @classmethod
    def command_sent(cls, command: str, target: str, success: bool = True) -> "CommandEvent":
        """Create a command sent event."""
        return cls(
            event_type=EventType.COMMAND_SENT,
            command=command,
            target=target,
            success=success,
            source="monitor",
        )


class ErrorEvent(EventModel):
    """Event for errors and exceptions."""

    event_type: Literal[EventType.ERROR_OCCURRED] = EventType.ERROR_OCCURRED
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    error_details: dict[str, Any] | None = Field(default=None)
    recoverable: bool = Field(default=True, description="Whether error is recoverable")
    stack_trace: str | None = Field(default=None)

    @classmethod
    def from_exception(cls, exc: Exception, recoverable: bool = True) -> "ErrorEvent":
        """Create from an exception."""
        import traceback

        return cls(
            error_type=type(exc).__name__,
            error_message=str(exc),
            recoverable=recoverable,
            stack_trace=traceback.format_exc(),
            source="system",
        )


class SystemEvent(EventModel):
    """Event for system-level occurrences."""

    event_type: EventType = Field(..., description="System event type")
    details: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def monitor_started(cls, config: dict[str, Any]) -> "SystemEvent":
        """Create a monitor started event."""
        return cls(
            event_type=EventType.MONITOR_STARTED,
            details={"config": config},
            source="system",
        )

    @classmethod
    def monitor_stopped(cls, runtime_seconds: float, stats: dict[str, Any]) -> "SystemEvent":
        """Create a monitor stopped event."""
        return cls(
            event_type=EventType.MONITOR_STOPPED,
            details={
                "runtime_seconds": runtime_seconds,
                "stats": stats,
            },
            source="system",
        )

    @classmethod
    def config_changed(cls, changes: dict[str, Any]) -> "SystemEvent":
        """Create a config changed event."""
        return cls(
            event_type=EventType.CONFIG_CHANGED,
            details={"changes": changes},
            source="system",
        )


class EventStore(BaseModel):
    """Store and manage events."""

    events: list[EventModel] = Field(default_factory=list)
    max_events: int = Field(default=1000, ge=100, description="Maximum events to keep")

    def add(self, event: EventModel) -> None:
        """Add an event to the store."""
        self.events.append(event)

        # Trim to max size, keeping newest
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

    def get_recent(self, count: int = 20) -> list[EventModel]:
        """Get recent events."""
        return self.events[-count:]

    def get_by_type(self, event_type: EventType) -> list[EventModel]:
        """Get events by type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_errors(self) -> list[ErrorEvent]:
        """Get error events."""
        return [e for e in self.events if isinstance(e, ErrorEvent)]

    def get_since(self, timestamp: datetime) -> list[EventModel]:
        """Get events since a timestamp."""
        return [e for e in self.events if e.created_at >= timestamp]

    @computed_field
    @property
    def total_count(self) -> int:
        """Get total event count."""
        return len(self.events)

    @computed_field
    @property
    def event_counts(self) -> dict[str, int]:
        """Get counts by event type."""
        counts = {}
        for event in self.events:
            event_type = str(event.event_type)
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts

    def to_timeline(self) -> list[dict[str, Any]]:
        """Convert to timeline format for display."""
        timeline = []
        for event in self.events:
            timeline.append(
                {
                    "timestamp": event.created_at.isoformat(),
                    "type": str(event.event_type),
                    "source": event.source,
                    "summary": self._summarize_event(event),
                },
            )
        return timeline

    def _summarize_event(self, event: EventModel) -> str:
        """Create a summary of an event."""
        if isinstance(event, StatusChangeEvent):
            return f"{event.previous_status} → {event.new_status}"
        if isinstance(event, UsageLimitEvent):
            return f"Wait {event.limit_info.format_wait_duration()}"
        if isinstance(event, CommandEvent):
            return f"Command: {event.command[:50]}..."
        if isinstance(event, ErrorEvent):
            return f"Error: {event.error_message}"
        return str(event.event_type)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of events for checkpointing."""
        return {
            "total_events": self.total_count,
            "event_counts": self.event_counts,
            "recent_events": [
                {
                    "timestamp": e.created_at.isoformat(),
                    "type": str(e.event_type),
                    "summary": self._summarize_event(e),
                }
                for e in self.get_recent(10)
            ],
        }
