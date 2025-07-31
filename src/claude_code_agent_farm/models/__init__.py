"""Pydantic models for Claude Single Agent Monitor.

This module provides strongly-typed models for all data structures
used in the single agent monitoring system.
"""

from claude_code_agent_farm.models.base import (
    CommandModel,
    EventModel,
    MetricModel,
    SerializableModel,
    StateModel,
    TimestampedModel,
    ValidatedPathModel,
)
from claude_code_agent_farm.models.commands import (
    ClaudeCommand,
    CommandExecution,
    CommandHistory,
    CommandStatus,
    CommandType,
    ShellCommand,
    TmuxCommand,
)
from claude_code_agent_farm.models.events import (
    AgentLifecycleEvent,
    CommandEvent,
    ErrorEvent,
    EventStore,
    EventType,
    StatusChangeEvent,
    SystemEvent,
    UsageLimitEvent,
)
from claude_code_agent_farm.models.time import (
    ParsedTime,
    TimePatternType,
    TimeRange,
    TimezoneMapping,
    UsageLimitTimeInfo,
)

# Legacy models still in models.py - will be migrated
from claude_code_agent_farm.models import (
    AgentConfig,
    AgentFarmConfig,
    AgentFarmResult,
    AgentResult,
    AgentSession,
    AgentStatus,
    SingleAgentConfig,
    UsageLimitInfo,
)

__all__ = [
    # Base models
    "CommandModel",
    "EventModel",
    "MetricModel",
    "SerializableModel",
    "StateModel",
    "TimestampedModel",
    "ValidatedPathModel",
    # Command models
    "ClaudeCommand",
    "CommandExecution",
    "CommandHistory",
    "CommandStatus",
    "CommandType",
    "ShellCommand",
    "TmuxCommand",
    # Event models
    "AgentLifecycleEvent",
    "CommandEvent",
    "ErrorEvent",
    "EventStore",
    "EventType",
    "StatusChangeEvent",
    "SystemEvent",
    "UsageLimitEvent",
    # Time models
    "ParsedTime",
    "TimePatternType",
    "TimeRange",
    "TimezoneMapping",
    "UsageLimitTimeInfo",
    # Legacy models
    "AgentConfig",
    "AgentFarmConfig",
    "AgentFarmResult",
    "AgentResult",
    "AgentSession",
    "AgentStatus",
    "SingleAgentConfig",
    "UsageLimitInfo",
]