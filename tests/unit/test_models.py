"""Unit tests for Pydantic models."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from claude_code_agent_farm.models_new.session import (
    AgentSession,
    AgentStatus,
    UsageLimitInfo,
)
from claude_code_agent_farm.models_new.commands import (
    ClaudeCommand,
    CommandExecution,
    CommandHistory,
    CommandStatus,
    CommandType,
    TmuxCommand,
)
from claude_code_agent_farm.models_new.events import (
    AgentLifecycleEvent,
    CommandEvent,
    ErrorEvent,
    EventStore,
    EventType,
    StatusChangeEvent,
    UsageLimitEvent,
)
from claude_code_agent_farm.models_new.time import (
    ParsedTime,
    TimePatternType,
    TimezoneMapping,
    UsageLimitTimeInfo,
)


@pytest.mark.unit
class TestAgentModels:
    """Test core agent models."""

    def test_agent_status_enum(self):
        """Test AgentStatus enumeration."""
        assert AgentStatus.WORKING.value == "working"
        assert AgentStatus.READY.value == "ready"
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.ERROR.value == "error"
        assert AgentStatus.USAGE_LIMIT.value == "usage_limit"
        assert AgentStatus.STARTING.value == "starting"
        assert AgentStatus.UNKNOWN.value == "unknown"

    def test_usage_limit_info(self):
        """Test UsageLimitInfo model."""
        # Basic creation
        info = UsageLimitInfo(
            message="Usage limit reached",
            retry_time=datetime.now() + timedelta(hours=2),
            wait_duration=timedelta(hours=2),
        )

        assert info.has_valid_time is True
        assert info.wait_seconds > 0

        # Without retry time
        info_no_time = UsageLimitInfo(message="Limit reached")
        assert info_no_time.has_valid_time is False
        assert info_no_time.wait_seconds == 0.0

    def test_agent_session(self):
        """Test AgentSession model."""
        session = AgentSession(prompt="Test prompt for carenji")

        # Check defaults
        assert session.status == AgentStatus.STARTING
        assert session.total_runs == 0
        assert session.restart_count == 0
        assert session.usage_limit_hits == 0
        assert session.is_waiting_for_limit is False
        assert session.wait_until is None

        # Test runtime formatting
        runtime = session.runtime
        assert ":" in runtime  # Should be formatted as H:MM:SS

        # Test recording usage limit
        limit_info = UsageLimitInfo(message="Limit hit", retry_time=datetime.now() + timedelta(hours=1))
        session.record_usage_limit(limit_info)

        assert session.usage_limit_hits == 1
        assert session.is_waiting_for_limit is True
        assert session.wait_until is not None
        assert session.last_usage_limit == limit_info

        # Test recording restart
        session.record_restart()
        assert session.restart_count == 1
        assert session.last_activity > session.started_at


@pytest.mark.unit
class TestEventModels:
    """Test event tracking models."""

    def test_event_types(self):
        """Test EventType enumeration."""
        assert EventType.STATUS_CHANGE.value == "status_change"
        assert EventType.USAGE_LIMIT.value == "usage_limit"
        assert EventType.AGENT_START.value == "agent_start"
        assert EventType.COMMAND.value == "command"
        assert EventType.ERROR.value == "error"

    def test_status_change_event(self):
        """Test StatusChangeEvent."""
        event = StatusChangeEvent(previous_status="working", new_status="ready", source="monitor")

        assert event.event_type == EventType.STATUS_CHANGE
        assert event.previous_status == "working"
        assert event.new_status == "ready"

        # Test JSON serialization
        json_data = event.model_dump_json()
        assert "status_change" in json_data
        assert "working" in json_data
        assert "ready" in json_data

    def test_usage_limit_event(self):
        """Test UsageLimitEvent."""
        limit_info = UsageLimitTimeInfo(
            raw_message="Limit reached at 3PM PST",
            parsed_time=ParsedTime(hour=15, minute=0, timezone="PST", pattern_type=TimePatternType.SPECIFIC_TIME),
            retry_datetime=datetime.now() + timedelta(hours=2),
            wait_duration=timedelta(hours=2),
        )

        event = UsageLimitEvent(message="Usage limit reached", limit_info=limit_info)

        assert event.event_type == EventType.USAGE_LIMIT
        assert event.limit_info.raw_message == "Limit reached at 3PM PST"

    def test_agent_lifecycle_event(self):
        """Test AgentLifecycleEvent."""
        event = AgentLifecycleEvent(action="start", details={"prompt": "Test carenji", "session": "test-123"})

        assert event.event_type == EventType.AGENT_START
        assert event.action == "start"
        assert event.details["prompt"] == "Test carenji"

    def test_command_event(self):
        """Test CommandEvent."""
        event = CommandEvent(command_type="tmux", command="send-keys 'test' Enter", exit_code=0, duration=1.5)

        assert event.event_type == EventType.COMMAND
        assert event.command_type == "tmux"
        assert event.success is True  # exit_code == 0
        assert event.duration == 1.5

    def test_error_event(self):
        """Test ErrorEvent."""
        event = ErrorEvent(error_type="RuntimeError", message="Test error", details={"line": 42, "file": "test.py"})

        assert event.event_type == EventType.ERROR
        assert event.error_type == "RuntimeError"
        assert event.source == "monitor"  # default

    def test_event_store(self):
        """Test EventStore functionality."""
        store = EventStore()

        # Add events
        event1 = StatusChangeEvent(previous_status="starting", new_status="working", source="test")
        event2 = ErrorEvent(error_type="TestError", message="Test")

        store.add(event1)
        store.add(event2)

        # Check storage
        assert store.count() == 2
        assert len(store.get_all()) == 2

        # Get by type
        status_events = store.get_by_type(EventType.STATUS_CHANGE)
        assert len(status_events) == 1
        assert status_events[0] == event1

        error_events = store.get_by_type(EventType.ERROR)
        assert len(error_events) == 1
        assert error_events[0] == event2

        # Clear
        store.clear()
        assert store.count() == 0


@pytest.mark.unit
class TestCommandModels:
    """Test command execution models."""

    def test_command_types(self):
        """Test CommandType enumeration."""
        assert CommandType.TMUX.value == "tmux"
        assert CommandType.CLAUDE.value == "claude"
        assert CommandType.SHELL.value == "shell"

    def test_tmux_command(self):
        """Test TmuxCommand model."""
        # Basic command
        cmd = TmuxCommand(session="test-session", command="list-panes")

        shell_cmd = cmd.to_shell_command()
        assert shell_cmd == ["tmux", "list-panes", "-t", "test-session"]

        # Send keys command
        cmd = TmuxCommand(session="test-session", command="send-keys", args=["test text", "Enter"], send_keys=True)

        shell_cmd = cmd.to_shell_command()
        assert "send-keys" in shell_cmd
        assert "test text" in shell_cmd
        assert "Enter" in shell_cmd

    def test_claude_command(self):
        """Test ClaudeCommand model."""
        cmd = ClaudeCommand(
            prompt="Fix carenji errors", flags=["--dangerously-skip-permissions"], working_dir=Path("/workspace")
        )

        assert cmd.command_type == CommandType.CLAUDE
        assert cmd.prompt == "Fix carenji errors"
        assert "--dangerously-skip-permissions" in cmd.flags

    def test_command_execution(self):
        """Test CommandExecution tracking."""
        cmd = TmuxCommand(session="test", command="capture-pane")
        execution = CommandExecution(command=cmd)

        # Start execution
        execution.start()
        assert execution.status == CommandStatus.RUNNING
        assert execution.start_time is not None

        # Complete execution
        execution.complete(exit_code=0, stdout="output", stderr="")

        assert execution.status == CommandStatus.COMPLETED
        assert execution.exit_code == 0
        assert execution.stdout == "output"
        assert execution.end_time is not None
        assert execution.duration > 0

        # Test failure
        cmd2 = TmuxCommand(session="test", command="fail")
        execution2 = CommandExecution(command=cmd2)
        execution2.start()
        execution2.fail("Command failed")

        assert execution2.status == CommandStatus.FAILED
        assert execution2.error == "Command failed"

    def test_command_history(self):
        """Test CommandHistory tracking."""
        history = CommandHistory()

        # Add executions
        cmd1 = TmuxCommand(session="test", command="cmd1")
        exec1 = CommandExecution(command=cmd1)
        exec1.start()
        exec1.complete(0)

        cmd2 = ClaudeCommand(prompt="test")
        exec2 = CommandExecution(command=cmd2)
        exec2.start()
        exec2.fail("error")

        history.add(exec1)
        history.add(exec2)

        # Check history
        assert history.count() == 2

        # Get by status
        completed = history.get_by_status(CommandStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0] == exec1

        failed = history.get_by_status(CommandStatus.FAILED)
        assert len(failed) == 1
        assert failed[0] == exec2

        # Get by type
        tmux_cmds = history.get_by_type(CommandType.TMUX)
        assert len(tmux_cmds) == 1

        # Get last
        assert history.get_last() == exec2
        assert history.get_last_successful() == exec1


@pytest.mark.unit
class TestTimeModels:
    """Test time parsing models."""

    def test_parsed_time(self):
        """Test ParsedTime model."""
        parsed = ParsedTime(hour=15, minute=30, timezone="PST", pattern_type=TimePatternType.SPECIFIC_TIME)

        assert parsed.hour == 15
        assert parsed.minute == 30
        assert parsed.timezone == "PST"
        assert parsed.pattern_type == TimePatternType.SPECIFIC_TIME

        # Validation
        with pytest.raises(ValidationError):
            ParsedTime(hour=25, minute=0, timezone="PST")  # Invalid hour

        with pytest.raises(ValidationError):
            ParsedTime(hour=0, minute=60, timezone="PST")  # Invalid minute

    def test_usage_limit_time_info(self):
        """Test UsageLimitTimeInfo model."""
        parsed_time = ParsedTime(hour=14, minute=0, timezone="EST", pattern_type=TimePatternType.SPECIFIC_TIME)

        info = UsageLimitTimeInfo(
            raw_message="Try again at 2:00 PM EST",
            parsed_time=parsed_time,
            retry_datetime=datetime.now() + timedelta(hours=3),
            wait_duration=timedelta(hours=3),
        )

        assert info.raw_message == "Try again at 2:00 PM EST"
        assert info.parsed_time.hour == 14
        assert info.wait_duration == timedelta(hours=3)

    def test_timezone_mapping(self):
        """Test TimezoneMapping model."""
        mapping = TimezoneMapping(abbreviation="PST", iana_name="America/Los_Angeles", utc_offset=-8)

        assert mapping.abbreviation == "PST"
        assert mapping.iana_name == "America/Los_Angeles"
        assert mapping.utc_offset == -8
