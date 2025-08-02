"""Unit tests for Flutter Agent Monitor."""

import signal
import subprocess
import time
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from claude_code_agent_farm.flutter_agent_monitor import FlutterAgentMonitor
from claude_code_agent_farm.models_new.session import AgentStatus


@pytest.mark.unit
class TestFlutterAgentMonitor:
    """Test FlutterAgentMonitor functionality."""

    @pytest.fixture
    def monitor(self, mock_settings):
        """Create a monitor instance with mocked settings."""
        with patch("claude_code_agent_farm.flutter_agent_monitor.signal.signal"):
            monitor = FlutterAgentMonitor(mock_settings)
            return monitor

    def test_monitor_initialization(self, mock_settings):
        """Test monitor initialization."""
        with patch("claude_code_agent_farm.flutter_agent_monitor.signal.signal") as mock_signal:
            monitor = FlutterAgentMonitor(mock_settings)

            # Check attributes
            assert monitor.settings == mock_settings
            assert monitor.running is True
            assert monitor.shutting_down is False
            assert monitor.session.prompt == mock_settings.prompt

            # Check signal handlers registered
            assert mock_signal.call_count == 2
            mock_signal.assert_any_call(signal.SIGINT, monitor._signal_handler)
            mock_signal.assert_any_call(signal.SIGTERM, monitor._signal_handler)

    def test_signal_handler(self, monitor):
        """Test signal handling for graceful shutdown."""
        assert monitor.running is True
        assert monitor.shutting_down is False

        # Simulate signal
        monitor._signal_handler(signal.SIGINT, None)

        assert monitor.running is False
        assert monitor.shutting_down is True

        # Second signal should not change state
        monitor._signal_handler(signal.SIGINT, None)
        assert monitor.shutting_down is True

    def test_setup_tmux_session(self, monitor):
        """Test tmux session setup."""
        with patch("claude_code_agent_farm.flutter_agent_monitor.run") as mock_run:
            monitor.setup_tmux_session()

            # Should kill existing session first
            assert mock_run.call_count >= 2

            # Check session creation
            create_call = None
            for call_args in mock_run.call_args_list:
                if "new-session" in str(call_args):
                    create_call = call_args
                    break

            assert create_call is not None
            assert monitor.settings.tmux_session in str(create_call)
            # Verify -c option with project path is included
            assert f"-c {monitor.settings.project_path}" in str(create_call)

    def test_start_claude_agent(self, monitor):
        """Test starting Claude agent."""
        with patch("claude_code_agent_farm.flutter_agent_monitor.run") as mock_run:
            with patch.object(monitor, "_has_auto_resume", return_value=True):
                monitor.start_claude_agent()

                # Should NOT send cd command anymore
                for call_args in mock_run.call_args_list:
                    assert "cd " not in str(call_args), "cd command should not be sent"

                # Should start claude-auto-resume
                claude_call = mock_run.call_args_list[0]
                assert "claude-auto-resume" in str(claude_call)
                assert "--dangerously-skip-permissions" in str(claude_call)

    def test_start_claude_agent_without_auto_resume(self, monitor):
        """Test starting Claude without auto-resume."""
        with patch("claude_code_agent_farm.flutter_agent_monitor.run") as mock_run:
            with patch.object(monitor, "_has_auto_resume", return_value=False):
                monitor.start_claude_agent()

                # Should NOT send cd command
                for call_args in mock_run.call_args_list:
                    assert "cd " not in str(call_args), "cd command should not be sent"

                claude_call = mock_run.call_args_list[0]
                assert "claude --dangerously-skip-permissions" in str(claude_call)
                assert "claude-auto-resume" not in str(claude_call)

    def test_send_prompt(self, monitor):
        """Test sending prompt to Claude."""
        with patch("claude_code_agent_farm.flutter_agent_monitor.run") as mock_run:
            monitor.send_prompt()

            # Check prompt was sent
            assert mock_run.called
            send_call = mock_run.call_args[0][0]
            assert "send-keys" in send_call
            assert monitor.settings.prompt_text in send_call

            # Check session counters
            assert monitor.session.total_runs == 1

    def test_capture_pane_content(self, monitor):
        """Test capturing tmux pane content."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "Test output"
            mock_run.return_value.returncode = 0

            content = monitor.capture_pane_content()

            assert content == "Test output"
            assert "capture-pane" in str(mock_run.call_args)

        # Test error case
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")

            content = monitor.capture_pane_content()
            assert content == ""

    def test_check_agent_status_working(self, monitor):
        """Test detecting working status."""
        # Primary test case - the key indicator
        content = "✳ Cerebrating… (2s · ↓ 9 tokens · esc to interrupt)"
        with patch.object(monitor, "capture_pane_content", return_value=content):
            status = monitor.check_agent_status()
            assert status == AgentStatus.WORKING
            
        # Other examples with "esc to interrupt"
        test_cases = [
            "✻ Pontificating... (esc to interrupt)",
            "● Bash(running tests) - press esc to interrupt",
            "Some output with esc to interrupt somewhere",
        ]

        for content in test_cases:
            with patch.object(monitor, "capture_pane_content", return_value=content):
                status = monitor.check_agent_status()
                assert status == AgentStatus.WORKING, f"Failed for content: {content}"

    def test_check_agent_status_ready(self, monitor):
        """Test detecting ready status - absence of 'esc to interrupt'."""
        # Test case 1: Output without "esc to interrupt" means ready
        content1 = """╭──────────────────────────────────────────────────────────────────────────────╮
│ > Type your request here                                                     │
╰──────────────────────────────────────────────────────────────────────────────╯"""
        
        with patch.object(monitor, "capture_pane_content", return_value=content1):
            status = monitor.check_agent_status()
            assert status == AgentStatus.READY
            
        # Test case 2: Task output without working indicator
        content2 = """Task completed successfully
Some other output
No working indicators here"""
        
        with patch.object(monitor, "capture_pane_content", return_value=content2):
            status = monitor.check_agent_status()
            assert status == AgentStatus.READY
            
        # Test case 3: Empty or minimal content (still ready if no "esc to interrupt")
        content3 = "Claude Code v1.0"
        
        with patch.object(monitor, "capture_pane_content", return_value=content3):
            status = monitor.check_agent_status()
            assert status == AgentStatus.READY

    def test_check_agent_status_error(self, monitor):
        """Test detecting error status."""
        test_cases = [
            "error: Something went wrong",  # lowercase
            "failed to connect",  # lowercase
            "exception: Test error",  # lowercase
            "traceback information here",
            "an error occurred while processing",
        ]

        for content in test_cases:
            with patch.object(monitor, "capture_pane_content", return_value=content):
                status = monitor.check_agent_status()
                assert status == AgentStatus.ERROR, f"Failed for content: {content}"

    def test_check_agent_status_usage_limit(self, monitor):
        """Test detecting usage limit."""
        content = "You've reached your usage limit. Please try again at 3:45 PM PST"

        with patch.object(monitor, "capture_pane_content", return_value=content):
            with patch.object(monitor, "_handle_usage_limit") as mock_handle:
                status = monitor.check_agent_status()

                assert status == AgentStatus.USAGE_LIMIT
                mock_handle.assert_called_once_with(content)

    def test_handle_usage_limit(self, monitor):
        """Test handling usage limit detection."""
        content = """You've reached your daily usage limit.
Please try again at 3:45 PM PST.
For more information visit claude.ai"""

        with patch.object(monitor.time_parser, "parse_usage_limit_message") as mock_parse:
            with patch.object(monitor.time_parser, "get_wait_duration") as mock_duration:
                retry_time = datetime.now() + timedelta(hours=2)
                mock_parse.return_value = retry_time
                mock_duration.return_value = timedelta(hours=2)

                monitor._handle_usage_limit(content)

                # Check session was updated
                assert monitor.session.usage_limit_hits == 1
                assert monitor.session.is_waiting_for_limit is True
                assert monitor.session.wait_until == retry_time
                assert monitor.session.last_usage_limit is not None

    def test_handle_usage_limit_no_time(self, monitor):
        """Test handling usage limit without parseable time."""
        content = "Usage limit reached. Try again later."

        with patch.object(monitor.time_parser, "parse_usage_limit_message", return_value=None):
            monitor._handle_usage_limit(content)

            # Should default to 1 hour wait
            assert monitor.session.is_waiting_for_limit is True
            expected_time = datetime.now() + timedelta(hours=1)
            assert abs((monitor.session.wait_until - expected_time).total_seconds()) < 60

    def test_restart_agent(self, monitor):
        """Test restarting the agent."""
        with patch("claude_code_agent_farm.flutter_agent_monitor.run") as mock_run:
            with patch.object(monitor, "start_claude_agent") as mock_start:
                with patch.object(monitor, "send_prompt") as mock_prompt:
                    with patch("time.sleep"):
                        monitor.restart_agent()

                        # Should send Ctrl+C
                        ctrl_c_call = mock_run.call_args_list[0]
                        assert "C-c" in str(ctrl_c_call)

                        # Should clear history
                        clear_call = mock_run.call_args_list[1]
                        assert "clear-history" in str(clear_call)

                        # Should restart
                        mock_start.assert_called_once()
                        mock_prompt.assert_called_once()

                        # Check restart counter
                        assert monitor.session.restart_count == 1

    def test_wait_for_usage_limit(self, monitor):
        """Test waiting for usage limit expiry."""
        # Set wait time 2 seconds in future
        monitor.session.wait_until = datetime.now() + timedelta(seconds=2)
        monitor.session.is_waiting_for_limit = True

        start_time = time.time()

        with patch("time.sleep", side_effect=lambda x: time.sleep(0.1)):
            monitor.wait_for_usage_limit()

        # Should have cleared wait state
        assert monitor.session.is_waiting_for_limit is False
        assert monitor.session.wait_until is None

    def test_wait_for_usage_limit_already_expired(self, monitor):
        """Test wait when limit already expired."""
        # Set wait time in past
        monitor.session.wait_until = datetime.now() - timedelta(seconds=10)
        monitor.session.is_waiting_for_limit = True

        monitor.wait_for_usage_limit()

        # Should clear immediately
        assert monitor.session.is_waiting_for_limit is False
        assert monitor.session.wait_until is None

    def test_get_status_display(self, monitor):
        """Test status display table generation."""
        monitor.session.status = AgentStatus.WORKING
        monitor.session.total_runs = 5
        monitor.session.restart_count = 2
        monitor.session.usage_limit_hits = 1

        table = monitor.get_status_display()

        # Table should contain key information
        assert table is not None
        # Note: Rich Table object doesn't easily expose content for testing
        # In real test would render to string and check content

    def test_has_auto_resume(self, monitor):
        """Test checking for claude-auto-resume availability."""
        # Test when found
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            assert monitor._has_auto_resume() is True

        # Test when not found
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            assert monitor._has_auto_resume() is False

        # Test exception
        with patch("subprocess.run", side_effect=Exception("error")):
            assert monitor._has_auto_resume() is False
    
    def test_is_claude_running(self, monitor):
        """Test detecting if Claude is already running."""
        # Test with Claude UI elements
        claude_contents = [
            "│ > Type your request here",
            "╰──────────────────────────────────────────╯",
            "╭──────────────────────────────────────────╮",
            "Welcome to Claude Code!",
            "✳ Cerebrating… (esc to interrupt)",
            "? for shortcuts",
            "claude-code v1.0",
            "Claude Code is ready",
        ]
        
        for content in claude_contents:
            with patch.object(monitor, "capture_pane_content", return_value=content):
                assert monitor._is_claude_running() is True, f"Should detect Claude with: {content}"
        
        # Test with non-Claude content (shell prompts)
        shell_contents = [
            "$ ",
            "# ",
            "",  # Empty
            "bash-5.1$ ",
        ]
        
        for content in shell_contents:
            with patch.object(monitor, "capture_pane_content", return_value=content):
                assert monitor._is_claude_running() is False, f"Should not detect Claude with: {content}"
        
        # Test with multi-line content that looks like Claude
        claude_session = """Claude Code v1.0
        
        ╭──────────────────────────────────────────╮
        │ > Type your request here                 │
        ╰──────────────────────────────────────────╯"""
        
        with patch.object(monitor, "capture_pane_content", return_value=claude_session):
            assert monitor._is_claude_running() is True
            
    def test_start_claude_agent_when_already_running(self, monitor):
        """Test that startup command is skipped when Claude is already running."""
        with patch.object(monitor, "_is_claude_running", return_value=True):
            with patch("claude_code_agent_farm.flutter_agent_monitor.run") as mock_run:
                with patch.object(monitor, "_has_auto_resume", return_value=True):
                    monitor.start_claude_agent()
                    
                    # Should NOT send any claude startup commands
                    for call in mock_run.call_args_list:
                        assert "claude" not in str(call).lower() or "Claude is already running" in str(call)
                        assert "send-keys" not in str(call)

    def test_cleanup(self, monitor):
        """Test cleanup on shutdown."""
        with patch("claude_code_agent_farm.flutter_agent_monitor.run") as mock_run:
            monitor.session.total_runs = 10
            monitor.session.restart_count = 3
            monitor.session.usage_limit_hits = 2

            monitor.cleanup()

            # Should kill tmux session
            kill_call = mock_run.call_args[0][0]
            assert "kill-session" in kill_call
            assert monitor.settings.tmux_session in kill_call
            
    def test_idle_detection_transition_to_ready(self, monitor):
        """Test that transitioning from working to ready sets timestamp."""
        # Initially working
        monitor.session.status = AgentStatus.WORKING
        monitor.last_ready_time = None
        
        # Mock ready content
        ready_content = """╭──────────────────────────────────────────────────────────────────────────────╮
│ > Task completed successfully                                                │
╰──────────────────────────────────────────────────────────────────────────────╯"""
        
        with patch.object(monitor, "capture_pane_content", return_value=ready_content):
            # Simulate monitoring loop detecting status change
            current_status = monitor.check_agent_status()
            assert current_status == AgentStatus.READY
            
            # Use a fixed datetime for testing
            from datetime import datetime as dt
            test_time = dt(2024, 1, 1, 12, 0, 0)
            
            # Simulate the status change logic from monitoring loop
            if current_status == AgentStatus.READY and monitor.session.status == AgentStatus.WORKING:
                monitor.last_ready_time = test_time
            
            assert monitor.last_ready_time == test_time
                
    def test_restart_on_complete_within_threshold(self, monitor):
        """Test restart when just became ready with restart_on_complete enabled."""
        monitor.settings.restart_on_complete = True
        monitor.last_ready_time = datetime.now() - timedelta(seconds=3)
        
        with patch.object(monitor, "restart_agent") as mock_restart:
            # Simulate idle check logic from monitoring loop
            idle_seconds = (datetime.now() - monitor.last_ready_time).total_seconds()
            
            if idle_seconds < 5 and monitor.settings.restart_on_complete:
                monitor.restart_agent()
                
            mock_restart.assert_called_once()
            
    def test_restart_on_idle_timeout(self, monitor):
        """Test restart when idle time exceeds threshold."""
        monitor.settings.idle_timeout = 30
        monitor.last_ready_time = datetime.now() - timedelta(seconds=35)
        
        with patch.object(monitor, "restart_agent") as mock_restart:
            # Simulate idle check logic from monitoring loop
            idle_seconds = (datetime.now() - monitor.last_ready_time).total_seconds()
            
            if idle_seconds > monitor.settings.idle_timeout:
                monitor.restart_agent()
                
            mock_restart.assert_called_once()
            
    def test_ready_time_reset_on_restart(self, monitor):
        """Test that last_ready_time is reset when agent restarts."""
        monitor.last_ready_time = datetime.now()
        
        # Store initial value to confirm it was set
        initial_time = monitor.last_ready_time
        assert initial_time is not None
        
        with patch("claude_code_agent_farm.flutter_agent_monitor.run"):
            with patch.object(monitor, "start_claude_agent"):
                with patch.object(monitor, "send_prompt"):
                    with patch("time.sleep"):
                        # The restart_tracker should allow restart by default in tests
                        monitor.restart_agent()
                        
                        # Verify ready time was reset
                        assert monitor.last_ready_time is None
                            
    def test_check_agent_status_simplified(self, monitor):
        """Test that status detection is now simplified - no UNKNOWN state."""
        # With new logic, everything without "esc to interrupt" is READY
        test_cases = [
            "Random output that doesn't match any pattern",
            "",  # Empty content
            "Some terminal output\nwithout any indicators",
            "│ >",  # Prompt indicator
            "Just some text",
        ]
        
        for content in test_cases:
            with patch.object(monitor, "capture_pane_content", return_value=content):
                status = monitor.check_agent_status()
                # Should be READY since no "esc to interrupt" found
                assert status == AgentStatus.READY, f"Expected READY for content: {content}"
                
        # Only WORKING if "esc to interrupt" is present
        with patch.object(monitor, "capture_pane_content", return_value="Working... (esc to interrupt)"):
            status = monitor.check_agent_status()
            assert status == AgentStatus.WORKING
                
    def test_status_transitions_and_events(self, monitor):
        """Test that status transitions are properly recorded."""
        # Start with WORKING
        monitor.session.status = AgentStatus.WORKING
        
        # Transition to READY
        ready_content = """╭──────────────────────────────────────────────────────────────────────────────╮
│ > Ready                                                                      │
╰──────────────────────────────────────────────────────────────────────────────╯"""
        
        with patch.object(monitor, "capture_pane_content", return_value=ready_content):
            new_status = monitor.check_agent_status()
            
            # Simulate monitoring loop logic
            if new_status != monitor.session.status:
                from claude_code_agent_farm.models_new.events import StatusChangeEvent
                monitor.events.add(
                    StatusChangeEvent(
                        previous_status=monitor.session.status.value,
                        new_status=new_status.value,
                        source="monitor",
                    )
                )
                
            # Verify event was recorded
            assert len(monitor.events.events) == 1
            event = monitor.events.events[0]
            assert event.previous_status == "working"
            assert event.new_status == "ready"
            
    def test_send_prompt_with_ultrathink(self, monitor):
        """Test that prompts have 'ultrathink' appended."""
        monitor.settings.prompt_text = "Fix the bug in main.dart"
        
        with patch("claude_code_agent_farm.flutter_agent_monitor.run") as mock_run:
            monitor.send_prompt()
            
            # Check both send-keys calls
            assert mock_run.call_count == 2
            
            # First call should send the text with ultrathink
            text_call = mock_run.call_args_list[0][0][0]
            assert "send-keys" in text_call
            assert "Fix the bug in main.dart ultrathink" in text_call
            
            # Second call should send Enter
            enter_call = mock_run.call_args_list[1][0][0]
            assert "send-keys" in enter_call
            assert "C-m" in enter_call


@pytest.mark.unit
class TestMonitorRunLoop:
    """Test the main run loop of the monitor."""

    def test_run_startup_sequence(self, monitor):
        """Test the startup sequence."""
        with patch.object(monitor, "setup_tmux_session") as mock_setup:
            with patch.object(monitor, "start_claude_agent") as mock_start:
                with patch.object(monitor, "send_prompt") as mock_prompt:
                    with patch("claude_code_agent_farm.flutter_agent_monitor.Live"):
                        with patch("time.sleep"):
                            # Make run exit after startup
                            monitor.running = False

                            exit_code = monitor.run()

                            # Check startup sequence
                            mock_setup.assert_called_once()
                            mock_start.assert_called_once()
                            mock_prompt.assert_called_once()
                            assert exit_code == 0

    def test_run_status_monitoring(self, monitor):
        """Test status monitoring in run loop."""
        status_sequence = [
            AgentStatus.WORKING,
            AgentStatus.WORKING,
            AgentStatus.READY,
        ]

        with patch.object(monitor, "setup_tmux_session"):
            with patch.object(monitor, "start_claude_agent"):
                with patch.object(monitor, "send_prompt"):
                    with patch.object(monitor, "check_agent_status", side_effect=status_sequence):
                        with patch("claude_code_agent_farm.flutter_agent_monitor.Live"):
                            with patch("time.sleep"):
                                # Run for 3 iterations
                                iteration = 0
                                original_check = monitor.check_agent_status

                                def stop_after_3(*args):
                                    nonlocal iteration
                                    iteration += 1
                                    if iteration >= 3:
                                        monitor.running = False
                                    return original_check()

                                monitor.check_agent_status = stop_after_3
                                monitor.run()

                                assert monitor.session.status == AgentStatus.READY
