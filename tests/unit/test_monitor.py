"""Unit tests for Flutter Agent Monitor."""

import signal
import subprocess
import time
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, MagicMock, call

import pytest

from claude_code_agent_farm.flutter_agent_monitor import FlutterAgentMonitor
from claude_code_agent_farm.models import AgentStatus, UsageLimitInfo
from claude_code_agent_farm import constants


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
    
    def test_start_claude_agent(self, monitor):
        """Test starting Claude agent."""
        with patch("claude_code_agent_farm.flutter_agent_monitor.run") as mock_run:
            with patch.object(monitor, "_has_auto_resume", return_value=True):
                monitor.start_claude_agent()
                
                # Should send cd command
                cd_call = mock_run.call_args_list[0]
                assert f"cd {monitor.settings.project_path}" in str(cd_call)
                
                # Should start claude-auto-resume
                claude_call = mock_run.call_args_list[1]
                assert "claude-auto-resume" in str(claude_call)
                assert "--dangerously-skip-permissions" in str(claude_call)
    
    def test_start_claude_agent_without_auto_resume(self, monitor):
        """Test starting Claude without auto-resume."""
        with patch("claude_code_agent_farm.flutter_agent_monitor.run") as mock_run:
            with patch.object(monitor, "_has_auto_resume", return_value=False):
                monitor.start_claude_agent()
                
                claude_call = mock_run.call_args_list[1]
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
        test_cases = [
            "Thinking about the problem...",
            "Analyzing the codebase...",
            "Reading file: main.dart",
            "Running command: flutter test",
        ]
        
        for content in test_cases:
            with patch.object(monitor, "capture_pane_content", return_value=content):
                status = monitor.check_agent_status()
                assert status == AgentStatus.WORKING
    
    def test_check_agent_status_ready(self, monitor):
        """Test detecting ready status."""
        with patch.object(monitor, "capture_pane_content", return_value=">> "):
            status = monitor.check_agent_status()
            assert status == AgentStatus.READY
    
    def test_check_agent_status_error(self, monitor):
        """Test detecting error status."""
        test_cases = [
            "Error: Something went wrong",
            "Failed to connect",
            "Exception: Test error",
        ]
        
        for content in test_cases:
            with patch.object(monitor, "capture_pane_content", return_value=content):
                status = monitor.check_agent_status()
                assert status == AgentStatus.ERROR
    
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