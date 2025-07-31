"""Integration tests for agent lifecycle management."""

import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

import pytest

from claude_code_agent_farm.flutter_agent_monitor import FlutterAgentMonitor
from claude_code_agent_farm.flutter_agent_settings import FlutterAgentSettings
from claude_code_agent_farm.models import AgentStatus, AgentEvent
from claude_code_agent_farm.utils import run


@pytest.mark.integration
class TestAgentLifecycle:
    """Test complete agent lifecycle including startup, monitoring, and restart."""
    
    @pytest.fixture
    def mock_settings(self, mock_carenji_project):
        """Create test settings."""
        return FlutterAgentSettings(
            claude_project_path=mock_carenji_project,
            tmux_session_name="test-lifecycle",
            prompt_text="Implement unit tests for the Carenji medication module",
            monitor_interval=1.0,  # Fast for testing
            max_runs=3,
            error_restart_delay=2
        )
    
    @pytest.fixture
    def monitor(self, mock_settings):
        """Create monitor instance with mocked subprocess."""
        with patch("subprocess.run") as mock_run:
            with patch("claude_code_agent_farm.flutter_agent_monitor.signal.signal"):
                mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
                monitor = FlutterAgentMonitor(mock_settings)
                
                # Mock tmux operations
                monitor._create_tmux_session = Mock()
                monitor._kill_tmux_session = Mock()
                
                yield monitor
    
    def test_agent_startup(self, monitor):
        """Test agent startup process."""
        # Mock Claude start response
        with patch.object(monitor, "capture_pane_content", return_value="Claude ready\n>> "):
            monitor.start_agent()
            
            # Verify tmux session created
            monitor._create_tmux_session.assert_called_once()
            
            # Verify session initialized
            assert monitor.session is not None
            assert monitor.session.status == AgentStatus.READY
            assert monitor.session.start_time is not None
            assert monitor.session.prompt == monitor.settings.prompt_text
    
    def test_send_prompt_to_agent(self, monitor):
        """Test sending prompt to agent."""
        monitor.start_agent()
        
        # Mock sending keys
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            
            monitor.send_prompt()
            
            # Verify tmux send-keys called with prompt
            expected_call = [
                "tmux", "send-keys", "-t", f"{monitor.settings.tmux_session_name}:0",
                monitor.settings.prompt_text, "Enter"
            ]
            mock_run.assert_called_with(expected_call, check=True)
    
    def test_monitor_agent_working(self, monitor):
        """Test monitoring agent while working."""
        monitor.start_agent()
        
        # Simulate agent working
        responses = [
            "Working on implementing tests...",
            "Creating test files...",
            "Running flutter test...",
            "✓ All tests passed!\n>> "
        ]
        
        with patch.object(monitor, "capture_pane_content", side_effect=responses):
            # First checks should show working
            assert monitor.check_agent_status() == AgentStatus.WORKING
            assert monitor.check_agent_status() == AgentStatus.WORKING
            assert monitor.check_agent_status() == AgentStatus.WORKING
            
            # Final check shows ready
            assert monitor.check_agent_status() == AgentStatus.READY
    
    def test_task_completion_detection(self, monitor):
        """Test detecting when task is complete."""
        monitor.start_agent()
        monitor.session.status = AgentStatus.WORKING
        
        # Mock task completion
        with patch.object(monitor, "capture_pane_content", return_value="Task completed successfully!\n>> "):
            status = monitor.check_agent_status()
            
            assert status == AgentStatus.READY
            # Should record completion event
            assert any(event.event_type == "task_completed" for event in monitor.session.events)
    
    def test_automatic_restart_on_completion(self, monitor):
        """Test agent restarts automatically when task completes."""
        monitor.start_agent()
        monitor.send_prompt()
        
        # Simulate task completion
        monitor.session.status = AgentStatus.READY
        monitor.session.runs = 1
        
        with patch.object(monitor, "restart_agent") as mock_restart:
            with patch.object(monitor, "check_agent_status", return_value=AgentStatus.READY):
                # Run one monitoring cycle
                monitor.running = True
                monitor._monitor_once()
                
                # Should restart if not at max runs
                if monitor.session.runs < monitor.settings.max_runs:
                    mock_restart.assert_called_once()
    
    def test_error_detection_and_recovery(self, monitor):
        """Test detecting errors and recovering."""
        monitor.start_agent()
        monitor.session.status = AgentStatus.WORKING
        
        # Simulate error
        error_output = "Error: Failed to connect to Firebase emulator"
        with patch.object(monitor, "capture_pane_content", return_value=error_output):
            status = monitor.check_agent_status()
            
            assert status == AgentStatus.ERROR
            assert any(event.event_type == "error" for event in monitor.session.events)
    
    def test_restart_after_error(self, monitor):
        """Test agent restarts after error with delay."""
        monitor.start_agent()
        monitor.session.status = AgentStatus.ERROR
        monitor.session.last_error_time = datetime.now()
        
        with patch.object(monitor, "restart_agent") as mock_restart:
            with patch("time.sleep") as mock_sleep:
                # Trigger restart logic
                monitor._handle_error_state()
                
                # Should wait before restart
                mock_sleep.assert_called_with(monitor.settings.error_restart_delay)
                mock_restart.assert_called_once()
    
    def test_usage_limit_handling(self, monitor):
        """Test handling usage limit."""
        monitor.start_agent()
        
        # Simulate usage limit
        usage_output = "You've reached your usage limit. Please try again at 3:45 PM PST"
        with patch.object(monitor, "capture_pane_content", return_value=usage_output):
            status = monitor.check_agent_status()
            
            assert status == AgentStatus.USAGE_LIMIT
            assert monitor.session.usage_limit_info is not None
            assert monitor.session.usage_limit_info.retry_time.hour == 15
            assert monitor.session.usage_limit_info.retry_time.minute == 45
    
    def test_scheduled_restart_after_usage_limit(self, monitor):
        """Test scheduled restart after usage limit."""
        monitor.start_agent()
        
        # Set usage limit with retry time in the past
        monitor.session.status = AgentStatus.USAGE_LIMIT
        retry_time = datetime.now() - timedelta(minutes=1)
        monitor.session.usage_limit_info = Mock(retry_time=retry_time)
        
        with patch.object(monitor, "restart_agent") as mock_restart:
            # Check if should restart
            monitor._handle_usage_limit()
            
            # Should restart since retry time passed
            mock_restart.assert_called_once()
    
    def test_max_runs_limit(self, monitor):
        """Test stopping after max runs reached."""
        monitor.settings.max_runs = 2
        monitor.start_agent()
        monitor.session.runs = 2
        
        with patch.object(monitor, "restart_agent") as mock_restart:
            # Try to trigger restart
            monitor._check_restart_conditions()
            
            # Should not restart
            mock_restart.assert_not_called()
            
            # Should record max runs event
            assert any(event.event_type == "max_runs_reached" for event in monitor.session.events)
    
    def test_graceful_shutdown(self, monitor):
        """Test graceful shutdown on signal."""
        monitor.start_agent()
        monitor.running = True
        
        # Simulate shutdown
        monitor.shutdown()
        
        assert not monitor.running
        monitor._kill_tmux_session.assert_called_once()
        
        # Should save session
        assert any(event.event_type == "shutdown" for event in monitor.session.events)
    
    def test_session_persistence(self, monitor, temp_dir):
        """Test session state is saved and can be restored."""
        monitor.start_agent()
        monitor.session.runs = 2
        monitor.session.events.append(
            AgentEvent(event_type="test", message="Test event")
        )
        
        # Save session
        session_file = temp_dir / "test_session.json"
        monitor.save_session(str(session_file))
        
        # Verify file created
        assert session_file.exists()
        
        # Load and verify
        data = json.loads(session_file.read_text())
        assert data["prompt"] == monitor.settings.prompt_text
        assert data["runs"] == 2
        assert len(data["events"]) > 0
    
    def test_restart_maintains_context(self, monitor):
        """Test that restart maintains the same prompt and context."""
        original_prompt = "Test the Carenji medication ordering feature"
        monitor.settings.prompt_text = original_prompt
        monitor.start_agent()
        
        # Record some context
        monitor.session.runs = 1
        monitor.session.events.append(
            AgentEvent(event_type="custom", message="Previous run completed task X")
        )
        
        with patch.object(monitor, "_create_tmux_session"):
            with patch.object(monitor, "send_prompt") as mock_send:
                monitor.restart_agent()
                
                # Verify same prompt used
                assert monitor.settings.prompt_text == original_prompt
                mock_send.assert_called_once()
                
                # Verify run count incremented
                assert monitor.session.runs == 2
                
                # Verify events preserved
                assert any(event.message == "Previous run completed task X" 
                          for event in monitor.session.events)
    
    def test_monitor_loop(self, monitor):
        """Test the main monitoring loop."""
        monitor.settings.monitor_interval = 0.1  # Fast for testing
        monitor.start_agent()
        
        check_count = 0
        def mock_check():
            nonlocal check_count
            check_count += 1
            if check_count >= 3:
                monitor.running = False
            return AgentStatus.WORKING
        
        with patch.object(monitor, "check_agent_status", side_effect=mock_check):
            with patch("time.sleep"):
                monitor.monitor()
                
                # Should have checked multiple times
                assert check_count >= 3
    
    @pytest.mark.docker
    def test_real_tmux_lifecycle(self, docker_container):
        """Test real tmux lifecycle in Docker."""
        session_name = "test-real-lifecycle"
        
        # Start tmux session
        result = subprocess.run(
            ["docker", "exec", docker_container,
             "tmux", "new-session", "-d", "-s", session_name],
            capture_output=True
        )
        assert result.returncode == 0
        
        # Send claude command
        result = subprocess.run(
            ["docker", "exec", docker_container,
             "tmux", "send-keys", "-t", session_name,
             "echo 'Mock Claude response'", "Enter"],
            capture_output=True
        )
        assert result.returncode == 0
        
        time.sleep(0.5)
        
        # Capture output
        result = subprocess.run(
            ["docker", "exec", docker_container,
             "tmux", "capture-pane", "-t", session_name, "-p"],
            capture_output=True,
            text=True
        )
        
        assert "Mock Claude response" in result.stdout
        
        # Kill session
        subprocess.run(
            ["docker", "exec", docker_container,
             "tmux", "kill-session", "-t", session_name],
            capture_output=True
        )
    
    def test_concurrent_monitoring(self, monitor):
        """Test handling concurrent status checks."""
        monitor.start_agent()
        
        # Simulate rapid status changes
        statuses = [AgentStatus.WORKING, AgentStatus.READY, AgentStatus.WORKING]
        
        with patch.object(monitor, "capture_pane_content", side_effect=[
            "Working...", ">> ", "Starting new task..."
        ]):
            results = []
            for _ in range(3):
                results.append(monitor.check_agent_status())
            
            assert results == statuses
    
    def test_error_patterns(self, monitor):
        """Test detecting various error patterns."""
        monitor.start_agent()
        
        error_patterns = [
            ("Error: Command not found", AgentStatus.ERROR),
            ("fatal: repository not found", AgentStatus.ERROR),
            ("Exception in thread", AgentStatus.ERROR),
            ("ECONNREFUSED", AgentStatus.ERROR),
            ("✗ Tests failed", AgentStatus.ERROR),
            ("ModuleNotFoundError", AgentStatus.ERROR)
        ]
        
        for error_text, expected_status in error_patterns:
            with patch.object(monitor, "capture_pane_content", return_value=error_text):
                status = monitor.check_agent_status()
                assert status == expected_status, f"Failed to detect error in: {error_text}"