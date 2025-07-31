"""End-to-end tests for usage limit handling."""

import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

import pytest
import pytz

from claude_code_agent_farm.flutter_agent_monitor import FlutterAgentMonitor
from claude_code_agent_farm.flutter_agent_settings import FlutterAgentSettings
from claude_code_agent_farm.models import AgentStatus, UsageLimitInfo
from claude_code_agent_farm.utils import UsageLimitTimeParser


@pytest.mark.e2e
class TestUsageLimits:
    """Test end-to-end usage limit scenarios."""
    
    @pytest.fixture
    def usage_limit_messages(self):
        """Collection of real usage limit messages."""
        return [
            "You've reached your usage limit. Please try again at 3:45 PM PST",
            "Usage limit reached. Please try again at 11:30 PM EST",
            "You've hit the rate limit. Try again at 9:00 AM UTC",
            "Daily limit exceeded. Please try again at 12:00 AM PST tomorrow",
            "Sorry, you've reached your limit. Come back at 6:15 PM CST",
            "Rate limit hit. Please wait until 2:30 AM GMT",
            "You've reached your usage limit. Please try again at 10:45 AM PDT",
            "Limit reached for today. Try again at 00:00 UTC",
        ]
    
    @pytest.fixture
    def monitor_with_usage_limit(self, mock_settings):
        """Create monitor that will hit usage limit."""
        with patch("subprocess.run") as mock_run:
            with patch("claude_code_agent_farm.flutter_agent_monitor.signal.signal"):
                monitor = FlutterAgentMonitor(mock_settings)
                
                # Mock tmux operations
                mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
                monitor._create_tmux_session = Mock()
                monitor._kill_tmux_session = Mock()
                
                yield monitor
    
    def test_usage_limit_detection(self, monitor_with_usage_limit, usage_limit_messages):
        """Test detecting various usage limit messages."""
        monitor = monitor_with_usage_limit
        monitor.start_agent()
        
        for message in usage_limit_messages:
            with patch.object(monitor, "capture_pane_content", return_value=message):
                status = monitor.check_agent_status()
                
                assert status == AgentStatus.USAGE_LIMIT
                assert monitor.session.usage_limit_info is not None
                assert monitor.session.usage_limit_info.retry_time is not None
    
    def test_retry_time_parsing(self, monitor_with_usage_limit):
        """Test parsing retry times from usage limit messages."""
        monitor = monitor_with_usage_limit
        monitor.start_agent()
        
        test_cases = [
            ("Please try again at 3:45 PM PST", 15, 45, "PST"),
            ("Try again at 11:30 PM EST", 23, 30, "EST"),
            ("Come back at 9:00 AM UTC", 9, 0, "UTC"),
            ("Try again at 12:00 AM PST tomorrow", 0, 0, "PST"),
        ]
        
        for message, expected_hour, expected_minute, tz in test_cases:
            full_message = f"You've reached your usage limit. {message}"
            
            with patch.object(monitor, "capture_pane_content", return_value=full_message):
                monitor.check_agent_status()
                
                limit_info = monitor.session.usage_limit_info
                assert limit_info is not None
                
                # Convert to the message's timezone for comparison
                retry_local = limit_info.retry_time
                assert retry_local.hour == expected_hour
                assert retry_local.minute == expected_minute
    
    def test_scheduled_restart_timing(self, monitor_with_usage_limit):
        """Test that restart is scheduled for the correct time."""
        monitor = monitor_with_usage_limit
        monitor.start_agent()
        
        # Hit usage limit with retry in 5 minutes
        future_time = datetime.now() + timedelta(minutes=5)
        message = f"You've reached your usage limit. Please try again at {future_time.strftime('%-I:%M %p')} PST"
        
        with patch.object(monitor, "capture_pane_content", return_value=message):
            monitor.check_agent_status()
        
        # Should not restart immediately
        with patch.object(monitor, "restart_agent") as mock_restart:
            monitor._handle_usage_limit()
            mock_restart.assert_not_called()
        
        # Mock time to after retry time
        with patch("claude_code_agent_farm.flutter_agent_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = future_time + timedelta(minutes=1)
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            
            with patch.object(monitor, "restart_agent") as mock_restart:
                monitor._handle_usage_limit()
                mock_restart.assert_called_once()
    
    def test_usage_limit_wait_behavior(self, monitor_with_usage_limit):
        """Test waiting behavior when usage limit is hit."""
        monitor = monitor_with_usage_limit
        monitor.start_agent()
        
        # Set usage limit with retry in 10 minutes
        future_time = datetime.now() + timedelta(minutes=10)
        message = f"Usage limit reached. Try again at {future_time.strftime('%-I:%M %p')} PST"
        
        with patch.object(monitor, "capture_pane_content", return_value=message):
            monitor.check_agent_status()
        
        # Monitor should wait and check periodically
        wait_times = []
        
        def mock_sleep(seconds):
            wait_times.append(seconds)
        
        with patch("time.sleep", side_effect=mock_sleep):
            with patch.object(monitor, "running", side_effect=[True, True, False]):
                with patch("claude_code_agent_farm.flutter_agent_monitor.datetime") as mock_dt:
                    mock_dt.now.return_value = datetime.now()
                    monitor._wait_for_usage_limit_reset()
        
        # Should have waited multiple times
        assert len(wait_times) > 0
        # Wait times should be reasonable (not too short, not too long)
        assert all(30 <= t <= 300 for t in wait_times)
    
    def test_multiple_usage_limits_in_session(self, monitor_with_usage_limit):
        """Test handling multiple usage limits in one session."""
        monitor = monitor_with_usage_limit
        monitor.start_agent()
        
        # First usage limit
        with patch.object(monitor, "capture_pane_content", 
                         return_value="Usage limit. Try again at 3:00 PM PST"):
            monitor.check_agent_status()
        
        assert monitor.session.usage_limit_hits == 1
        
        # Simulate restart after limit resets
        monitor.session.status = AgentStatus.WORKING
        
        # Second usage limit
        with patch.object(monitor, "capture_pane_content",
                         return_value="Usage limit. Try again at 5:00 PM PST"):
            monitor.check_agent_status()
        
        assert monitor.session.usage_limit_hits == 2
        
        # Check events recorded
        usage_events = [e for e in monitor.session.events 
                       if e.event_type == "usage_limit"]
        assert len(usage_events) == 2
    
    def test_timezone_handling(self, monitor_with_usage_limit):
        """Test proper timezone handling for different regions."""
        monitor = monitor_with_usage_limit
        monitor.start_agent()
        
        # Test various timezone formats
        tz_tests = [
            ("3:45 PM PST", "US/Pacific"),
            ("3:45 PM PDT", "US/Pacific"),
            ("3:45 PM EST", "US/Eastern"),
            ("3:45 PM CST", "US/Central"),
            ("3:45 PM GMT", "GMT"),
            ("3:45 PM UTC", "UTC"),
        ]
        
        for time_str, expected_tz in tz_tests:
            message = f"Usage limit. Try again at {time_str}"
            
            with patch.object(monitor, "capture_pane_content", return_value=message):
                monitor.check_agent_status()
                
                limit_info = monitor.session.usage_limit_info
                assert limit_info is not None
                # Should have parsed the time
                assert limit_info.retry_time is not None
    
    def test_tomorrow_handling(self, monitor_with_usage_limit):
        """Test handling 'tomorrow' in retry times."""
        monitor = monitor_with_usage_limit
        monitor.start_agent()
        
        # Message with "tomorrow"
        message = "Usage limit reached. Try again at 12:00 AM PST tomorrow"
        
        with patch.object(monitor, "capture_pane_content", return_value=message):
            monitor.check_agent_status()
        
        limit_info = monitor.session.usage_limit_info
        assert limit_info is not None
        
        # Should be midnight tomorrow
        assert limit_info.retry_time.hour == 0
        assert limit_info.retry_time.minute == 0
        
        # Should be in the future
        assert limit_info.retry_time > datetime.now()
    
    def test_usage_limit_recovery(self, monitor_with_usage_limit):
        """Test recovery after usage limit resets."""
        monitor = monitor_with_usage_limit
        monitor.start_agent()
        
        # Hit usage limit
        past_time = datetime.now() - timedelta(minutes=5)
        message = f"Usage limit. Try again at {past_time.strftime('%-I:%M %p')} PST"
        
        with patch.object(monitor, "capture_pane_content", return_value=message):
            monitor.check_agent_status()
        
        # Should restart immediately since retry time is in the past
        with patch.object(monitor, "restart_agent") as mock_restart:
            with patch.object(monitor, "send_prompt") as mock_send:
                monitor._handle_usage_limit()
                
                mock_restart.assert_called_once()
        
        # After restart, should continue working
        monitor.session.status = AgentStatus.WORKING
        
        with patch.object(monitor, "capture_pane_content", 
                         return_value="Working on the Carenji tests..."):
            status = monitor.check_agent_status()
            assert status == AgentStatus.WORKING
    
    def test_usage_limit_with_long_wait(self, monitor_with_usage_limit):
        """Test handling usage limits with long wait times."""
        monitor = monitor_with_usage_limit
        monitor.start_agent()
        
        # Usage limit with retry tomorrow
        tomorrow = datetime.now() + timedelta(days=1)
        message = f"Daily limit exceeded. Try again at {tomorrow.strftime('%-I:%M %p')} PST tomorrow"
        
        with patch.object(monitor, "capture_pane_content", return_value=message):
            monitor.check_agent_status()
        
        limit_info = monitor.session.usage_limit_info
        assert limit_info is not None
        
        # Should recognize it's a long wait
        wait_hours = (limit_info.retry_time - datetime.now()).total_seconds() / 3600
        assert wait_hours > 12  # More than 12 hours
    
    def test_usage_limit_monitoring_loop(self, monitor_with_usage_limit):
        """Test the monitoring loop during usage limit wait."""
        monitor = monitor_with_usage_limit
        monitor.start_agent()
        monitor.settings.usage_limit_check_interval = 1  # Fast for testing
        
        # Hit usage limit with retry in 2 seconds (for quick test)
        future_time = datetime.now() + timedelta(seconds=2)
        message = f"Usage limit. Try again at {future_time.strftime('%-I:%M:%S %p')} PST"
        
        with patch.object(monitor, "capture_pane_content", return_value=message):
            monitor.check_agent_status()
        
        restart_called = False
        
        def mock_restart():
            nonlocal restart_called
            restart_called = True
            monitor.running = False  # Stop the loop
        
        with patch.object(monitor, "restart_agent", side_effect=mock_restart):
            with patch("time.sleep", lambda x: time.sleep(0.1)):  # Speed up
                # Start monitoring
                monitor.running = True
                start_time = time.time()
                monitor._monitor_once()
                
                # Wait a bit for the retry time to pass
                while time.time() - start_time < 3 and not restart_called:
                    time.sleep(0.1)
                    if datetime.now() >= future_time:
                        monitor._handle_usage_limit()
                
                # Should have restarted after retry time
                assert restart_called
    
    @pytest.mark.docker
    def test_real_usage_limit_scenario(self, docker_container):
        """Test usage limit handling in real Docker environment."""
        session_name = "test-usage-limit"
        
        # Create tmux session
        subprocess.run(
            ["docker", "exec", docker_container,
             "tmux", "new-session", "-d", "-s", session_name],
            check=True
        )
        
        try:
            # Simulate usage limit message
            usage_msg = "You've reached your usage limit. Please try again at 5:00 PM PST"
            
            subprocess.run(
                ["docker", "exec", docker_container,
                 "tmux", "send-keys", "-t", session_name,
                 f"echo \"{usage_msg}\"", "Enter"],
                check=True
            )
            
            time.sleep(0.5)
            
            # Capture and verify
            result = subprocess.run(
                ["docker", "exec", docker_container,
                 "tmux", "capture-pane", "-t", session_name, "-p"],
                capture_output=True,
                text=True
            )
            
            assert "usage limit" in result.stdout
            assert "5:00 PM PST" in result.stdout
            
        finally:
            # Cleanup
            subprocess.run(
                ["docker", "exec", docker_container,
                 "tmux", "kill-session", "-t", session_name],
                capture_output=True
            )
    
    def test_usage_limit_edge_cases(self, monitor_with_usage_limit):
        """Test edge cases in usage limit handling."""
        monitor = monitor_with_usage_limit
        monitor.start_agent()
        
        # Test malformed message
        with patch.object(monitor, "capture_pane_content", 
                         return_value="Usage limit reached but no time given"):
            status = monitor.check_agent_status()
            # Should still detect as usage limit even without time
            assert status == AgentStatus.USAGE_LIMIT
        
        # Test with only time, no timezone
        with patch.object(monitor, "capture_pane_content",
                         return_value="Usage limit. Try again at 3:45 PM"):
            monitor.check_agent_status()
            # Should use default timezone
            assert monitor.session.usage_limit_info is not None
        
        # Test midnight edge case
        with patch.object(monitor, "capture_pane_content",
                         return_value="Usage limit. Try again at 12:00 AM PST"):
            monitor.check_agent_status()
            limit_info = monitor.session.usage_limit_info
            assert limit_info.retry_time.hour == 0
            assert limit_info.retry_time.minute == 0