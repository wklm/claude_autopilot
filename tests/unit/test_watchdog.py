"""Unit tests for watchdog timer functionality."""

from datetime import datetime, timedelta
from unittest.mock import patch

from claude_code_agent_farm.models_new.watchdog import (
    ActivityType,
    HungAgentDetector,
    WatchdogState,
    WatchdogTimer,
)


class TestWatchdogTimer:
    """Test watchdog timer functionality."""

    def test_initial_state(self):
        """Test initial watchdog state."""
        watchdog = WatchdogTimer()

        assert watchdog.state == WatchdogState.STOPPED
        assert watchdog.last_activity is None
        assert watchdog.timer_started is None

    def test_start_watchdog(self):
        """Test starting the watchdog."""
        watchdog = WatchdogTimer()

        with patch("claude_code_agent_farm.model_types.watchdog.datetime") as mock_dt:
            now = datetime(2024, 1, 1, 12, 0, 0)
            mock_dt.now.return_value = now

            watchdog.start()

            assert watchdog.state == WatchdogState.ACTIVE
            assert watchdog.timer_started == now
            assert watchdog.last_activity == now

    def test_stop_watchdog(self):
        """Test stopping the watchdog."""
        watchdog = WatchdogTimer()
        watchdog.start()

        watchdog.stop()

        assert watchdog.state == WatchdogState.STOPPED
        assert watchdog.timer_started is None

    def test_pause_resume(self):
        """Test pausing and resuming."""
        watchdog = WatchdogTimer()
        watchdog.start()

        # Pause
        watchdog.pause()
        assert watchdog.state == WatchdogState.PAUSED

        # Resume
        watchdog.resume()
        assert watchdog.state == WatchdogState.ACTIVE
        assert watchdog.last_activity is not None  # Reset on resume

    def test_feed_updates_activity(self):
        """Test feeding the watchdog."""
        watchdog = WatchdogTimer()
        watchdog.start()

        initial_activity = watchdog.last_activity

        # Feed the watchdog
        watchdog.feed(ActivityType.PANE_OUTPUT, "Output changed", content_length=100)

        assert watchdog.last_activity > initial_activity
        assert len(watchdog.activity_log) == 1

        # Check activity log
        activity = watchdog.activity_log[0]
        assert activity.activity_type == ActivityType.PANE_OUTPUT
        assert activity.description == "Output changed"
        assert activity.details["content_length"] == 100

    def test_feed_ignored_when_inactive(self):
        """Test feed is ignored when not active."""
        watchdog = WatchdogTimer()

        # Feed without starting
        watchdog.feed(ActivityType.HEARTBEAT, "Test")

        assert len(watchdog.activity_log) == 0
        assert watchdog.last_activity is None

    def test_timeout_detection(self):
        """Test timeout detection."""
        watchdog = WatchdogTimer(timeout_seconds=60)
        watchdog.start()

        # No timeout initially
        timed_out, reason = watchdog.check_timeout()
        assert not timed_out

        # Simulate timeout
        watchdog.last_activity = datetime.now() - timedelta(seconds=61)

        timed_out, reason = watchdog.check_timeout()
        assert timed_out
        assert "No activity for 61 seconds" in reason
        assert watchdog.state == WatchdogState.TRIGGERED

    def test_grace_period(self):
        """Test grace period after start."""
        watchdog = WatchdogTimer(timeout_seconds=60, grace_period_seconds=30)

        with patch("claude_code_agent_farm.model_types.watchdog.datetime") as mock_dt:
            start_time = datetime(2024, 1, 1, 12, 0, 0)
            mock_dt.now.return_value = start_time

            watchdog.start()

            # During grace period
            mock_dt.now.return_value = start_time + timedelta(seconds=20)
            watchdog.last_activity = start_time  # No activity

            timed_out, reason = watchdog.check_timeout()
            assert not timed_out  # Grace period protects

            # After grace period
            mock_dt.now.return_value = start_time + timedelta(seconds=31)

            timed_out, reason = watchdog.check_timeout()
            assert not timed_out  # Still within timeout

            # After grace period and timeout
            mock_dt.now.return_value = start_time + timedelta(seconds=91)

            timed_out, reason = watchdog.check_timeout()
            assert timed_out

    def test_content_change_detection(self):
        """Test content change detection."""
        watchdog = WatchdogTimer()

        # First content
        assert watchdog.is_content_changed("Hello world")
        assert watchdog.last_pane_content_hash is not None

        # Same content
        assert not watchdog.is_content_changed("Hello world")

        # Different content
        assert watchdog.is_content_changed("Hello world!")
        assert watchdog.last_meaningful_change is not None

    def test_activity_log_limit(self):
        """Test activity log size limit."""
        watchdog = WatchdogTimer()
        watchdog.start()

        # Add many activities
        for i in range(1500):
            watchdog.feed(ActivityType.HEARTBEAT, f"Activity {i}")

        # Should maintain limit
        assert len(watchdog.activity_log) == 500

        # Should keep most recent
        assert watchdog.activity_log[-1].description == "Activity 1499"

    def test_time_since_activity(self):
        """Test calculating time since activity."""
        watchdog = WatchdogTimer()

        # No activity yet
        assert watchdog.get_time_since_activity() is None

        # With activity
        watchdog.last_activity = datetime.now() - timedelta(seconds=30)
        time_delta = watchdog.get_time_since_activity()

        assert time_delta is not None
        assert 29 <= time_delta.total_seconds() <= 31

    def test_activity_summary(self):
        """Test getting activity summary."""
        watchdog = WatchdogTimer()
        watchdog.start()

        # Add various activities
        watchdog.feed(ActivityType.PANE_OUTPUT, "Output 1")
        watchdog.feed(ActivityType.PANE_OUTPUT, "Output 2")
        watchdog.feed(ActivityType.STATUS_CHANGE, "Status changed")
        watchdog.feed(ActivityType.HEARTBEAT, "Heartbeat")

        summary = watchdog.get_activity_summary(last_n_minutes=5)

        assert summary[ActivityType.PANE_OUTPUT] == 2
        assert summary[ActivityType.STATUS_CHANGE] == 1
        assert summary[ActivityType.HEARTBEAT] == 1

    def test_trigger_count(self):
        """Test trigger counting."""
        watchdog = WatchdogTimer(timeout_seconds=60)
        watchdog.start()

        # Trigger multiple times
        for i in range(3):
            watchdog.last_activity = datetime.now() - timedelta(seconds=61)
            watchdog.check_timeout()
            watchdog.state = WatchdogState.ACTIVE  # Reset for next trigger

        assert watchdog.trigger_count == 3
        assert watchdog.last_trigger is not None


class TestHungAgentDetector:
    """Test hung agent detection."""

    def test_stuck_pattern_detection(self):
        """Test detecting stuck patterns."""
        detector = HungAgentDetector(stuck_pattern_threshold=3)

        # First occurrence
        is_stuck, reason = detector.check_stuck_patterns("Thinking...")
        assert not is_stuck

        # Second occurrence
        is_stuck, reason = detector.check_stuck_patterns("Thinking...")
        assert not is_stuck

        # Third occurrence - stuck
        is_stuck, reason = detector.check_stuck_patterns("Thinking...")
        assert is_stuck
        assert "Stuck on 'Thinking...'" in reason
        assert "repeated 3 times" in reason

    def test_pattern_reset(self):
        """Test pattern count resets when not present."""
        detector = HungAgentDetector()

        # Build up count
        detector.check_stuck_patterns("Processing...")
        detector.check_stuck_patterns("Processing...")

        assert detector.repeated_patterns["Processing..."] == 2

        # Different content resets count
        detector.check_stuck_patterns("Done processing")

        assert "Processing..." not in detector.repeated_patterns

    def test_multiple_patterns(self):
        """Test tracking multiple patterns."""
        detector = HungAgentDetector(stuck_pattern_threshold=2)

        # Different patterns
        detector.check_stuck_patterns("Waiting for response")
        detector.check_stuck_patterns("Retrying connection")
        detector.check_stuck_patterns("Waiting for response")

        # First pattern stuck
        is_stuck, reason = detector.check_stuck_patterns("Waiting for response")
        assert is_stuck

        # Second pattern not stuck yet
        is_stuck, reason = detector.check_stuck_patterns("Retrying connection")
        assert not is_stuck

    def test_common_stuck_indicators(self):
        """Test common stuck indicators are detected."""
        detector = HungAgentDetector(stuck_pattern_threshold=1)

        stuck_messages = ["Thinking...", "Processing...", "Loading...", "Waiting for input", "Retrying operation"]

        for msg in stuck_messages:
            detector.repeated_patterns.clear()  # Reset
            is_stuck, _ = detector.check_stuck_patterns(msg)
            assert is_stuck, f"Should detect '{msg}' as stuck"
