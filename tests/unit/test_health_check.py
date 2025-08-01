"""Unit tests for health check and restart reliability."""

from datetime import datetime, timedelta

from claude_code_agent_farm.models_new.health_check import (
    AgentHealthMonitor,
    HealthCheckResult,
    HealthCheckType,
    HealthStatus,
    RestartAttemptTracker,
    RestartHealthCheck,
)


class TestHealthCheckResult:
    """Test health check result functionality."""

    def test_health_check_creation(self):
        """Test creating health check results."""
        result = HealthCheckResult(
            check_type=HealthCheckType.TMUX_SESSION,
            status=HealthStatus.HEALTHY,
            message="Tmux session is active",
            duration_ms=50,
            details={"session_name": "test"},
        )

        assert result.check_type == HealthCheckType.TMUX_SESSION
        assert result.status == HealthStatus.HEALTHY
        assert result.is_healthy
        assert result.duration_ms == 50
        assert result.details["session_name"] == "test"

    def test_unhealthy_check(self):
        """Test unhealthy check result."""
        result = HealthCheckResult(
            check_type=HealthCheckType.AGENT_RESPONSIVE, status=HealthStatus.UNHEALTHY, message="Agent not responding"
        )

        assert not result.is_healthy
        assert result.status == HealthStatus.UNHEALTHY


class TestRestartHealthCheck:
    """Test restart health check functionality."""

    def test_add_checks(self):
        """Test adding pre and post checks."""
        restart_check = RestartHealthCheck()

        # Add pre-restart checks
        pre_check = HealthCheckResult(
            check_type=HealthCheckType.DISK_SPACE, status=HealthStatus.HEALTHY, message="Sufficient disk space"
        )
        restart_check.add_pre_check(pre_check)

        # Add post-restart checks
        post_check = HealthCheckResult(
            check_type=HealthCheckType.AGENT_RESPONSIVE, status=HealthStatus.HEALTHY, message="Agent responding"
        )
        restart_check.add_post_check(post_check)

        assert len(restart_check.pre_restart_checks) == 1
        assert len(restart_check.post_restart_checks) == 1

    def test_all_checks_healthy(self):
        """Test checking if all checks are healthy."""
        restart_check = RestartHealthCheck()

        # All healthy
        restart_check.add_pre_check(
            HealthCheckResult(check_type=HealthCheckType.TMUX_SESSION, status=HealthStatus.HEALTHY, message="OK")
        )
        restart_check.add_pre_check(
            HealthCheckResult(check_type=HealthCheckType.DISK_SPACE, status=HealthStatus.HEALTHY, message="OK")
        )

        assert restart_check.all_pre_checks_healthy

        # One unhealthy
        restart_check.add_post_check(
            HealthCheckResult(
                check_type=HealthCheckType.AGENT_RESPONSIVE, status=HealthStatus.UNHEALTHY, message="Not responding"
            )
        )

        assert not restart_check.all_post_checks_healthy

    def test_restart_success_tracking(self):
        """Test tracking restart success."""
        restart_check = RestartHealthCheck()

        # Successful restart
        restart_check.restart_timestamp = datetime.now()
        restart_check.restart_duration_ms = 5000
        restart_check.restart_successful = True

        assert restart_check.restart_successful
        assert restart_check.restart_duration_ms == 5000

        # Failed restart
        restart_check.restart_successful = False
        restart_check.failure_reason = "Tmux pane died"

        assert not restart_check.restart_successful
        assert "Tmux pane died" in restart_check.failure_reason


class TestRestartAttemptTracker:
    """Test restart attempt tracking."""

    def test_can_restart_initial(self):
        """Test initial restart is allowed."""
        tracker = RestartAttemptTracker()

        can_restart, reason = tracker.can_restart()
        assert can_restart
        assert reason is None

    def test_max_attempts_limit(self):
        """Test max attempts limit."""
        tracker = RestartAttemptTracker(max_attempts=3)

        # Record attempts up to limit
        for _ in range(3):
            tracker.record_attempt()

        can_restart, reason = tracker.can_restart()
        assert not can_restart
        assert "Maximum restart attempts" in reason

    def test_cooldown_period(self):
        """Test cooldown between restarts."""
        tracker = RestartAttemptTracker(cooldown_seconds=60)

        # First attempt
        tracker.record_attempt()

        # Immediate retry should fail
        can_restart, reason = tracker.can_restart()
        assert not can_restart
        assert "Cooldown period active" in reason
        assert "60 seconds" in reason

    def test_attempt_window_cleanup(self):
        """Test old attempts are cleaned up."""
        tracker = RestartAttemptTracker(
            max_attempts=3,
            attempt_window_seconds=3600,  # 1 hour
        )

        # Add old attempts
        old_time = datetime.now() - timedelta(hours=2)
        tracker.attempts = [old_time, old_time]

        # Should clean up and allow restart
        can_restart, reason = tracker.can_restart()
        assert can_restart
        assert len(tracker.attempts) == 0

    def test_success_rate_calculation(self):
        """Test calculating restart success rate."""
        tracker = RestartAttemptTracker()

        # No attempts
        assert tracker.get_success_rate() == 0.0

        # Add successful attempts
        for _ in range(3):
            health_check = RestartHealthCheck(restart_successful=True)
            tracker.record_attempt(health_check)

        assert tracker.get_success_rate() == 1.0

        # Add failed attempt
        failed_check = RestartHealthCheck(restart_successful=False)
        tracker.record_attempt(failed_check)

        assert tracker.get_success_rate() == 0.75  # 3/4


class TestAgentHealthMonitor:
    """Test agent health monitoring."""

    def test_initial_state(self):
        """Test initial monitor state."""
        monitor = AgentHealthMonitor()

        assert monitor.current_status == HealthStatus.UNKNOWN
        assert monitor.consecutive_failures == 0
        assert monitor.last_check_time is None

    def test_record_healthy_check(self):
        """Test recording healthy check."""
        monitor = AgentHealthMonitor()

        result = HealthCheckResult(
            check_type=HealthCheckType.AGENT_RESPONSIVE, status=HealthStatus.HEALTHY, message="Agent is healthy"
        )

        monitor.record_check(result)

        assert monitor.current_status == HealthStatus.HEALTHY
        assert monitor.consecutive_failures == 0
        assert len(monitor.check_history) == 1

    def test_degraded_status(self):
        """Test degraded status after failures."""
        monitor = AgentHealthMonitor(unhealthy_threshold=3)

        # First failure
        monitor.record_check(
            HealthCheckResult(
                check_type=HealthCheckType.AGENT_RESPONSIVE, status=HealthStatus.UNHEALTHY, message="Not responding"
            )
        )

        assert monitor.current_status == HealthStatus.DEGRADED
        assert monitor.consecutive_failures == 1

        # Second failure
        monitor.record_check(
            HealthCheckResult(
                check_type=HealthCheckType.AGENT_RESPONSIVE,
                status=HealthStatus.UNHEALTHY,
                message="Still not responding",
            )
        )

        assert monitor.current_status == HealthStatus.DEGRADED
        assert monitor.consecutive_failures == 2

    def test_unhealthy_status_threshold(self):
        """Test unhealthy status after threshold."""
        monitor = AgentHealthMonitor(unhealthy_threshold=3)

        # Fail up to threshold
        for _ in range(3):
            monitor.record_check(
                HealthCheckResult(
                    check_type=HealthCheckType.AGENT_RESPONSIVE, status=HealthStatus.UNHEALTHY, message="Failed"
                )
            )

        assert monitor.current_status == HealthStatus.UNHEALTHY
        assert monitor.consecutive_failures == 3

    def test_recovery_resets_failures(self):
        """Test recovery resets failure count."""
        monitor = AgentHealthMonitor()

        # Some failures
        monitor.consecutive_failures = 5
        monitor.current_status = HealthStatus.UNHEALTHY

        # Healthy check
        monitor.record_check(
            HealthCheckResult(
                check_type=HealthCheckType.AGENT_RESPONSIVE, status=HealthStatus.HEALTHY, message="Recovered"
            )
        )

        assert monitor.consecutive_failures == 0
        assert monitor.current_status == HealthStatus.HEALTHY

    def test_status_change_tracking(self):
        """Test tracking status changes."""
        monitor = AgentHealthMonitor()

        # Initial healthy
        monitor.record_check(
            HealthCheckResult(check_type=HealthCheckType.AGENT_RESPONSIVE, status=HealthStatus.HEALTHY, message="OK")
        )

        # Change to unhealthy
        monitor.record_check(
            HealthCheckResult(
                check_type=HealthCheckType.AGENT_RESPONSIVE, status=HealthStatus.UNHEALTHY, message="Failed"
            )
        )

        assert len(monitor.status_changes) == 2

        # Check recorded changes
        first_change = monitor.status_changes[0]
        assert first_change[1] == HealthStatus.UNKNOWN  # From
        assert first_change[2] == HealthStatus.HEALTHY  # To

        second_change = monitor.status_changes[1]
        assert second_change[1] == HealthStatus.HEALTHY  # From
        assert second_change[2] == HealthStatus.DEGRADED  # To

    def test_should_check_timing(self):
        """Test check interval timing."""
        monitor = AgentHealthMonitor(check_interval_seconds=30)

        # Should check initially
        assert monitor.should_check()

        # Record a check
        monitor.last_check_time = datetime.now()

        # Should not check immediately
        assert not monitor.should_check()

        # Should check after interval
        monitor.last_check_time = datetime.now() - timedelta(seconds=31)
        assert monitor.should_check()

    def test_history_limit(self):
        """Test check history is limited."""
        monitor = AgentHealthMonitor()

        # Add many checks
        for i in range(150):
            monitor.record_check(
                HealthCheckResult(
                    check_type=HealthCheckType.AGENT_RESPONSIVE, status=HealthStatus.HEALTHY, message=f"Check {i}"
                )
            )

        # Should maintain limit
        assert len(monitor.check_history) == 100
