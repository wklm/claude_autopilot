"""Health check models for monitoring agent reliability."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from claude_code_agent_farm.utils import now_utc


class HealthStatus(str, Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheckType(str, Enum):
    """Types of health checks."""

    TMUX_SESSION = "tmux_session"
    AGENT_RESPONSIVE = "agent_responsive"
    MEMORY_USAGE = "memory_usage"
    DISK_SPACE = "disk_space"
    NETWORK_CONNECTIVITY = "network_connectivity"
    PROCESS_ALIVE = "process_alive"


class HealthCheckResult(BaseModel):
    """Result of a single health check."""

    check_type: HealthCheckType
    status: HealthStatus
    message: str
    timestamp: datetime = Field(default_factory=now_utc)
    duration_ms: int | None = Field(default=None, description="Check duration in milliseconds")
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Check if the result indicates health."""
        return self.status == HealthStatus.HEALTHY


class RestartHealthCheck(BaseModel):
    """Health checks specifically for restart operations."""

    pre_restart_checks: list[HealthCheckResult] = Field(default_factory=list)
    post_restart_checks: list[HealthCheckResult] = Field(default_factory=list)
    restart_timestamp: datetime | None = Field(default=None)
    restart_duration_ms: int | None = Field(default=None)
    restart_successful: bool = Field(default=False)
    failure_reason: str | None = Field(default=None)

    def add_pre_check(self, result: HealthCheckResult) -> None:
        """Add a pre-restart health check result."""
        self.pre_restart_checks.append(result)

    def add_post_check(self, result: HealthCheckResult) -> None:
        """Add a post-restart health check result."""
        self.post_restart_checks.append(result)

    @property
    def all_pre_checks_healthy(self) -> bool:
        """Check if all pre-restart checks are healthy."""
        return all(check.is_healthy for check in self.pre_restart_checks)

    @property
    def all_post_checks_healthy(self) -> bool:
        """Check if all post-restart checks are healthy."""
        return all(check.is_healthy for check in self.post_restart_checks)


class RestartAttemptTracker(BaseModel):
    """Track restart attempts with cooldown and limits."""

    max_attempts: int = Field(default=5, ge=1, description="Maximum restart attempts")
    cooldown_seconds: int = Field(default=300, ge=0, description="Cooldown between restart bursts")
    attempt_window_seconds: int = Field(default=3600, ge=60, description="Time window for attempts")

    attempts: list[datetime] = Field(default_factory=list)
    health_checks: list[RestartHealthCheck] = Field(default_factory=list)

    def can_restart(self) -> tuple[bool, str | None]:
        """Check if a restart is allowed based on limits and cooldown."""
        now = now_utc()

        # Clean up old attempts outside the window
        cutoff_time = now - timedelta(seconds=self.attempt_window_seconds)
        self.attempts = [t for t in self.attempts if t > cutoff_time]

        # Check if we've hit the limit
        if len(self.attempts) >= self.max_attempts:
            return (
                False,
                f"Maximum restart attempts ({self.max_attempts}) reached in the last {self.attempt_window_seconds} seconds",
            )

        # Check cooldown
        if self.attempts:
            last_attempt = max(self.attempts)
            time_since_last = (now - last_attempt).total_seconds()
            if time_since_last < self.cooldown_seconds:
                remaining = int(self.cooldown_seconds - time_since_last)
                return False, f"Cooldown period active. Wait {remaining} seconds before next restart"

        return True, None

    def record_attempt(self, health_check: RestartHealthCheck | None = None) -> None:
        """Record a restart attempt."""
        self.attempts.append(now_utc())
        if health_check:
            self.health_checks.append(health_check)

    def get_success_rate(self) -> float:
        """Get the success rate of recent restarts."""
        if not self.health_checks:
            return 0.0

        successful = sum(1 for hc in self.health_checks if hc.restart_successful)
        return successful / len(self.health_checks)


class AgentHealthMonitor(BaseModel):
    """Overall health monitoring for the agent."""

    # Health check configuration
    check_interval_seconds: int = Field(default=30, ge=5)
    unhealthy_threshold: int = Field(default=3, ge=1, description="Consecutive failures before unhealthy")

    # Current state
    last_check_time: datetime | None = Field(default=None)
    consecutive_failures: int = Field(default=0)
    current_status: HealthStatus = Field(default=HealthStatus.UNKNOWN)

    # History
    check_history: list[HealthCheckResult] = Field(default_factory=list)
    status_changes: list[tuple[datetime, HealthStatus, HealthStatus]] = Field(default_factory=list)

    def record_check(self, result: HealthCheckResult) -> None:
        """Record a health check result and update status."""
        self.last_check_time = now_utc()
        self.check_history.append(result)

        # Maintain history limit
        if len(self.check_history) > 100:
            self.check_history = self.check_history[-100:]

        # Update consecutive failures
        if result.is_healthy:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

        # Update status
        old_status = self.current_status
        if self.consecutive_failures == 0:
            self.current_status = HealthStatus.HEALTHY
        elif self.consecutive_failures < self.unhealthy_threshold:
            self.current_status = HealthStatus.DEGRADED
        else:
            self.current_status = HealthStatus.UNHEALTHY

        # Record status change
        if old_status != self.current_status:
            self.status_changes.append((now_utc(), old_status, self.current_status))

    def should_check(self) -> bool:
        """Check if it's time for another health check."""
        if self.last_check_time is None:
            return True

        time_since_last = (now_utc() - self.last_check_time).total_seconds()
        return time_since_last >= self.check_interval_seconds
