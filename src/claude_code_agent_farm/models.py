"""Legacy Pydantic models for Claude Single Agent Monitor.

This file contains legacy models that are being phased out in favor of
the new structured models in the models/ directory. These are kept for
backward compatibility during the transition.
"""

from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Agent status enumeration."""
    
    WORKING = "working"
    READY = "ready"
    IDLE = "idle"
    ERROR = "error"
    USAGE_LIMIT = "usage_limit"
    STARTING = "starting"
    UNKNOWN = "unknown"


class SingleAgentConfig(BaseModel):
    """Configuration for single-agent Docker monitoring."""
    
    # Project configuration
    project_path: Path = Field(..., description="Path to the project directory")
    prompt_text: str = Field(..., description="The prompt to send to Claude")
    
    # Behavior configuration
    wait_on_limit: bool = Field(default=True, description="Wait when usage limit is hit")
    restart_on_complete: bool = Field(default=True, description="Restart when task completes")
    restart_on_error: bool = Field(default=True, description="Restart when an error occurs")
    
    # Monitoring configuration
    check_interval: int = Field(default=5, ge=1, description="Seconds between status checks")
    idle_timeout: int = Field(default=300, ge=10, description="Seconds before considering agent idle")
    
    # tmux configuration
    tmux_session: str = Field(default="claude-agent", description="tmux session name")


class UsageLimitInfo(BaseModel):
    """Information about usage limit and retry time."""
    
    message: str = Field(..., description="The usage limit message from Claude")
    retry_time: Optional[datetime] = Field(default=None, description="When usage will be available")
    wait_duration: Optional[timedelta] = Field(default=None, description="How long to wait")
    
    @property
    def has_valid_time(self) -> bool:
        """Check if we have a valid retry time."""
        return self.retry_time is not None
    
    @property
    def wait_seconds(self) -> float:
        """Get wait duration in seconds."""
        if not self.wait_duration:
            return 0.0
        return max(0.0, self.wait_duration.total_seconds())


class AgentSession(BaseModel):
    """Track current agent session state."""
    
    session_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    started_at: datetime = Field(default_factory=datetime.now)
    prompt: str = Field(..., description="The prompt being used")
    
    # Status tracking
    status: AgentStatus = Field(default=AgentStatus.STARTING)
    last_activity: datetime = Field(default_factory=datetime.now)
    
    # Counters
    total_runs: int = Field(default=0, ge=0, description="Total number of runs")
    restart_count: int = Field(default=0, ge=0, description="Number of restarts")
    usage_limit_hits: int = Field(default=0, ge=0, description="Number of usage limit hits")
    
    # Usage limit tracking
    is_waiting_for_limit: bool = Field(default=False)
    wait_until: Optional[datetime] = Field(default=None)
    last_usage_limit: Optional[UsageLimitInfo] = Field(default=None)
    
    @property
    def runtime(self) -> str:
        """Get formatted runtime."""
        delta = datetime.now() - self.started_at
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    
    def record_usage_limit(self, limit_info: UsageLimitInfo) -> None:
        """Record a usage limit hit."""
        self.usage_limit_hits += 1
        self.last_usage_limit = limit_info
        self.is_waiting_for_limit = True
        if limit_info.retry_time:
            self.wait_until = limit_info.retry_time
        else:
            # Default to 1 hour if no time provided
            self.wait_until = datetime.now() + timedelta(hours=1)
    
    def record_restart(self) -> None:
        """Record an agent restart."""
        self.restart_count += 1
        self.last_activity = datetime.now()