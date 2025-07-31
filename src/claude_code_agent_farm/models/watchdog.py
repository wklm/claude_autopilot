"""Watchdog timer for detecting hung agents."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


class WatchdogState(str, Enum):
    """Watchdog timer states."""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    PAUSED = "paused"
    STOPPED = "stopped"


class ActivityType(str, Enum):
    """Types of agent activity."""
    PANE_OUTPUT = "pane_output"
    STATUS_CHANGE = "status_change"
    USER_INPUT = "user_input"
    TASK_PROGRESS = "task_progress"
    HEARTBEAT = "heartbeat"


class AgentActivity(BaseModel):
    """Record of agent activity."""
    
    activity_type: ActivityType
    timestamp: datetime = Field(default_factory=datetime.now)
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)


class WatchdogTimer(BaseModel):
    """Watchdog timer for detecting hung or unresponsive agents."""
    
    # Configuration
    timeout_seconds: int = Field(default=300, ge=30, description="Timeout before considering agent hung")
    grace_period_seconds: int = Field(default=60, ge=0, description="Grace period after restart")
    check_interval_seconds: int = Field(default=10, ge=1, description="How often to check for timeout")
    
    # State
    state: WatchdogState = Field(default=WatchdogState.STOPPED)
    last_activity: Optional[datetime] = Field(default=None)
    last_check: Optional[datetime] = Field(default=None)
    timer_started: Optional[datetime] = Field(default=None)
    
    # Activity tracking
    activity_log: list[AgentActivity] = Field(default_factory=list)
    trigger_count: int = Field(default=0)
    last_trigger: Optional[datetime] = Field(default=None)
    
    # Content tracking for detecting real changes
    last_pane_content_hash: Optional[str] = Field(default=None)
    last_meaningful_change: Optional[datetime] = Field(default=None)
    
    def start(self) -> None:
        """Start the watchdog timer."""
        self.state = WatchdogState.ACTIVE
        self.timer_started = datetime.now()
        self.last_activity = datetime.now()
        self.last_check = datetime.now()
    
    def stop(self) -> None:
        """Stop the watchdog timer."""
        self.state = WatchdogState.STOPPED
        self.timer_started = None
    
    def pause(self) -> None:
        """Pause the watchdog timer."""
        if self.state == WatchdogState.ACTIVE:
            self.state = WatchdogState.PAUSED
    
    def resume(self) -> None:
        """Resume the watchdog timer."""
        if self.state == WatchdogState.PAUSED:
            self.state = WatchdogState.ACTIVE
            self.last_activity = datetime.now()  # Reset to avoid immediate trigger
    
    def feed(self, activity_type: ActivityType, description: str, **details) -> None:
        \"\"\"Feed the watchdog to indicate agent activity.\"\"\"\n        if self.state != WatchdogState.ACTIVE:\n            return\n        \n        activity = AgentActivity(\n            activity_type=activity_type,\n            description=description,\n            details=details\n        )\n        \n        self.activity_log.append(activity)\n        self.last_activity = activity.timestamp\n        \n        # Maintain log size\n        if len(self.activity_log) > 1000:\n            self.activity_log = self.activity_log[-500:]\n    \n    def check_timeout(self) -> tuple[bool, Optional[str]]:\n        \"\"\"Check if the watchdog has timed out.\"\"\"\n        if self.state != WatchdogState.ACTIVE:\n            return False, None\n        \n        now = datetime.now()\n        self.last_check = now\n        \n        # Check if in grace period\n        if self.timer_started and (now - self.timer_started).total_seconds() < self.grace_period_seconds:\n            return False, None\n        \n        # Check for timeout\n        if self.last_activity:\n            time_since_activity = (now - self.last_activity).total_seconds()\n            if time_since_activity > self.timeout_seconds:\n                self.state = WatchdogState.TRIGGERED\n                self.trigger_count += 1\n                self.last_trigger = now\n                return True, f\"No activity for {int(time_since_activity)} seconds\"\n        \n        return False, None\n    \n    def is_content_changed(self, content: str) -> bool:\n        \"\"\"Check if pane content has meaningfully changed.\"\"\"\n        import hashlib\n        \n        # Calculate hash of content\n        content_hash = hashlib.md5(content.encode()).hexdigest()\n        \n        # Check if content changed\n        if self.last_pane_content_hash != content_hash:\n            self.last_pane_content_hash = content_hash\n            self.last_meaningful_change = datetime.now()\n            return True\n        \n        return False\n    \n    def get_time_since_activity(self) -> Optional[timedelta]:\n        \"\"\"Get time since last activity.\"\"\"\n        if self.last_activity:\n            return datetime.now() - self.last_activity\n        return None\n    \n    def get_activity_summary(self, last_n_minutes: int = 5) -> Dict[ActivityType, int]:\n        \"\"\"Get summary of recent activity.\"\"\"\n        cutoff_time = datetime.now() - timedelta(minutes=last_n_minutes)\n        recent_activities = [a for a in self.activity_log if a.timestamp > cutoff_time]\n        \n        summary = {}\n        for activity in recent_activities:\n            summary[activity.activity_type] = summary.get(activity.activity_type, 0) + 1\n        \n        return summary\n\n\nclass HungAgentDetector(BaseModel):\n    \"\"\"Detector for various types of hung agent scenarios.\"\"\"\n    \n    # Detection thresholds\n    no_output_threshold: int = Field(default=300, description=\"Seconds without output\")\n    stuck_pattern_threshold: int = Field(default=3, description=\"Repetitions to consider stuck\")\n    memory_growth_threshold: float = Field(default=0.5, description=\"Memory growth ratio threshold\")\n    \n    # Pattern detection\n    repeated_patterns: Dict[str, int] = Field(default_factory=dict)\n    last_memory_check: Optional[float] = Field(default=None)\n    \n    def check_stuck_patterns(self, content: str) -> tuple[bool, Optional[str]]:\n        \"\"\"Check for stuck/repeated patterns in output.\"\"\"\n        # Look for common stuck patterns\n        stuck_indicators = [\n            \"Thinking...\",\n            \"Processing...\",\n            \"Loading...\",\n            \"Waiting for\",\n            \"Retrying\",\n        ]\n        \n        for pattern in stuck_indicators:\n            if pattern in content:\n                count = self.repeated_patterns.get(pattern, 0) + 1\n                self.repeated_patterns[pattern] = count\n                \n                if count >= self.stuck_pattern_threshold:\n                    return True, f\"Stuck on '{pattern}' (repeated {count} times)\"\n        \n        # Clear counts for patterns not in current content\n        for pattern in list(self.repeated_patterns.keys()):\n            if pattern not in content:\n                del self.repeated_patterns[pattern]\n        \n        return False, None