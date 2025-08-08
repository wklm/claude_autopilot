"""Checkpoint models for session state persistence and recovery."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from claude_code_agent_farm.models_new.session import AgentStatus
from claude_code_agent_farm.utils import now_utc


class CheckpointData(BaseModel):
    """Data structure for session checkpoints."""

    # Session identification
    session_id: str
    checkpoint_id: str = Field(default_factory=lambda: now_utc().strftime("%Y%m%d_%H%M%S_%f"))
    created_at: datetime = Field(default_factory=now_utc)

    # Session state
    prompt: str
    status: AgentStatus
    last_activity: datetime

    # Counters
    total_runs: int
    restart_count: int
    usage_limit_hits: int

    # Usage limit state
    is_waiting_for_limit: bool
    wait_until: datetime | None
    last_usage_limit: dict[str, Any] | None  # Serialized UsageLimitInfo

    # Retry strategy state
    retry_strategy_state: dict[str, Any] | None = None

    # Health and watchdog state
    health_monitor_state: dict[str, Any] | None = None
    restart_tracker_state: dict[str, Any] | None = None
    watchdog_state: dict[str, Any] | None = None

    # Additional context
    events_summary: dict[str, Any] = Field(default_factory=dict)
    last_pane_content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointManager(BaseModel):
    """Manage checkpoint creation and restoration."""

    # Configuration
    checkpoint_dir: Path = Field(default=Path.home() / ".claude_agent_farm" / "checkpoints")
    max_checkpoints_per_session: int = Field(default=10, ge=1)
    checkpoint_interval_seconds: int = Field(default=300, ge=30)  # 5 minutes

    # State
    last_checkpoint_time: datetime | None = Field(default=None)
    checkpoints: dict[str, list[CheckpointData]] = Field(default_factory=dict)  # session_id -> checkpoints

    def __init__(self, **kwargs):
        """Initialize checkpoint manager and ensure directory exists."""
        super().__init__(**kwargs)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def should_checkpoint(self) -> bool:
        """Check if it's time to create a checkpoint."""
        if self.last_checkpoint_time is None:
            return True

        time_since_last = (now_utc() - self.last_checkpoint_time).total_seconds()
        return time_since_last >= self.checkpoint_interval_seconds

    def create_checkpoint(self, session_data: dict[str, Any]) -> CheckpointData:
        """Create a new checkpoint from session data."""
        checkpoint = CheckpointData(**session_data)

        # Add to in-memory cache
        session_id = checkpoint.session_id
        if session_id not in self.checkpoints:
            self.checkpoints[session_id] = []

        self.checkpoints[session_id].append(checkpoint)

        # Maintain max checkpoints
        if len(self.checkpoints[session_id]) > self.max_checkpoints_per_session:
            # Remove oldest checkpoints
            self.checkpoints[session_id] = self.checkpoints[session_id][-self.max_checkpoints_per_session :]

        # Save to disk
        self._save_checkpoint(checkpoint)

        self.last_checkpoint_time = now_utc()
        return checkpoint

    def _save_checkpoint(self, checkpoint: CheckpointData) -> None:
        """Save checkpoint to disk."""
        session_dir = self.checkpoint_dir / checkpoint.session_id
        session_dir.mkdir(exist_ok=True)

        checkpoint_file = session_dir / f"{checkpoint.checkpoint_id}.json"
        checkpoint_file.write_text(checkpoint.model_dump_json(indent=2))

    def restore_latest_checkpoint(self, session_id: str) -> CheckpointData | None:
        """Restore the latest checkpoint for a session."""
        # Check in-memory cache first
        if self.checkpoints.get(session_id):
            return self.checkpoints[session_id][-1]

        # Load from disk
        session_dir = self.checkpoint_dir / session_id
        if not session_dir.exists():
            return None

        checkpoint_files = sorted(session_dir.glob("*.json"))
        if not checkpoint_files:
            return None

        # Load latest checkpoint
        latest_file = checkpoint_files[-1]
        try:
            data = json.loads(latest_file.read_text())
            return CheckpointData(**data)
        except Exception as e:
            print(f"Failed to load checkpoint {latest_file}: {e}")
            return None

    def list_checkpoints(self, session_id: str) -> list[CheckpointData]:
        """List all checkpoints for a session."""
        # Load from disk if not in cache
        if session_id not in self.checkpoints:
            session_dir = self.checkpoint_dir / session_id
            if session_dir.exists():
                checkpoints = []
                for file in sorted(session_dir.glob("*.json")):
                    try:
                        data = json.loads(file.read_text())
                        checkpoints.append(CheckpointData(**data))
                    except Exception:
                        continue
                self.checkpoints[session_id] = checkpoints

        return self.checkpoints.get(session_id, [])

    def clean_old_checkpoints(self, days: int = 7) -> int:
        """Clean checkpoints older than specified days."""
        cutoff_time = now_utc().timestamp() - (days * 24 * 60 * 60)
        removed_count = 0

        for session_dir in self.checkpoint_dir.iterdir():
            if not session_dir.is_dir():
                continue

            for checkpoint_file in session_dir.glob("*.json"):
                if checkpoint_file.stat().st_mtime < cutoff_time:
                    checkpoint_file.unlink()
                    removed_count += 1

            # Remove empty directories
            if not list(session_dir.iterdir()):
                session_dir.rmdir()

        return removed_count


class RecoveryStrategy(BaseModel):
    """Strategy for recovering from crashes or interruptions."""

    # Recovery options
    auto_recover: bool = Field(default=True, description="Automatically recover from last checkpoint")
    recover_prompt: bool = Field(default=True, description="Re-send the original prompt")
    recover_state: bool = Field(default=True, description="Restore counters and state")
    recover_retry_strategy: bool = Field(default=True, description="Restore retry strategy state")

    # Recovery behavior
    max_recovery_attempts: int = Field(default=3, ge=1)
    recovery_timeout_seconds: int = Field(default=300, ge=30)

    def should_recover(self, checkpoint: CheckpointData | None) -> tuple[bool, str | None]:
        """Determine if recovery should be attempted."""
        if not self.auto_recover:
            return False, "Auto-recovery is disabled"

        if checkpoint is None:
            return False, "No checkpoint available"

        # Check if checkpoint is too old
        age_seconds = (now_utc() - checkpoint.created_at).total_seconds()
        if age_seconds > self.recovery_timeout_seconds:
            return False, f"Checkpoint is too old ({int(age_seconds)} seconds)"

        return True, None
