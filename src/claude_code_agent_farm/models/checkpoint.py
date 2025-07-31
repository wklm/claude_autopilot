"""Checkpoint models for session state persistence and recovery."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field

from claude_code_agent_farm.models import AgentStatus, UsageLimitInfo


class CheckpointData(BaseModel):
    """Data structure for session checkpoints."""
    
    # Session identification
    session_id: str
    checkpoint_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    created_at: datetime = Field(default_factory=datetime.now)
    
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
    wait_until: Optional[datetime]
    last_usage_limit: Optional[Dict[str, Any]]  # Serialized UsageLimitInfo
    
    # Retry strategy state
    retry_strategy_state: Optional[Dict[str, Any]] = None
    
    # Health and watchdog state
    health_monitor_state: Optional[Dict[str, Any]] = None
    restart_tracker_state: Optional[Dict[str, Any]] = None
    watchdog_state: Optional[Dict[str, Any]] = None
    
    # Additional context
    events_summary: Dict[str, int] = Field(default_factory=dict)
    last_pane_content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CheckpointManager(BaseModel):
    """Manage checkpoint creation and restoration."""
    
    # Configuration
    checkpoint_dir: Path = Field(default=Path.home() / ".claude_agent_farm" / "checkpoints")
    max_checkpoints_per_session: int = Field(default=10, ge=1)
    checkpoint_interval_seconds: int = Field(default=300, ge=30)  # 5 minutes
    
    # State
    last_checkpoint_time: Optional[datetime] = Field(default=None)
    checkpoints: Dict[str, list[CheckpointData]] = Field(default_factory=dict)  # session_id -> checkpoints
    
    def __init__(self, **kwargs):
        """Initialize checkpoint manager and ensure directory exists."""
        super().__init__(**kwargs)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def should_checkpoint(self) -> bool:
        """Check if it's time to create a checkpoint."""
        if self.last_checkpoint_time is None:
            return True
        
        time_since_last = (datetime.now() - self.last_checkpoint_time).total_seconds()
        return time_since_last >= self.checkpoint_interval_seconds
    
    def create_checkpoint(self, session_data: Dict[str, Any]) -> CheckpointData:
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
            self.checkpoints[session_id] = self.checkpoints[session_id][-self.max_checkpoints_per_session:]
        
        # Save to disk
        self._save_checkpoint(checkpoint)
        
        self.last_checkpoint_time = datetime.now()
        return checkpoint
    
    def _save_checkpoint(self, checkpoint: CheckpointData) -> None:
        """Save checkpoint to disk."""
        session_dir = self.checkpoint_dir / checkpoint.session_id
        session_dir.mkdir(exist_ok=True)
        
        checkpoint_file = session_dir / f"{checkpoint.checkpoint_id}.json"
        checkpoint_file.write_text(checkpoint.model_dump_json(indent=2))
    
    def restore_latest_checkpoint(self, session_id: str) -> Optional[CheckpointData]:
        """Restore the latest checkpoint for a session."""
        # Check in-memory cache first
        if session_id in self.checkpoints and self.checkpoints[session_id]:
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
        \"\"\"List all checkpoints for a session.\"\"\"\n        # Load from disk if not in cache\n        if session_id not in self.checkpoints:\n            session_dir = self.checkpoint_dir / session_id\n            if session_dir.exists():\n                checkpoints = []\n                for file in sorted(session_dir.glob(\"*.json\")):\n                    try:\n                        data = json.loads(file.read_text())\n                        checkpoints.append(CheckpointData(**data))\n                    except Exception:\n                        continue\n                self.checkpoints[session_id] = checkpoints\n        \n        return self.checkpoints.get(session_id, [])\n    \n    def clean_old_checkpoints(self, days: int = 7) -> int:\n        \"\"\"Clean checkpoints older than specified days.\"\"\"\n        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)\n        removed_count = 0\n        \n        for session_dir in self.checkpoint_dir.iterdir():\n            if not session_dir.is_dir():\n                continue\n            \n            for checkpoint_file in session_dir.glob(\"*.json\"):\n                if checkpoint_file.stat().st_mtime < cutoff_time:\n                    checkpoint_file.unlink()\n                    removed_count += 1\n            \n            # Remove empty directories\n            if not list(session_dir.iterdir()):\n                session_dir.rmdir()\n        \n        return removed_count\n\n\nclass RecoveryStrategy(BaseModel):\n    \"\"\"Strategy for recovering from crashes or interruptions.\"\"\"\n    \n    # Recovery options\n    auto_recover: bool = Field(default=True, description=\"Automatically recover from last checkpoint\")\n    recover_prompt: bool = Field(default=True, description=\"Re-send the original prompt\")\n    recover_state: bool = Field(default=True, description=\"Restore counters and state\")\n    recover_retry_strategy: bool = Field(default=True, description=\"Restore retry strategy state\")\n    \n    # Recovery behavior\n    max_recovery_attempts: int = Field(default=3, ge=1)\n    recovery_timeout_seconds: int = Field(default=300, ge=30)\n    \n    def should_recover(self, checkpoint: Optional[CheckpointData]) -> tuple[bool, Optional[str]]:\n        \"\"\"Determine if recovery should be attempted.\"\"\"\n        if not self.auto_recover:\n            return False, \"Auto-recovery is disabled\"\n        \n        if checkpoint is None:\n            return False, \"No checkpoint available\"\n        \n        # Check if checkpoint is too old\n        age_seconds = (datetime.now() - checkpoint.created_at).total_seconds()\n        if age_seconds > self.recovery_timeout_seconds:\n            return False, f\"Checkpoint is too old ({int(age_seconds)} seconds)\"\n        \n        return True, None