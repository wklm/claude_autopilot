"""Unit tests for checkpoint and recovery functionality."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from claude_code_agent_farm.models import AgentStatus
from claude_code_agent_farm.models.checkpoint import (
    CheckpointData,
    CheckpointManager,
    RecoveryStrategy,
)


class TestCheckpointData:
    """Test checkpoint data structure."""
    
    def test_checkpoint_creation(self):
        """Test creating checkpoint data."""
        checkpoint = CheckpointData(
            session_id="test_session",
            prompt="Test prompt",
            status=AgentStatus.WORKING,
            last_activity=datetime.now(),
            total_runs=5,
            restart_count=2,
            usage_limit_hits=1,
            is_waiting_for_limit=False,
            wait_until=None
        )
        
        assert checkpoint.session_id == "test_session"
        assert checkpoint.prompt == "Test prompt"
        assert checkpoint.status == AgentStatus.WORKING
        assert checkpoint.total_runs == 5
        assert checkpoint.checkpoint_id is not None
    
    def test_checkpoint_with_state(self):
        """Test checkpoint with various state data."""
        checkpoint = CheckpointData(
            session_id="test_session",
            prompt="Test prompt",
            status=AgentStatus.USAGE_LIMIT,
            last_activity=datetime.now(),
            total_runs=10,
            restart_count=3,
            usage_limit_hits=2,
            is_waiting_for_limit=True,
            wait_until=datetime.now() + timedelta(hours=1),
            retry_strategy_state={"current_attempt": 2},
            health_monitor_state={"consecutive_failures": 0},
            events_summary={"restart": 3, "usage_limit": 2}
        )
        
        assert checkpoint.is_waiting_for_limit
        assert checkpoint.wait_until is not None
        assert checkpoint.retry_strategy_state["current_attempt"] == 2
        assert checkpoint.events_summary["restart"] == 3


class TestCheckpointManager:
    """Test checkpoint manager functionality."""
    
    @pytest.fixture
    def temp_checkpoint_dir(self, tmp_path):
        """Create temporary checkpoint directory."""
        return tmp_path / "checkpoints"
    
    @pytest.fixture
    def manager(self, temp_checkpoint_dir):
        """Create checkpoint manager with temp directory."""
        return CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            checkpoint_interval_seconds=60
        )
    
    def test_manager_initialization(self, manager, temp_checkpoint_dir):
        """Test manager creates directory."""
        assert temp_checkpoint_dir.exists()
        assert manager.checkpoint_dir == temp_checkpoint_dir
    
    def test_should_checkpoint_timing(self, manager):
        """Test checkpoint interval timing."""
        # Should checkpoint initially
        assert manager.should_checkpoint()
        
        # After checkpoint
        manager.last_checkpoint_time = datetime.now()
        assert not manager.should_checkpoint()
        
        # After interval
        manager.last_checkpoint_time = datetime.now() - timedelta(seconds=61)
        assert manager.should_checkpoint()
    
    def test_create_checkpoint(self, manager):
        """Test creating a checkpoint."""
        session_data = {
            "session_id": "test_session",
            "prompt": "Test prompt",
            "status": AgentStatus.WORKING,
            "last_activity": datetime.now(),
            "total_runs": 5,
            "restart_count": 1,
            "usage_limit_hits": 0,
            "is_waiting_for_limit": False,
            "wait_until": None
        }
        
        checkpoint = manager.create_checkpoint(session_data)
        
        assert checkpoint.session_id == "test_session"
        assert checkpoint.total_runs == 5
        assert manager.last_checkpoint_time is not None
        
        # Check in-memory cache
        assert "test_session" in manager.checkpoints
        assert len(manager.checkpoints["test_session"]) == 1
    
    def test_checkpoint_saved_to_disk(self, manager, temp_checkpoint_dir):
        """Test checkpoint is saved to disk."""
        session_data = {
            "session_id": "test_session",
            "prompt": "Test prompt",
            "status": AgentStatus.READY,
            "last_activity": datetime.now(),
            "total_runs": 3,
            "restart_count": 0,
            "usage_limit_hits": 0,
            "is_waiting_for_limit": False,
            "wait_until": None
        }
        
        checkpoint = manager.create_checkpoint(session_data)
        
        # Check file exists
        session_dir = temp_checkpoint_dir / "test_session"
        assert session_dir.exists()
        
        checkpoint_files = list(session_dir.glob("*.json"))
        assert len(checkpoint_files) == 1
        
        # Verify content
        content = json.loads(checkpoint_files[0].read_text())
        assert content["session_id"] == "test_session"
        assert content["total_runs"] == 3
    
    def test_max_checkpoints_limit(self, manager):
        """Test maximum checkpoints per session."""
        manager.max_checkpoints_per_session = 3
        
        # Create more than max checkpoints
        for i in range(5):
            session_data = {
                "session_id": "test_session",
                "prompt": "Test prompt",
                "status": AgentStatus.WORKING,
                "last_activity": datetime.now(),
                "total_runs": i,
                "restart_count": 0,
                "usage_limit_hits": 0,
                "is_waiting_for_limit": False,
                "wait_until": None
            }
            manager.create_checkpoint(session_data)
        
        # Should only keep most recent 3
        assert len(manager.checkpoints["test_session"]) == 3
        
        # Verify they are the most recent
        runs = [cp.total_runs for cp in manager.checkpoints["test_session"]]
        assert runs == [2, 3, 4]  # Last 3 values
    
    def test_restore_latest_checkpoint(self, manager):
        """Test restoring latest checkpoint."""
        # Create multiple checkpoints
        for i in range(3):
            session_data = {
                "session_id": "test_session",
                "prompt": "Test prompt",
                "status": AgentStatus.WORKING,
                "last_activity": datetime.now(),
                "total_runs": i * 10,
                "restart_count": i,
                "usage_limit_hits": 0,
                "is_waiting_for_limit": False,
                "wait_until": None
            }
            manager.create_checkpoint(session_data)
        
        # Restore latest
        restored = manager.restore_latest_checkpoint("test_session")
        
        assert restored is not None
        assert restored.total_runs == 20  # Latest value
        assert restored.restart_count == 2
    
    def test_restore_from_disk(self, manager, temp_checkpoint_dir):
        """Test restoring from disk when not in cache."""
        # Create checkpoint
        session_data = {
            "session_id": "disk_session",
            "prompt": "Test prompt",
            "status": AgentStatus.IDLE,
            "last_activity": datetime.now(),
            "total_runs": 42,
            "restart_count": 7,
            "usage_limit_hits": 3,
            "is_waiting_for_limit": False,
            "wait_until": None
        }
        checkpoint = manager.create_checkpoint(session_data)
        
        # Clear cache
        manager.checkpoints.clear()
        
        # Restore from disk
        restored = manager.restore_latest_checkpoint("disk_session")
        
        assert restored is not None
        assert restored.total_runs == 42
        assert restored.restart_count == 7
        assert restored.usage_limit_hits == 3
    
    def test_list_checkpoints(self, manager):
        """Test listing all checkpoints."""
        # Create checkpoints
        for i in range(3):
            session_data = {
                "session_id": "list_session",
                "prompt": "Test prompt",
                "status": AgentStatus.WORKING,
                "last_activity": datetime.now(),
                "total_runs": i,
                "restart_count": 0,
                "usage_limit_hits": 0,
                "is_waiting_for_limit": False,
                "wait_until": None
            }
            manager.create_checkpoint(session_data)
        
        checkpoints = manager.list_checkpoints("list_session")
        
        assert len(checkpoints) == 3
        assert all(cp.session_id == "list_session" for cp in checkpoints)
        assert [cp.total_runs for cp in checkpoints] == [0, 1, 2]
    
    def test_clean_old_checkpoints(self, manager, temp_checkpoint_dir):
        """Test cleaning old checkpoints."""
        # Create old checkpoint manually
        old_session_dir = temp_checkpoint_dir / "old_session"
        old_session_dir.mkdir()
        
        old_file = old_session_dir / "old_checkpoint.json"
        old_data = {
            "session_id": "old_session",
            "checkpoint_id": "old_001",
            "created_at": (datetime.now() - timedelta(days=10)).isoformat(),
            "prompt": "Old prompt",
            "status": "working",
            "last_activity": datetime.now().isoformat(),
            "total_runs": 1,
            "restart_count": 0,
            "usage_limit_hits": 0,
            "is_waiting_for_limit": False,
            "wait_until": None
        }
        old_file.write_text(json.dumps(old_data))
        
        # Make file old
        import os
        old_timestamp = (datetime.now() - timedelta(days=10)).timestamp()
        os.utime(old_file, (old_timestamp, old_timestamp))
        
        # Create recent checkpoint
        recent_data = {
            "session_id": "recent_session",
            "prompt": "Recent prompt",
            "status": AgentStatus.WORKING,
            "last_activity": datetime.now(),
            "total_runs": 1,
            "restart_count": 0,
            "usage_limit_hits": 0,
            "is_waiting_for_limit": False,
            "wait_until": None
        }
        manager.create_checkpoint(recent_data)
        
        # Clean old checkpoints (older than 7 days)
        removed = manager.clean_old_checkpoints(days=7)
        
        assert removed == 1
        assert not old_file.exists()
        assert not old_session_dir.exists()  # Empty dir removed


class TestRecoveryStrategy:
    """Test recovery strategy functionality."""
    
    def test_default_recovery_enabled(self):
        """Test recovery is enabled by default."""
        strategy = RecoveryStrategy()
        
        assert strategy.auto_recover
        assert strategy.recover_prompt
        assert strategy.recover_state
        assert strategy.recover_retry_strategy
    
    def test_should_recover_no_checkpoint(self):
        """Test recovery denied without checkpoint."""
        strategy = RecoveryStrategy()
        
        should_recover, reason = strategy.should_recover(None)
        
        assert not should_recover
        assert "No checkpoint available" in reason
    
    def test_should_recover_disabled(self):
        """Test recovery when disabled."""
        strategy = RecoveryStrategy(auto_recover=False)
        checkpoint = Mock(spec=CheckpointData)
        
        should_recover, reason = strategy.should_recover(checkpoint)
        
        assert not should_recover
        assert "Auto-recovery is disabled" in reason
    
    def test_should_recover_old_checkpoint(self):
        """Test recovery denied for old checkpoint."""
        strategy = RecoveryStrategy(recovery_timeout_seconds=300)
        
        checkpoint = CheckpointData(
            session_id="test",
            created_at=datetime.now() - timedelta(seconds=400),
            prompt="Test",
            status=AgentStatus.WORKING,
            last_activity=datetime.now(),
            total_runs=1,
            restart_count=0,
            usage_limit_hits=0,
            is_waiting_for_limit=False,
            wait_until=None
        )
        
        should_recover, reason = strategy.should_recover(checkpoint)
        
        assert not should_recover
        assert "Checkpoint is too old" in reason
        assert "400 seconds" in reason
    
    def test_should_recover_valid(self):
        """Test recovery allowed for valid checkpoint."""
        strategy = RecoveryStrategy()
        
        checkpoint = CheckpointData(
            session_id="test",
            created_at=datetime.now() - timedelta(seconds=60),
            prompt="Test",
            status=AgentStatus.WORKING,
            last_activity=datetime.now(),
            total_runs=1,
            restart_count=0,
            usage_limit_hits=0,
            is_waiting_for_limit=False,
            wait_until=None
        )
        
        should_recover, reason = strategy.should_recover(checkpoint)
        
        assert should_recover
        assert reason is None