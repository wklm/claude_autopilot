"""Tests for configuration management."""

import json
import tempfile
from pathlib import Path

import pytest

from claude_code_agent_farm.config import constants
from claude_code_agent_farm.config.settings import Settings


def test_constants():
    """Test that constants are properly defined."""
    assert constants.DEFAULT_NUM_AGENTS == 20
    assert constants.DEFAULT_SESSION_NAME == "claude_agents"
    assert constants.DEFAULT_CONTEXT_THRESHOLD == 20
    assert constants.STATUS_WORKING == "working"
    assert constants.STATUS_READY == "ready"


def test_settings_load_from_file():
    """Test loading settings from JSON file."""
    config_data = {
        "path": "/test/project",
        "agents": 10,
        "session": "test-session"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name
    
    try:
        settings = Settings()
        settings.load_from_file(temp_path)
        
        assert settings.get("path") == "/test/project"
        assert settings.get("agents") == 10
        assert settings.get("session") == "test-session"
    finally:
        Path(temp_path).unlink()


def test_settings_validate():
    """Test settings validation."""
    settings = Settings()
    
    # Valid settings
    settings.update(path="/test/project", agents=10, session="valid-session")
    settings.validate()  # Should not raise
    
    # Invalid: no path
    settings.config.clear()
    with pytest.raises(ValueError, match="must include 'path'"):
        settings.validate()
    
    # Invalid: too many agents
    settings.update(path="/test", agents=150)
    with pytest.raises(ValueError, match="between 1 and 100"):
        settings.validate()
    
    # Invalid: bad session name
    settings.update(path="/test", agents=10, session="invalid session!")
    with pytest.raises(ValueError, match="Invalid session name"):
        settings.validate()


def test_settings_get_with_default():
    """Test getting settings with default values."""
    settings = Settings()
    
    assert settings.get("nonexistent") is None
    assert settings.get("nonexistent", "default") == "default"
    
    settings.update(key="value")
    assert settings.get("key") == "value"