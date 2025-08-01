"""Shared pytest fixtures and configuration for Claude Flutter Firebase Agent tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pytest

from claude_code_agent_farm.flutter_agent_settings import FlutterAgentSettings


# Test environment setup
@pytest.fixture(autouse=True)
def test_environment():
    """Set up test environment variables."""
    original_env = os.environ.copy()

    # Set test-specific environment variables
    os.environ["CLAUDE_FLUTTER_AGENT_TESTING"] = "1"
    os.environ["CLAUDE_DANGEROUSLY_SKIP_PERMISSIONS"] = "1"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_carenji_project(temp_dir: Path) -> Path:
    """Create a mock Carenji project structure."""
    project_dir = temp_dir / "carenji"
    project_dir.mkdir()

    # Create pubspec.yaml
    pubspec_content = """name: carenji
description: "carenji app."
version: 1.0.0+1

environment:
  sdk: ">=3.4.0"

dependencies:
  flutter:
    sdk: flutter
  firebase_core: ^3.9.0
  firebase_auth: ^5.3.3
  cloud_firestore: ^5.6.0
  built_value: ^8.10.1
  injectable: ^2.5.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  build_runner: ^2.4.15
  injectable_generator: ^2.7.0
  
flutter:
  uses-material-design: true
"""
    (project_dir / "pubspec.yaml").write_text(pubspec_content)

    # Create firebase.json
    firebase_content = """{
  "firestore": {
    "rules": "firestore.rules",
    "indexes": "firestore.indexes.json"
  },
  "emulators": {
    "auth": {"host": "127.0.0.1", "port": 9098},
    "firestore": {"host": "127.0.0.1", "port": 8079},
    "functions": {"host": "127.0.0.1", "port": 5001},
    "ui": {"enabled": true, "host": "127.0.0.1", "port": 4001}
  },
  "flutter": {
    "platforms": {
      "dart": {
        "lib/firebase_options.dart": {
          "projectId": "carenji-24ab8"
        }
      }
    }
  }
}"""
    (project_dir / "firebase.json").write_text(firebase_content)

    # Create CLAUDE.md
    claude_md_content = """# CLAUDE.md - AI Development Notes
This file contains important notes for AI assistants working on Carenji.

## Project Overview
Carenji is a comprehensive healthcare management system for nursing homes.

## Key Features
- Medication Management
- Vital Signs Monitoring  
- Staff Scheduling
- Family Portal
"""
    (project_dir / "CLAUDE.md").write_text(claude_md_content)

    # Create lib directory structure
    lib_dir = project_dir / "lib"
    lib_dir.mkdir()
    (lib_dir / "main.dart").write_text("void main() {}")

    # Create test directory
    test_dir = project_dir / "test"
    test_dir.mkdir()
    (test_dir / "widget_test.dart").write_text("// Widget tests")

    return project_dir


@pytest.fixture
def mock_settings(mock_carenji_project: Path) -> FlutterAgentSettings:
    """Create mock Flutter agent settings."""
    return FlutterAgentSettings(
        project_path=mock_carenji_project,
        prompt_text="Test prompt for carenji development",
        tmux_session="test-session",
        check_interval=1,
        idle_timeout=10,
        wait_on_limit=True,
        restart_on_complete=True,
        restart_on_error=True,
    )


@pytest.fixture
def mock_claude_cli():
    """Mock the Claude CLI command."""
    with patch("subprocess.run") as mock_run:
        # Default successful response
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        yield mock_run


@pytest.fixture
def mock_tmux():
    """Mock tmux commands."""
    with patch("subprocess.run") as mock_run:

        def tmux_side_effect(cmd, **kwargs):
            result = Mock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""

            # Handle different tmux commands
            if isinstance(cmd, list) and len(cmd) > 1:
                if cmd[1] == "capture-pane":
                    result.stdout = ">> Ready for input\n"
                elif cmd[1] == "list-sessions":
                    result.stdout = "test-session: 1 windows"
            elif isinstance(cmd, str):
                if "capture-pane" in cmd:
                    result.stdout = ">> Ready for input\n"

            return result

        mock_run.side_effect = tmux_side_effect
        yield mock_run


@pytest.fixture
def mock_docker():
    """Mock Docker commands."""
    with patch("subprocess.run") as mock_run:

        def docker_side_effect(cmd, **kwargs):
            result = Mock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""

            if isinstance(cmd, list) and "docker" in cmd:
                if "ps" in cmd:
                    result.stdout = "claude-carenji-agent"
                elif "images" in cmd:
                    result.stdout = "claude-flutter-firebase-agent"

            return result

        mock_run.side_effect = docker_side_effect
        yield mock_run


@pytest.fixture
def usage_limit_response() -> str:
    """Sample usage limit response from Claude."""
    return """Your usage limit has been reached. Please try again at 3:45 PM PST.

For more information about usage limits, visit claude.ai/usage"""


@pytest.fixture
def error_response() -> str:
    """Sample error response from Claude."""
    return """Error: An unexpected error occurred while processing your request.
Please try again or contact support if the issue persists."""


@pytest.fixture
def working_response() -> str:
    """Sample working response from Claude."""
    return """Analyzing the carenji project structure...

I can see this is a Flutter healthcare app with Firebase backend. 
Let me examine the codebase to understand the architecture better."""


@pytest.fixture
def task_complete_response() -> str:
    """Sample task completion response from Claude."""
    return """I've completed the requested task:

✓ Fixed all Flutter analyzer errors
✓ Updated imports to follow conventions  
✓ Added missing type annotations
✓ Ran tests - all passing with 85% coverage

The carenji app is now ready for deployment.
>> """


# Markers for different test types
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no external dependencies)")
    config.addinivalue_line("markers", "integration: Integration tests (may require Docker)")
    config.addinivalue_line("markers", "e2e: End-to-end tests (slow, full system tests)")
    config.addinivalue_line("markers", "docker: Tests requiring Docker")
    config.addinivalue_line("markers", "slow: Slow running tests (>5s)")
    config.addinivalue_line("markers", "carenji: Tests specific to Carenji app functionality")
    config.addinivalue_line("markers", "firebase: Tests requiring Firebase emulators")
    config.addinivalue_line("markers", "mcp: Tests for Flutter MCP integration")
