"""Unit tests for Flutter Agent Settings module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from claude_code_agent_farm.flutter_agent_settings import (
    DockerEnvironment,
    FlutterAgentSettings,
    load_settings,
)


@pytest.mark.unit
class TestFlutterAgentSettings:
    """Test FlutterAgentSettings configuration and validation."""

    def test_default_settings(self):
        """Test settings with default values."""
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "is_dir", return_value=True):
                settings = FlutterAgentSettings()

                assert settings.project_path == Path("/home/wojtek/dev/carenji")
                assert settings.tmux_session == "claude-carenji"
                assert settings.check_interval == 5
                assert settings.idle_timeout == 300
                assert settings.wait_on_limit is True
                assert settings.firebase_project_id == "carenji-24ab8"
                assert settings.mcp_enabled is True

    def test_settings_from_env(self):
        """Test loading settings from environment variables."""
        env_vars = {
            "CLAUDE_PROJECT_PATH": "/custom/path",
            "CLAUDE_TMUX_SESSION": "custom-session",
            "CLAUDE_CHECK_INTERVAL": "10",
            "CLAUDE_FIREBASE_PROJECT_ID": "test-project",
            "CLAUDE_MCP_ENABLED": "false",
        }

        with patch.dict(os.environ, env_vars):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "is_dir", return_value=True):
                    settings = FlutterAgentSettings()

                    assert settings.project_path == Path("/custom/path")
                    assert settings.tmux_session == "custom-session"
                    assert settings.check_interval == 10
                    assert settings.firebase_project_id == "test-project"
                    assert settings.mcp_enabled is False

    def test_prompt_configuration(self, mock_carenji_project):
        """Test prompt configuration from file and text."""
        # Test with prompt text
        settings = FlutterAgentSettings(project_path=mock_carenji_project, prompt_text="Test carenji development")
        assert "Test carenji development" in settings.prompt
        assert "CLAUDE.md guidelines" in settings.prompt

        # Test with prompt file
        prompt_file = mock_carenji_project / "prompt.txt"
        prompt_file.write_text("Fix Flutter analyzer errors")

        settings = FlutterAgentSettings(project_path=mock_carenji_project, prompt_file=prompt_file)
        assert "Fix Flutter analyzer errors" in settings.prompt
        assert "CLAUDE.md guidelines" in settings.prompt

    def test_prompt_no_enhancement(self, mock_carenji_project):
        """Test that prompts already mentioning carenji aren't enhanced."""
        settings = FlutterAgentSettings(
            project_path=mock_carenji_project, prompt_text="Fix carenji app following CLAUDE.md"
        )
        assert settings.prompt == "Fix carenji app following CLAUDE.md"
        assert settings.prompt.count("CLAUDE.md") == 1

    def test_project_path_validation(self, temp_dir):
        """Test project path validation."""
        # Non-existent path
        with pytest.raises(ValidationError, match="does not exist"):
            FlutterAgentSettings(project_path=temp_dir / "nonexistent")

        # File instead of directory
        file_path = temp_dir / "file.txt"
        file_path.write_text("test")
        with pytest.raises(ValidationError, match="not a directory"):
            FlutterAgentSettings(project_path=file_path)

    def test_firebase_emulator_configuration(self, mock_carenji_project):
        """Test Firebase emulator settings."""
        settings = FlutterAgentSettings(project_path=mock_carenji_project, prompt_text="test")

        assert settings.firebase_emulator_ports["auth"] == 9098
        assert settings.firebase_emulator_ports["firestore"] == 8079
        assert settings.firebase_emulator_ports["functions"] == 5001
        assert settings.firebase_emulator_ports["ui"] == 4001
        assert settings.firebase_emulator_host == "127.0.0.1"

    def test_carenji_features(self, mock_carenji_project):
        """Test Carenji-specific feature configuration."""
        settings = FlutterAgentSettings(project_path=mock_carenji_project, prompt_text="test")

        assert "medication_management" in settings.carenji_features_enabled
        assert "vitals_monitoring" in settings.carenji_features_enabled
        assert "staff_scheduling" in settings.carenji_features_enabled
        assert "family_portal" in settings.carenji_features_enabled
        assert "barcode_scanning" in settings.carenji_features_enabled

        assert settings.carenji_test_coverage_threshold == 80
        assert settings.carenji_lint_on_save is True

    def test_mcp_configuration(self, mock_carenji_project):
        """Test Flutter MCP settings."""
        settings = FlutterAgentSettings(project_path=mock_carenji_project, prompt_text="test")

        assert settings.mcp_vmservice_port == 8182
        assert settings.mcp_dds_port == 8181
        assert settings.mcp_auto_detect is True

    def test_to_flutter_agent_config(self, mock_carenji_project):
        """Test conversion to config dict."""
        settings = FlutterAgentSettings(project_path=mock_carenji_project, prompt_text="test prompt")

        config = settings.to_flutter_agent_config()

        assert config["project_path"] == mock_carenji_project
        assert "test prompt" in config["prompt_text"]
        assert config["wait_on_limit"] is True
        assert config["restart_on_complete"] is True
        assert config["check_interval"] == 5

    def test_claude_config_paths(self, mock_carenji_project):
        """Test Claude configuration path detection."""
        settings = FlutterAgentSettings(project_path=mock_carenji_project, prompt_text="test")

        # Mock config file existence
        with patch.object(Path, "exists") as mock_exists:
            # First call for project path validation, rest for config paths
            mock_exists.side_effect = [True, True, False, False, False, False]

            assert settings.has_claude_config is True
            assert settings.get_claude_config_path() is not None


@pytest.mark.unit
class TestDockerEnvironment:
    """Test Docker environment settings."""

    def test_default_docker_env(self):
        """Test default Docker environment values."""
        env = DockerEnvironment()

        assert Path("/home/claude") == env.HOME
        assert env.USER == "claude"
        assert env.PYTHONUNBUFFERED == "1"
        assert env.CLAUDE_DANGEROUSLY_SKIP_PERMISSIONS == "1"

    def test_is_docker_detection(self):
        """Test Docker environment detection."""
        env = DockerEnvironment(HOSTNAME="container123")

        with patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = True
            assert env.is_docker is True

            mock_exists.return_value = False
            assert env.is_docker is False

    def test_container_info(self):
        """Test container info formatting."""
        # With container name
        env = DockerEnvironment(CONTAINER_NAME="claude-carenji")
        assert env.container_info == "Container: claude-carenji"

        # With hostname only
        env = DockerEnvironment(HOSTNAME="abc123def456")
        assert env.container_info == "Container ID: abc123def456"

        # Neither
        env = DockerEnvironment()
        assert env.container_info == "Docker Container"


@pytest.mark.unit
class TestLoadSettings:
    """Test the load_settings function."""

    def test_load_settings_success(self, mock_carenji_project):
        """Test successful settings loading."""
        with patch.dict(os.environ, {"CLAUDE_PROJECT_PATH": str(mock_carenji_project)}):
            settings = load_settings()
            assert settings.project_path == mock_carenji_project

    def test_load_settings_with_carenji_check(self, mock_carenji_project, capsys):
        """Test settings loading with carenji project validation."""
        # Create a non-carenji project
        other_project = mock_carenji_project.parent / "other_project"
        other_project.mkdir()
        (other_project / "pubspec.yaml").write_text("name: other_app\n")

        with patch.dict(os.environ, {"CLAUDE_PROJECT_PATH": str(other_project), "CLAUDE_PROMPT_TEXT": "test"}):
            settings = load_settings()

            # Should load but warn
            captured = capsys.readouterr()
            assert "doesn't appear to be the carenji project" in captured.out

    def test_load_settings_error_handling(self, capsys):
        """Test error handling in load_settings."""
        with patch.dict(os.environ, {"CLAUDE_PROJECT_PATH": "/nonexistent"}):
            with pytest.raises(SystemExit) as exc_info:
                load_settings()

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Configuration Error" in captured.out

    def test_load_settings_prompt_hints(self, temp_dir, capsys):
        """Test helpful hints when prompt is missing."""
        # Create a directory without prompt
        project_dir = temp_dir / "project"
        project_dir.mkdir()

        with patch.dict(os.environ, {"CLAUDE_PROJECT_PATH": str(project_dir)}):
            with pytest.raises(SystemExit):
                load_settings()

            captured = capsys.readouterr()
            assert "CLAUDE_PROMPT_TEXT" in captured.out
            assert "Fix all Flutter analyzer errors" in captured.out
            assert "Implement medication tracking" in captured.out
