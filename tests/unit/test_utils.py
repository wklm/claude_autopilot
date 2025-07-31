"""Unit tests for utility functions."""

import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

import pytest
import pytz

from claude_code_agent_farm.utils import (
    UsageLimitTimeParser,
    run,
    check_flutter_project,
    check_firebase_project,
    check_carenji_project,
    get_firebase_emulator_status,
    start_firebase_emulators,
    get_flutter_mcp_command,
    format_duration,
    get_carenji_prompt_template,
)


@pytest.mark.unit
class TestUsageLimitTimeParser:
    """Test usage limit time parsing functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return UsageLimitTimeParser()
    
    def test_parse_time_with_timezone(self, parser):
        """Test parsing time with various timezones."""
        # Test PST
        message = "Usage limit reached. Please try again at 3:45 PM PST"
        result = parser.parse_usage_limit_message(message)
        assert result is not None
        assert result.hour == 15  # 3 PM in 24-hour format
        assert result.minute == 45
        
        # Test EST
        message = "Retry after 10:30 AM EST"
        result = parser.parse_usage_limit_message(message)
        assert result is not None
        assert result.hour == 10
        assert result.minute == 30
    
    def test_parse_duration_based_limits(self, parser):
        """Test parsing duration-based wait times."""
        # Hours
        message = "Please wait 2 hours before retrying"
        result = parser.parse_usage_limit_message(message)
        assert result is not None
        # Should be approximately 2 hours from now
        expected = datetime.now() + timedelta(hours=2)
        assert abs((result - expected).total_seconds()) < 60  # Within 1 minute
        
        # Minutes
        message = "Try again in 30 minutes"
        result = parser.parse_usage_limit_message(message)
        assert result is not None
        expected = datetime.now() + timedelta(minutes=30)
        assert abs((result - expected).total_seconds()) < 60
        
        # Hours and minutes
        message = "Retry in 1 hour and 45 minutes"
        result = parser.parse_usage_limit_message(message)
        assert result is not None
        expected = datetime.now() + timedelta(hours=1, minutes=45)
        assert abs((result - expected).total_seconds()) < 60
    
    def test_get_wait_duration(self, parser):
        """Test extracting wait duration from messages."""
        # Just hours
        duration = parser.get_wait_duration("wait 3 hours")
        assert duration == timedelta(hours=3)
        
        # Just minutes
        duration = parser.get_wait_duration("wait 45 minutes")
        assert duration == timedelta(minutes=45)
        
        # No duration found
        duration = parser.get_wait_duration("no time information here")
        assert duration is None
    
    def test_format_retry_time(self, parser):
        """Test formatting retry times for display."""
        now = datetime.now()
        
        # Today
        retry_time = now + timedelta(hours=2)
        formatted = parser.format_retry_time(retry_time)
        assert "today at" in formatted
        
        # Tomorrow
        retry_time = now + timedelta(days=1, hours=2)
        formatted = parser.format_retry_time(retry_time)
        assert "tomorrow at" in formatted
        
        # Other day
        retry_time = now + timedelta(days=3)
        formatted = parser.format_retry_time(retry_time)
        assert "at" in formatted
        assert "today" not in formatted
        assert "tomorrow" not in formatted
    
    def test_timezone_mappings(self, parser):
        """Test timezone abbreviation mappings."""
        assert parser.timezone_mappings["PST"] == "America/Los_Angeles"
        assert parser.timezone_mappings["EST"] == "America/New_York"
        assert parser.timezone_mappings["UTC"] == "UTC"
        assert parser.timezone_mappings["GMT"] == "UTC"
    
    def test_parse_invalid_messages(self, parser):
        """Test parsing messages without time information."""
        result = parser.parse_usage_limit_message("No time info here")
        assert result is None
        
        result = parser.parse_usage_limit_message("")
        assert result is None


@pytest.mark.unit
class TestRunCommand:
    """Test the run command utility."""
    
    def test_run_string_command(self):
        """Test running a string command."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            
            result = run("echo test")
            
            mock_run.assert_called_once()
            assert mock_run.call_args[0][0] == "echo test"
            assert mock_run.call_args[1]["shell"] is True
    
    def test_run_list_command(self):
        """Test running a list command."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            
            result = run(["echo", "test"])
            
            mock_run.assert_called_once()
            assert mock_run.call_args[0][0] == ["echo", "test"]
            assert mock_run.call_args[1]["shell"] is False
    
    def test_run_with_capture_output(self):
        """Test capturing command output."""
        with patch("subprocess.run") as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "output text"
            mock_result.stderr = ""
            mock_run.return_value = mock_result
            
            result = run("echo test", capture_output=True)
            
            assert result is not None
            assert result.stdout == "output text"
    
    def test_run_with_error(self):
        """Test handling command errors."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
            
            # With check=True (default), should raise
            with pytest.raises(subprocess.CalledProcessError):
                run("failing command")
            
            # With check=False, should return None
            result = run("failing command", check=False)
            assert result is None
    
    def test_run_with_cwd_and_env(self):
        """Test running with custom cwd and environment."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            
            cwd = Path("/custom/dir")
            env = {"KEY": "value"}
            
            run("command", cwd=cwd, env=env)
            
            assert mock_run.call_args[1]["cwd"] == cwd
            assert mock_run.call_args[1]["env"] == env


@pytest.mark.unit
class TestProjectChecks:
    """Test project detection functions."""
    
    def test_check_flutter_project(self, mock_carenji_project):
        """Test Flutter project detection."""
        # Valid Flutter project
        assert check_flutter_project(mock_carenji_project) is True
        
        # No pubspec.yaml
        empty_dir = mock_carenji_project.parent / "empty"
        empty_dir.mkdir()
        assert check_flutter_project(empty_dir) is False
        
        # pubspec.yaml without Flutter
        non_flutter = mock_carenji_project.parent / "non_flutter"
        non_flutter.mkdir()
        (non_flutter / "pubspec.yaml").write_text("name: test\n")
        assert check_flutter_project(non_flutter) is False
    
    def test_check_firebase_project(self, mock_carenji_project):
        """Test Firebase project detection."""
        # Valid Firebase project
        assert check_firebase_project(mock_carenji_project) is True
        
        # No firebase.json
        empty_dir = mock_carenji_project.parent / "empty"
        empty_dir.mkdir()
        assert check_firebase_project(empty_dir) is False
    
    def test_check_carenji_project(self, mock_carenji_project):
        """Test Carenji project detection."""
        # Valid Carenji project
        assert check_carenji_project(mock_carenji_project) is True
        
        # Flutter project but not Carenji
        other_flutter = mock_carenji_project.parent / "other_flutter"
        other_flutter.mkdir()
        (other_flutter / "pubspec.yaml").write_text(
            "name: other_app\ndependencies:\n  flutter:\n    sdk: flutter\n"
        )
        assert check_carenji_project(other_flutter) is False
        
        # Not even a Flutter project
        empty_dir = mock_carenji_project.parent / "empty"
        empty_dir.mkdir()
        assert check_carenji_project(empty_dir) is False


@pytest.mark.unit
class TestFirebaseEmulators:
    """Test Firebase emulator functions."""
    
    def test_get_firebase_emulator_status(self):
        """Test checking Firebase emulator status."""
        with patch("claude_code_agent_farm.utils.run") as mock_run:
            # Simulate successful curl to auth emulator
            mock_run.return_value = Mock(returncode=0)
            
            status = get_firebase_emulator_status()
            
            assert isinstance(status, dict)
            assert "auth" in status
            assert "firestore" in status
            assert "functions" in status
            assert "ui" in status
            
            # Check that curl was called for each service
            assert mock_run.call_count == 4
    
    def test_start_firebase_emulators_docker(self, mock_carenji_project):
        """Test starting Firebase emulators via Docker."""
        docker_compose = mock_carenji_project / "docker-compose.emulators.yml"
        docker_compose.write_text("version: '3.8'\n")
        
        with patch("claude_code_agent_farm.utils.run") as mock_run:
            mock_run.return_value = Mock()
            
            result = start_firebase_emulators(mock_carenji_project)
            
            assert result is True
            mock_run.assert_called_once()
            assert "docker-compose" in mock_run.call_args[0][0]
    
    def test_start_firebase_emulators_cli(self, mock_carenji_project):
        """Test starting Firebase emulators via CLI."""
        with patch("claude_code_agent_farm.utils.run") as mock_run:
            mock_run.return_value = Mock()
            
            result = start_firebase_emulators(mock_carenji_project)
            
            assert result is True
            mock_run.assert_called_once()
            assert "firebase emulators:start" in mock_run.call_args[0][0]


@pytest.mark.unit
class TestHelperFunctions:
    """Test other helper functions."""
    
    def test_get_flutter_mcp_command(self, mock_carenji_project):
        """Test generating Flutter run command with MCP flags."""
        command = get_flutter_mcp_command(mock_carenji_project)
        
        assert f"cd {mock_carenji_project}" in command
        assert "flutter run" in command
        assert "--debug" in command
        assert "--host-vmservice-port=8182" in command
        assert "--dds-port=8181" in command
        assert "--enable-vm-service" in command
        assert "--disable-service-auth-codes" in command
    
    def test_format_duration(self):
        """Test duration formatting."""
        # Seconds
        assert format_duration(45) == "45s"
        
        # Minutes and seconds
        assert format_duration(125) == "2m 5s"
        
        # Hours and minutes
        assert format_duration(3665) == "1h 1m"
        
        # Edge cases
        assert format_duration(0) == "0s"
        assert format_duration(59) == "59s"
        assert format_duration(60) == "1m 0s"
        assert format_duration(3600) == "1h 0m"
    
    def test_get_carenji_prompt_template(self):
        """Test getting Carenji prompt templates."""
        # Known templates
        prompt = get_carenji_prompt_template("fix_errors")
        assert "Flutter analyzer errors" in prompt
        assert "carenji" in prompt
        
        prompt = get_carenji_prompt_template("implement_feature")
        assert "clean architecture" in prompt
        
        prompt = get_carenji_prompt_template("write_tests")
        assert "comprehensive tests" in prompt
        
        prompt = get_carenji_prompt_template("review_pr")
        assert "adherence to carenji" in prompt
        
        # Unknown template - returns default
        prompt = get_carenji_prompt_template("unknown_task")
        assert "carenji Flutter app" in prompt
        assert "CLAUDE.md" in prompt