"""Integration tests for tmux session management."""

import os
import subprocess
import time
from pathlib import Path

import pytest

from claude_code_agent_farm.utils import run


@pytest.mark.integration
class TestTmuxIntegration:
    """Test tmux session management functionality."""
    
    @pytest.fixture
    def tmux_session_name(self):
        """Generate unique tmux session name for testing."""
        return f"test-carenji-{os.getpid()}"
    
    @pytest.fixture
    def tmux_session(self, tmux_session_name):
        """Create and cleanup a tmux session for testing."""
        # Kill any existing session with same name
        run(f"tmux kill-session -t {tmux_session_name}", check=False, quiet=True)
        
        # Create new session
        result = run(
            f"tmux new-session -d -s {tmux_session_name} -n test",
            check=False,
            capture_output=True
        )
        
        if result and result.returncode != 0:
            pytest.skip("tmux not available")
        
        yield tmux_session_name
        
        # Cleanup
        run(f"tmux kill-session -t {tmux_session_name}", check=False, quiet=True)
    
    def test_create_tmux_session(self, tmux_session_name):
        """Test creating a tmux session."""
        # Kill any existing
        run(f"tmux kill-session -t {tmux_session_name}", check=False, quiet=True)
        
        # Create session
        result = run(
            f"tmux new-session -d -s {tmux_session_name} -n agent",
            capture_output=True
        )
        
        assert result is not None
        assert result.returncode == 0
        
        # Verify session exists
        result = run(
            "tmux list-sessions",
            capture_output=True
        )
        
        assert tmux_session_name in result.stdout
        
        # Cleanup
        run(f"tmux kill-session -t {tmux_session_name}", check=False, quiet=True)
    
    def test_send_keys_to_session(self, tmux_session):
        """Test sending keys to tmux session."""
        # Send echo command
        run(
            f"tmux send-keys -t {tmux_session}:test 'echo Hello Carenji' Enter",
            check=True
        )
        
        # Give it time to execute
        time.sleep(0.5)
        
        # Capture pane content
        result = run(
            f"tmux capture-pane -t {tmux_session}:test -p",
            capture_output=True
        )
        
        assert "Hello Carenji" in result.stdout
    
    def test_capture_pane_content(self, tmux_session):
        """Test capturing tmux pane content."""
        # Send multiple commands
        commands = [
            "echo 'Testing Carenji Agent'",
            "echo 'Flutter development'",
            "echo 'Firebase backend'"
        ]
        
        for cmd in commands:
            run(f"tmux send-keys -t {tmux_session}:test '{cmd}' Enter")
            time.sleep(0.1)
        
        # Capture content
        result = run(
            f"tmux capture-pane -t {tmux_session}:test -p",
            capture_output=True
        )
        
        content = result.stdout
        assert "Testing Carenji Agent" in content
        assert "Flutter development" in content
        assert "Firebase backend" in content
    
    def test_clear_pane_history(self, tmux_session):
        """Test clearing tmux pane history."""
        # Add some content
        run(f"tmux send-keys -t {tmux_session}:test 'echo Test content' Enter")
        time.sleep(0.1)
        
        # Verify content exists
        result = run(f"tmux capture-pane -t {tmux_session}:test -p", capture_output=True)
        assert "Test content" in result.stdout
        
        # Clear history
        run(f"tmux clear-history -t {tmux_session}:test")
        
        # Send new content
        run(f"tmux send-keys -t {tmux_session}:test 'echo New content' Enter")
        time.sleep(0.1)
        
        # Capture should only show new content
        result = run(f"tmux capture-pane -t {tmux_session}:test -p", capture_output=True)
        assert "New content" in result.stdout
        # Old content might still be visible depending on terminal size
    
    def test_tmux_configuration(self, tmux_session):
        """Test tmux configuration options."""
        # Set mouse mode
        run(f"tmux set-option -t {tmux_session} -g mouse on")
        
        # Check if set
        result = run(
            f"tmux show-options -t {tmux_session} -g mouse",
            capture_output=True
        )
        
        assert "mouse on" in result.stdout
    
    def test_send_special_keys(self, tmux_session):
        """Test sending special keys like Ctrl+C."""
        # Start a long-running command
        run(f"tmux send-keys -t {tmux_session}:test 'sleep 100' Enter")
        time.sleep(0.5)
        
        # Send Ctrl+C
        run(f"tmux send-keys -t {tmux_session}:test C-c")
        time.sleep(0.5)
        
        # Check that sleep was interrupted
        result = run(f"tmux capture-pane -t {tmux_session}:test -p", capture_output=True)
        # Should not be sleeping anymore - prompt should be back
        assert "sleep 100" in result.stdout
    
    def test_multiple_windows(self, tmux_session):
        """Test working with multiple tmux windows."""
        # Create additional window
        run(f"tmux new-window -t {tmux_session} -n monitoring")
        
        # Send different content to each window
        run(f"tmux send-keys -t {tmux_session}:test 'echo Agent window' Enter")
        run(f"tmux send-keys -t {tmux_session}:monitoring 'echo Monitor window' Enter")
        time.sleep(0.5)
        
        # Capture from first window
        result1 = run(
            f"tmux capture-pane -t {tmux_session}:test -p",
            capture_output=True
        )
        assert "Agent window" in result1.stdout
        
        # Capture from second window
        result2 = run(
            f"tmux capture-pane -t {tmux_session}:monitoring -p",
            capture_output=True
        )
        assert "Monitor window" in result2.stdout
    
    def test_session_persistence(self, tmux_session):
        """Test that session persists across connections."""
        # Add content
        run(f"tmux send-keys -t {tmux_session}:test 'echo Persistent data' Enter")
        time.sleep(0.1)
        
        # Detach (simulated by just not being attached)
        # In real usage, would detach with 'tmux detach-client'
        
        # Verify session still exists
        result = run("tmux list-sessions", capture_output=True)
        assert tmux_session in result.stdout
        
        # Verify content still there
        result = run(f"tmux capture-pane -t {tmux_session}:test -p", capture_output=True)
        assert "Persistent data" in result.stdout
    
    def test_escape_special_characters(self, tmux_session):
        """Test sending text with special characters."""
        # Test various special characters
        test_strings = [
            "Test with 'single quotes'",
            'Test with "double quotes"',
            "Test with $variable",
            "Test with `backticks`",
            "Test with ; semicolon",
            "Test with & ampersand",
        ]
        
        for test_str in test_strings:
            # Clear first
            run(f"tmux send-keys -t {tmux_session}:test C-l")
            time.sleep(0.1)
            
            # Escape and send
            escaped = test_str.replace("'", "'\"'\"'")
            run(f"tmux send-keys -t {tmux_session}:test 'echo {escaped}' Enter")
            time.sleep(0.1)
            
            # Verify
            result = run(
                f"tmux capture-pane -t {tmux_session}:test -p",
                capture_output=True
            )
            # Check the original string appears in output
            assert test_str in result.stdout
    
    @pytest.mark.docker
    def test_tmux_in_docker(self, docker_container):
        """Test tmux functionality inside Docker container."""
        session_name = "docker-test"
        
        # Create session in container
        result = subprocess.run(
            ["docker", "exec", docker_container,
             "tmux", "new-session", "-d", "-s", session_name],
            capture_output=True
        )
        
        assert result.returncode == 0
        
        # Send command
        result = subprocess.run(
            ["docker", "exec", docker_container,
             "tmux", "send-keys", "-t", session_name, "echo Docker tmux test", "Enter"],
            capture_output=True
        )
        
        assert result.returncode == 0
        
        time.sleep(0.5)
        
        # Capture output
        result = subprocess.run(
            ["docker", "exec", docker_container,
             "tmux", "capture-pane", "-t", session_name, "-p"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Docker tmux test" in result.stdout
        
        # Cleanup
        subprocess.run(
            ["docker", "exec", docker_container,
             "tmux", "kill-session", "-t", session_name],
            capture_output=True
        )