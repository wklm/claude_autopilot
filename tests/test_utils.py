"""Tests for utility functions."""

import tempfile
from pathlib import Path

import pytest

from claude_code_agent_farm.utils import line_count, run


def test_run_command_success():
    """Test successful command execution."""
    returncode, stdout, stderr = run("echo 'Hello World'", capture=True)
    assert returncode == 0
    assert "Hello World" in stdout
    assert stderr == ""


def test_run_command_failure():
    """Test failed command execution."""
    with pytest.raises(Exception):
        run("false", check=True)


def test_run_command_no_check():
    """Test command execution without check."""
    returncode, stdout, stderr = run("false", check=False, capture=True)
    assert returncode != 0


def test_line_count_utf8():
    """Test line counting with UTF-8 file."""
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        f.write("Line 1\nLine 2\nLine 3\n")
        temp_path = Path(f.name)
    
    try:
        count = line_count(temp_path)
        assert count == 3
    finally:
        temp_path.unlink()


def test_line_count_empty_file():
    """Test line counting with empty file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        count = line_count(temp_path)
        assert count == 0
    finally:
        temp_path.unlink()


def test_line_count_nonexistent_file():
    """Test line counting with non-existent file."""
    count = line_count(Path("/tmp/nonexistent_file_12345.txt"))
    assert count == 0