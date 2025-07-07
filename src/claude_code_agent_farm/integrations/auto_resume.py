"""Integration with claude-auto-resume for automatic API quota handling."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from claude_code_agent_farm.utils import run


class ClaudeAutoResume:
    """Wrapper for claude-auto-resume functionality."""
    
    def __init__(self):
        self.script_path = self._find_or_install_script()
        
    def _find_or_install_script(self) -> Path:
        """Find claude-auto-resume script or install it if not found.
        
        Returns:
            Path to claude-auto-resume script
        """
        # Check if claude-auto-resume is in PATH
        claude_resume = shutil.which("claude-auto-resume")
        if claude_resume:
            return Path(claude_resume)
            
        # Check common installation locations
        common_paths = [
            Path.home() / ".local/bin/claude-auto-resume",
            Path("/usr/local/bin/claude-auto-resume"),
            Path("/opt/claude-auto-resume/claude-auto-resume.sh"),
        ]
        
        for path in common_paths:
            if path.exists():
                return path
                
        # If not found, provide installation instructions
        raise RuntimeError(
            "claude-auto-resume not found. Please install it from: "
            "https://github.com/terryso/claude-auto-resume"
        )
        
    def create_wrapper_script(self, output_path: Path) -> None:
        """Create a wrapper script that uses claude-auto-resume.
        
        Args:
            output_path: Path where to create the wrapper script
        """
        wrapper_content = f"""#!/bin/bash
# Auto-generated wrapper script for Claude with auto-resume capability

# Set up environment variables
export ENABLE_BACKGROUND_TASKS=1
export CLAUDE_DANGEROUSLY_SKIP_PERMISSIONS=1

# Add any additional environment setup
export FLUTTER_HOME=/opt/flutter
export ANDROID_HOME=/opt/android-sdk
export PATH="${{FLUTTER_HOME}}/bin:${{ANDROID_HOME}}/cmdline-tools/latest/bin:${{ANDROID_HOME}}/platform-tools:${{PATH}}"

# Set up Flutter cache directories
export PUB_CACHE="$HOME/.pub-cache"
export FLUTTER_STORAGE_BASE_URL="https://storage.googleapis.com"
export FLUTTER_CACHE_DIR="$HOME/.flutter/cache"

# Create cache directories if they don't exist
mkdir -p "$PUB_CACHE" 2>/dev/null || true
mkdir -p "$FLUTTER_CACHE_DIR" 2>/dev/null || true

# Execute claude through claude-auto-resume
# This will automatically handle API quota limits
exec {self.script_path} "$@"
"""
        
        output_path.write_text(wrapper_content)
        output_path.chmod(0o755)
        
    def wrap_command(self, command: str) -> str:
        """Wrap a claude command with auto-resume capability.
        
        Args:
            command: Original claude command
            
        Returns:
            Wrapped command that uses claude-auto-resume
        """
        # If command already uses claude-auto-resume, return as-is
        if "claude-auto-resume" in command:
            return command
            
        # Replace 'claude' or 'cc' with claude-auto-resume
        if command.startswith("claude "):
            return command.replace("claude ", f"{self.script_path} ", 1)
        elif command.startswith("cc "):
            return command.replace("cc ", f"{self.script_path} ", 1)
        elif command == "claude" or command == "cc":
            return str(self.script_path)
        else:
            # Assume the whole command should be passed to claude-auto-resume
            return f"{self.script_path} {command}"
            
    def test_installation(self) -> bool:
        """Test if claude-auto-resume is properly installed and working.
        
        Returns:
            True if installation is working
        """
        try:
            # Try to run claude-auto-resume with help flag
            result = subprocess.run(
                [str(self.script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0 or "usage:" in result.stdout.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False