"""Utility functions for Claude Flutter Firebase Agent.

This module provides helper functions and utilities used throughout the application,
including time parsing, command execution, and Flutter/Firebase specific helpers.
"""

import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union, List
import pytz
from rich.console import Console

console = Console(stderr=True)


class UsageLimitTimeParser:
    """Parse usage limit messages from Claude to extract retry times."""
    
    def __init__(self):
        self.timezone_mappings = {
            "PST": "America/Los_Angeles",
            "PDT": "America/Los_Angeles",
            "EST": "America/New_York",
            "EDT": "America/New_York",
            "CST": "America/Chicago",
            "CDT": "America/Chicago",
            "MST": "America/Denver",
            "MDT": "America/Denver",
            "UTC": "UTC",
            "GMT": "UTC",
        }
        
        # Patterns for parsing retry times
        self.time_patterns = [
            # "try again at 3:45 PM PST"
            r"try again at (\d{1,2}):(\d{2})\s*(AM|PM)\s*([A-Z]{2,3})",
            # "retry after 3:45 PM PST"
            r"retry after (\d{1,2}):(\d{2})\s*(AM|PM)\s*([A-Z]{2,3})",
            # "available at 3:45 PM PST"
            r"available at (\d{1,2}):(\d{2})\s*(AM|PM)\s*([A-Z]{2,3})",
            # "wait until 3:45 PM PST"
            r"wait until (\d{1,2}):(\d{2})\s*(AM|PM)\s*([A-Z]{2,3})",
        ]
        
        # Patterns for duration-based waiting
        self.duration_patterns = [
            # "wait 2 hours"
            r"wait (\d+)\s*hours?",
            # "try again in 30 minutes"
            r"try again in (\d+)\s*minutes?",
            # "retry in 2 hours and 30 minutes"
            r"retry in (\d+)\s*hours?\s*(?:and\s*)?(\d+)\s*minutes?",
        ]
    
    def parse_usage_limit_message(self, message: str) -> Optional[datetime]:
        """Parse a usage limit message to extract the retry time."""
        message_lower = message.lower()
        
        # Try time patterns first
        for pattern in self.time_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2))
                am_pm = match.group(3).upper()
                tz_abbr = match.group(4).upper()
                
                # Convert to 24-hour format
                if am_pm == "PM" and hour != 12:
                    hour += 12
                elif am_pm == "AM" and hour == 12:
                    hour = 0
                
                # Get timezone
                tz_name = self.timezone_mappings.get(tz_abbr, "UTC")
                tz = pytz.timezone(tz_name)
                
                # Create datetime in the specified timezone
                now = datetime.now(tz)
                retry_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                # If the time is in the past, assume it's tomorrow
                if retry_time <= now:
                    retry_time += timedelta(days=1)
                
                # Convert to local time
                return retry_time.astimezone()
        
        # If no specific time found, try duration patterns
        wait_duration = self.get_wait_duration(message)
        if wait_duration:
            return datetime.now() + wait_duration
        
        return None
    
    def get_wait_duration(self, message: str) -> Optional[timedelta]:
        """Extract wait duration from message."""
        message_lower = message.lower()
        
        # Try duration patterns
        for pattern in self.duration_patterns:
            match = re.search(pattern, message_lower)
            if match:
                if len(match.groups()) == 1:
                    # Just hours or minutes
                    value = int(match.group(1))
                    if "hour" in pattern:
                        return timedelta(hours=value)
                    else:
                        return timedelta(minutes=value)
                else:
                    # Hours and minutes
                    hours = int(match.group(1))
                    minutes = int(match.group(2)) if match.group(2) else 0
                    return timedelta(hours=hours, minutes=minutes)
        
        return None
    
    def format_retry_time(self, retry_time: datetime) -> str:
        """Format retry time for display."""
        now = datetime.now()
        if retry_time.date() == now.date():
            # Today
            return f"today at {retry_time.strftime('%I:%M %p')}"
        elif retry_time.date() == (now + timedelta(days=1)).date():
            # Tomorrow
            return f"tomorrow at {retry_time.strftime('%I:%M %p')}"
        else:
            # Other day
            return retry_time.strftime('%B %d at %I:%M %p')


def run(
    cmd: Union[str, List[str]], 
    check: bool = True, 
    quiet: bool = False,
    capture_output: bool = False,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
) -> Optional[subprocess.CompletedProcess]:
    """Run a shell command with error handling.
    
    Args:
        cmd: Command to run (string or list)
        check: Whether to raise on non-zero exit
        quiet: Whether to suppress output
        capture_output: Whether to capture output
        cwd: Working directory
        env: Environment variables
        
    Returns:
        CompletedProcess if capture_output, else None
    """
    try:
        # Convert to list if string
        if isinstance(cmd, str):
            shell = True
            cmd_list = cmd
        else:
            shell = False
            cmd_list = cmd
        
        # Prepare kwargs
        kwargs = {
            "shell": shell,
            "check": check,
            "cwd": cwd,
            "env": env,
        }
        
        if capture_output or quiet:
            kwargs["capture_output"] = True
            kwargs["text"] = True
        
        # Run command
        result = subprocess.run(cmd_list, **kwargs)
        
        # Show output if not quiet and not capturing
        if not quiet and not capture_output and hasattr(result, 'stdout') and result.stdout:
            console.print(result.stdout.strip())
        
        return result if capture_output else None
        
    except subprocess.CalledProcessError as e:
        if not quiet:
            console.print(f"[red]Command failed: {e.cmd}[/red]")
            if hasattr(e, 'stderr') and e.stderr:
                console.print(f"[red]Error: {e.stderr}[/red]")
        if check:
            raise
        return None
    except Exception as e:
        if not quiet:
            console.print(f"[red]Error running command: {e}[/red]")
        if check:
            raise
        return None


def check_flutter_project(path: Path) -> bool:
    """Check if the given path is a Flutter project."""
    pubspec = path / "pubspec.yaml"
    if not pubspec.exists():
        return False
    
    # Check for Flutter in pubspec
    try:
        content = pubspec.read_text()
        return "flutter:" in content and "sdk: flutter" in content
    except:
        return False


def check_firebase_project(path: Path) -> bool:
    """Check if the given path has Firebase configuration."""
    firebase_json = path / "firebase.json"
    return firebase_json.exists()


def check_carenji_project(path: Path) -> bool:
    """Check if the given path is the carenji project."""
    if not check_flutter_project(path):
        return False
    
    # Check for carenji-specific files
    pubspec = path / "pubspec.yaml"
    try:
        content = pubspec.read_text()
        return "name: carenji" in content
    except:
        return False


def get_firebase_emulator_status() -> dict:
    """Check status of Firebase emulators."""
    from claude_code_agent_farm.constants import CARENJI_FIREBASE_EMULATOR_PORTS
    
    status = {}
    for service, port in CARENJI_FIREBASE_EMULATOR_PORTS.items():
        try:
            result = run(f"curl -s http://localhost:{port}", check=False, capture_output=True)
            status[service] = result is not None and result.returncode == 0
        except:
            status[service] = False
    
    return status


def start_firebase_emulators(project_path: Path) -> bool:
    """Start Firebase emulators for the project."""
    # Check if docker-compose.emulators.yml exists
    docker_compose = project_path / "docker-compose.emulators.yml"
    if docker_compose.exists():
        console.print("[cyan]Starting Firebase emulators via Docker...[/cyan]")
        result = run(
            f"docker-compose -f {docker_compose} up -d",
            cwd=project_path,
            check=False
        )
        return result is not None
    else:
        # Try starting with Firebase CLI
        console.print("[cyan]Starting Firebase emulators via Firebase CLI...[/cyan]")
        result = run(
            "firebase emulators:start --only auth,firestore,functions,storage",
            cwd=project_path,
            check=False
        )
        return result is not None


def get_flutter_mcp_command(project_path: Path) -> str:
    """Get the Flutter run command with MCP flags."""
    from claude_code_agent_farm.constants import FLUTTER_RUN_FLAGS
    
    return f"cd {project_path} && flutter run {' '.join(FLUTTER_RUN_FLAGS)}"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def get_carenji_prompt_template(task_type: str) -> str:
    """Get a prompt template for specific carenji development tasks."""
    from claude_code_agent_farm.constants import CARENJI_PROMPT_TEMPLATES
    
    return CARENJI_PROMPT_TEMPLATES.get(
        task_type, 
        "Help with carenji Flutter app development following the guidelines in CLAUDE.md"
    )