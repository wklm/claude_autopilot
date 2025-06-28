#!/usr/bin/env python3
"""
Claude Code Agent Farm - Hybrid Orchestrator
Combines simplicity with robust monitoring and automatic agent management
"""

import contextlib
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime
from pathlib import Path
from random import randint
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import typer
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table
except ImportError:
    print("Please install required libraries: pip install typer rich")
    sys.exit(1)

app = typer.Typer(help="Claude Code Agent Farm - Parallel code fixing automation", rich_markup_mode="rich")
console = Console()

# ─────────────────────────────── Configuration ────────────────────────────── #


def interruptible_confirm(message: str, default: bool = False) -> bool:
    """A confirm prompt that can be interrupted by Ctrl+C"""
    try:
        return typer.confirm(message, default=default)
    except (KeyboardInterrupt, EOFError) as e:
        console.print("\n[yellow]Confirmation interrupted[/yellow]")
        raise KeyboardInterrupt() from e


DEFAULT_PROMPT = textwrap.dedent("""\
    I need you to start going through combined_typechecker_and_linter_problems.txt (just pick random chunks of 50 lines at a time from anywhere within the file, starting with a random starting line; since I have multiple agents working on this task, I want each agent to work on different problems!)

    As you select your chosen problems, mark them as such by prepending the line with [COMPLETED] so we can keep track of which ones have already been processed-- do this up front so there's no risk of forgetting to do it later and wasting time and compute on errors that are already being worked on or which were previously worked on. (Obviously, when selecting your random lines to work on, you should first filter out any rows that have "[COMPLETED]" in them so you don't accidentally work on already in-progress or completed tasks!)

    I want you to be SMART about fixing the problems. For example, if it's a type related problem, never try to use a stupid "band aid" fix and set the type to be Unknown or something dumb like that. If there's an unused variable or import, instead of just deleting it, figure out what we originally intended and whether that import or variable could be usefully and productively employed in the code to improve it so that it's no longer unused or unreferenced.

    Make all edits to the existing code files-- don't ever create a duplicative code file with the changes and give it some silly name; for instance, don't correct a problem in ComponentXYZ.tsx in a newly created file called ComponentXYZFixed.tsx or ComponentXYZNew.tsx-- always just revise ComponentXYZ.tsx in place!

    CRITICALLY IMPORTANT: You must adhere to ALL guidelines and advice in the NEXTJS15_BEST_PRACTICES.md document. I want to avoid technical debt and endless compatibility shims and workarounds and just fix things once and for all the RIGHT WAY. This code is still in development so we don't care at all about backwards compatibility. Note that we only use bun in this project, never npm. And you MUST check each proposed change against the @NEXT_BEST_PRACTICES_IMPLEMENTATION_PROGRESS.md guide!

    When you're done fixing the entire batch of selected problems, you can commit your progress to git with a detailed commit message (but don't go overboard making the commit message super long). Try to complete as much work as possible before coming back to me for more instructions-- what I've already asked you to do should keep you very busy for a while!
""")

# ─────────────────────────────── Constants ────────────────────────────────── #

MONITOR_STATE_FILE = ".claude_agent_farm_state.json"

# ─────────────────────────────── Helper Functions ─────────────────────────── #


def run(cmd: str, *, check: bool = True, quiet: bool = False, capture: bool = False) -> Tuple[int, str, str]:
    """Execute shell command with optional output capture

    When capture=False, output is streamed to terminal unless quiet=True
    When capture=True, output is captured and returned
    """
    if not quiet:
        console.log(cmd, style="cyan")

    # Parse command for shell safety when possible
    cmd_arg: Union[str, List[str]]
    try:
        # Try to parse as a list of arguments for safer execution
        cmd_list = shlex.split(cmd)
        use_shell = False
        cmd_arg = cmd_list
    except ValueError:
        # Fall back to shell=True for complex commands with pipes, redirects, etc.
        cmd_list = []  # Not used when shell=True
        use_shell = True
        cmd_arg = cmd

    if capture:
        result = subprocess.run(cmd_arg, shell=use_shell, capture_output=True, text=True, check=check)
        return result.returncode, result.stdout or "", result.stderr or ""
    else:
        # Stream output to terminal when not capturing
        # Preserve stderr even in quiet-mode so that exceptions contain detail
        if quiet:
            result = subprocess.run(cmd_arg, shell=use_shell, capture_output=True, text=True, check=check)
            return result.returncode, result.stdout or "", result.stderr or ""
        stdout_pipe = None
        stderr_pipe = subprocess.STDOUT
        try:
            result = subprocess.run(
                cmd_arg, shell=use_shell, check=check, stdout=stdout_pipe, stderr=stderr_pipe, text=True
            )
            return result.returncode, "", ""
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Command failed with exit code {e.returncode}: {cmd}[/red]")
            raise


def line_count(file_path: Path) -> int:
    """Count lines in a file"""
    try:
        # Try UTF-8 first, then fall back to latin-1
        try:
            with file_path.open("r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except UnicodeDecodeError:
            # Try common encodings
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    with file_path.open("r", encoding=encoding) as f:
                        return sum(1 for _ in f)
                except UnicodeDecodeError:
                    continue
            # Last resort: ignore errors
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not count lines in {file_path}: {e}[/yellow]")
        return 0


def tmux_send(target: str, data: str, enter: bool = True) -> None:
    """Send keystrokes to a tmux pane (binary-safe)"""
    max_retries = 3
    base_delay = 0.5

    for attempt in range(max_retries):
        try:
            # Use tmux's literal mode (-l) to avoid quoting issues
            if data:
                run(f"tmux send-keys -l -t {target} {shlex.quote(data)}", quiet=True)
                # CRITICAL: Small delay between pasting and Enter for Claude Code
                if enter:
                    time.sleep(0.2)
            if enter:
                run(f"tmux send-keys -t {target} C-m", quiet=True)
            break
        except subprocess.CalledProcessError:
            if attempt < max_retries - 1:
                # Exponential backoff: 0.5s, 1s, 2s
                time.sleep(base_delay * (2**attempt))
            else:
                raise


def tmux_capture(target: str) -> str:
    """Capture content from a tmux pane"""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            _, stdout, _ = run(f"tmux capture-pane -t {target} -p", quiet=True, capture=True)
            return stdout
        except subprocess.CalledProcessError:
            if attempt < max_retries - 1:
                time.sleep(0.2)
            else:
                # Return empty string on persistent failure
                return ""
    return ""


# ─────────────────────────────── Agent Monitor ────────────────────────────── #


class AgentMonitor:
    """Monitors Claude Code agents for health and performance"""

    def __init__(
        self,
        session: str,
        num_agents: int,
        pane_mapping: Dict[int, str],
        context_threshold: int = 20,
        idle_timeout: int = 60,
        max_errors: int = 3,
    ):
        self.session = session
        self.num_agents = num_agents
        self.pane_mapping = pane_mapping
        self.agents: Dict[int, Dict] = {}
        self.running = True
        self.start_time = datetime.now()
        self.context_threshold = context_threshold
        self.idle_timeout = idle_timeout
        self.max_errors = max_errors

        # Initialize agent tracking
        for i in range(num_agents):
            self.agents[i] = {
                "status": "starting",
                "start_time": datetime.now(),
                "cycles": 0,
                "last_context": 100,
                "errors": 0,
                "last_activity": datetime.now(),
                "restart_count": 0,
                "last_restart": None,
            }

    def detect_context_percentage(self, content: str) -> Optional[int]:
        """Extract context percentage from pane content"""
        # Try multiple patterns for robustness
        patterns = [
            r"Context left until\s*auto-compact:\s*(\d+)%",
            r"Context remaining:\s*(\d+)%",
            r"(\d+)%\s*context\s*remaining",
            r"Context:\s*(\d+)%",
        ]

        # Safety check for empty content
        if not content:
            return None

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def is_claude_ready(self, content: str) -> bool:
        """Check if Claude Code is ready for input"""
        # Multiple possible indicators that Claude is ready
        ready_indicators = [
            "Welcome to Claude Code!" in content,  # Welcome message
            ("│ > Try" in content),  # The prompt box with suggestion
            ("? for shortcuts" in content),  # Shortcuts hint at bottom
            ("╰─" in content and "│ >" in content),  # Box structure with prompt
            ("/help for help" in content),  # Help text in welcome message
            ("cwd:" in content and "Welcome to Claude" in content),  # Working directory shown
            ("Bypassing Permissions" in content and "│ >" in content),  # May appear with prompt
            ("│ >" in content and "─╯" in content),  # Prompt box bottom border
        ]
        return any(ready_indicators)

    def is_claude_working(self, content: str) -> bool:
        """Check if Claude Code is actively working"""
        indicators = ["✻ Pontificating", "● Bash(", "✻ Running", "✻ Thinking", "esc to interrupt"]
        return any(indicator in content for indicator in indicators)

    def has_welcome_screen(self, content: str) -> bool:
        """Check if Claude Code is showing the welcome/setup screen"""
        welcome_indicators = [
            # Setup/onboarding screens only
            "Choose the text style",
            "Choose your language",
            "Let's get started",
            "run /theme",
            "Dark mode✔",
            "Light mode",
            "colorblind-friendly",
            # Remove "Welcome to Claude Code" as it appears when ready
        ]
        return any(indicator in content for indicator in welcome_indicators)

    def has_settings_error(self, content: str) -> bool:
        """Check for settings corruption"""
        # First check if Claude is actually ready (avoid false positives)
        if self.is_claude_ready(content):
            return False
            
        error_indicators = [
            # Login/auth prompts
            "Select login method:",
            "Claude account with subscription",
            "Sign in to Claude",
            "Log in to Claude",
            "Enter your API key",
            "API key",
            # Configuration errors
            "Configuration error",
            "Settings corrupted",
            "Invalid API key",
            "Authentication failed",
            "Rate limit exceeded",
            "Unauthorized",
            "Permission denied",
            "Failed to load configuration",
            "Invalid configuration",
            "Error loading settings",
            "Settings file is corrupted",
            "Failed to parse settings",
            "Invalid settings",
            "Corrupted settings",
            "Config corrupted",
            "configuration is corrupted",
            "Unable to load settings",
            "Error reading settings",
            "Settings error",
            "config error",
            # Parse errors
            "TypeError",
            "SyntaxError",
            "JSONDecodeError",
            "ParseError",
            # Other login-related text
            "Choose your login method",
            "Continue with Claude account",
            "I have a Claude account",
            "Create account",
        ]
        return any(indicator in content for indicator in error_indicators)

    def check_agent(self, agent_id: int) -> Dict:
        """Check status of a single agent"""
        pane_target = self.pane_mapping.get(agent_id)
        if not pane_target:
            console.print(f"[red]Error: No pane mapping found for agent {agent_id}[/red]")
            return self.agents[agent_id]

        content = tmux_capture(pane_target)

        agent = self.agents[agent_id]

        # Update context percentage
        context = self.detect_context_percentage(content)
        if context is not None:
            agent["last_context"] = context

        # Check for errors
        if self.has_settings_error(content):
            agent["status"] = "error"
            agent["errors"] += 1
            return agent

        # Update status based on activity
        if self.is_claude_working(content):
            agent["status"] = "working"
            agent["last_activity"] = datetime.now()
        elif self.is_claude_ready(content):
            # Check if idle for too long
            idle_time = (datetime.now() - agent["last_activity"]).total_seconds()
            if idle_time > self.idle_timeout:
                agent["status"] = "idle"
            else:
                agent["status"] = "ready"
        else:
            agent["status"] = "unknown"

        return agent

    def needs_restart(self, agent_id: int) -> bool:
        """Determine if an agent needs to be restarted"""
        agent = self.agents[agent_id]

        # Restart conditions
        return bool(
            agent["status"] == "error"
            or agent["errors"] >= self.max_errors
            or agent["status"] == "idle"
            or agent["last_context"] <= self.context_threshold
        )

    def get_status_table(self) -> Table:
        """Generate status table for all agents"""
        table = Table(title=f"Claude Agent Farm - {datetime.now().strftime('%H:%M:%S')}")

        table.add_column("Agent", style="cyan", width=8)
        table.add_column("Status", style="green", width=10)
        table.add_column("Cycles", style="yellow", width=6)
        table.add_column("Context", style="magenta", width=8)
        table.add_column("Runtime", style="blue", width=12)
        table.add_column("Errors", style="red", width=6)

        for agent_id in sorted(self.agents.keys()):
            agent = self.agents[agent_id]
            runtime = str(datetime.now() - agent["start_time"]).split(".")[0]

            status_style = {
                "working": "[green]",
                "ready": "[cyan]",
                "idle": "[yellow]",
                "error": "[red]",
                "starting": "[yellow]",
                "unknown": "[dim]",
            }.get(agent["status"], "")

            table.add_row(
                f"Pane {agent_id:02d}",
                f"{status_style}{agent['status']}[/]",
                str(agent["cycles"]),
                f"{agent['last_context']}%",
                runtime,
                str(agent["errors"]),
            )

        return table


# ─────────────────────────────── Main Orchestrator ────────────────────────── #


class ClaudeAgentFarm:
    def __init__(
        self,
        path: str,
        agents: int = 20,
        session: str = "claude_agents",
        stagger: float = 10.0,  # Increased from 4.0 to prevent settings clobbering
        wait_after_cc: float = 15.0,  # Increased from 8.0 to ensure Claude Code is fully ready
        check_interval: int = 10,
        skip_regenerate: bool = False,
        skip_commit: bool = False,
        auto_restart: bool = False,
        no_monitor: bool = False,
        attach: bool = False,
        prompt_file: Optional[str] = None,
        config: Optional[str] = None,
        context_threshold: int = 20,
        idle_timeout: int = 60,
        max_errors: int = 3,
        tmux_kill_on_exit: bool = True,
        tmux_mouse: bool = True,
        fast_start: bool = False,
        full_backup: bool = False,
    ):
        # Store all parameters
        self.path = path
        self.agents = agents
        self.session = session
        self.stagger = stagger
        self.wait_after_cc = wait_after_cc
        self.check_interval = check_interval
        self.skip_regenerate = skip_regenerate
        self.skip_commit = skip_commit
        self.auto_restart = auto_restart
        self.no_monitor = no_monitor
        self.attach = attach
        self.prompt_file = prompt_file
        self.config = config
        self.context_threshold = context_threshold
        self.idle_timeout = idle_timeout
        self.max_errors = max_errors
        self.tmux_kill_on_exit = tmux_kill_on_exit
        self.tmux_mouse = tmux_mouse
        self.fast_start = fast_start
        self.full_backup = full_backup

        # Initialize pane mapping
        self.pane_mapping: Dict[int, str] = {}

        # Validate session name (tmux has restrictions)
        if not re.match(r"^[a-zA-Z0-9_-]+$", self.session):
            raise ValueError(
                f"Invalid tmux session name '{self.session}'. Use only letters, numbers, hyphens, and underscores."
            )

        # Apply config file if provided
        if config:
            self._load_config(config)

        # Validate agent count
        if self.agents > getattr(self, "max_agents", 50):
            raise ValueError(f"Agent count {self.agents} exceeds maximum {getattr(self, 'max_agents', 50)}")

        # Initialize other attributes
        self.project_path = Path(self.path).expanduser().resolve()
        self.combined_file = self.project_path / "combined_typechecker_and_linter_problems.txt"
        self.prompt_text = self._load_prompt()
        self.monitor: Optional[AgentMonitor] = None
        self.running = True
        self.shutting_down = False

        # Git settings from config
        self.git_branch: Optional[str] = getattr(self, "git_branch", None)
        self.git_remote: str = getattr(self, "git_remote", "origin")
        self._cleanup_registered = False
        self.state_file = self.project_path / MONITOR_STATE_FILE

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self, config_path: str) -> None:
        """Load settings from JSON config file"""
        config_file = Path(config_path)
        if config_file.exists():
            with config_file.open() as f:
                config_data = json.load(f)
                # Accept all config values, not just existing attributes
                for key, value in config_data.items():
                    setattr(self, key, value)

    def _signal_handler(self, sig: Any, frame: Any) -> None:
        """Handle shutdown signals gracefully"""
        if not self.shutting_down:
            self.shutting_down = True
            console.print("\n[yellow]Received interrupt signal. Shutting down gracefully...[/yellow]")
            self.running = False

    def _backup_claude_settings(self) -> Optional[str]:
        """Backup essential Claude Code settings (excluding large caches)"""
        claude_dir = Path.home() / ".claude"
        if not claude_dir.exists():
            console.print("[yellow]No Claude Code directory found to backup[/yellow]")
            return None
        
        try:
            # Create backup directory in project
            backup_dir = self.project_path / ".claude_agent_farm_backups"
            backup_dir.mkdir(exist_ok=True)
            
            # Create timestamped backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_type = "full" if self.full_backup else "essential"
            backup_file = backup_dir / f"claude_backup_{backup_type}_{timestamp}.tar.gz"
            
            # Create compressed backup
            import tarfile
            
            if self.full_backup:
                console.print("[dim]Creating FULL backup of ~/.claude directory (this may take a while)...[/dim]")
                
                with tarfile.open(backup_file, "w:gz") as tar:
                    # Use filter to preserve all metadata
                    def reset_ids(tarinfo):
                        # Preserve all metadata but reset user/group to current user
                        # This prevents permission issues on restore
                        tarinfo.uid = os.getuid()
                        tarinfo.gid = os.getgid()
                        return tarinfo
                    
                    tar.add(claude_dir, arcname="claude", filter=reset_ids)
                
                size_mb = backup_file.stat().st_size / (1024 * 1024)
                console.print(f"[green]✓ Full backup completed: {backup_file.name} ({size_mb:.1f} MB)[/green]")
            else:
                console.print("[dim]Creating backup of essential Claude settings...[/dim]")
                
                with tarfile.open(backup_file, "w:gz") as tar:
                    # Filter to preserve metadata
                    def reset_ids(tarinfo):
                        tarinfo.uid = os.getuid()
                        tarinfo.gid = os.getgid()
                        return tarinfo
                    
                    # Add settings.json if it exists
                    settings_file = claude_dir / "settings.json"
                    if settings_file.exists():
                        tar.add(settings_file, arcname="claude/settings.json", filter=reset_ids)
                    
                    # Add ide directory (usually empty or small)
                    ide_dir = claude_dir / "ide"
                    if ide_dir.exists():
                        tar.add(ide_dir, arcname="claude/ide", filter=reset_ids)
                    
                    # Add statsig directory (small, contains feature flags)
                    statsig_dir = claude_dir / "statsig"
                    if statsig_dir.exists():
                        tar.add(statsig_dir, arcname="claude/statsig", filter=reset_ids)
                    
                    # Optionally add todos (usually small)
                    todos_dir = claude_dir / "todos"
                    if todos_dir.exists():
                        # Check size first
                        todos_size = sum(f.stat().st_size for f in todos_dir.rglob("*") if f.is_file())
                        if todos_size < 10 * 1024 * 1024:  # Less than 10MB
                            tar.add(todos_dir, arcname="claude/todos", filter=reset_ids)
                        else:
                            console.print(f"[dim]Skipping todos directory ({todos_size / 1024 / 1024:.1f} MB)[/dim]")
                    
                    # Skip projects directory - it's just caches
                    console.print("[dim]Skipping projects/ directory (caches)[/dim]")
                
                # Get backup size
                size_kb = backup_file.stat().st_size / 1024
                console.print(f"[green]✓ Backed up Claude settings to {backup_file.name} ({size_kb:.1f} KB)[/green]")
            
            # Clean up old backups (keep last 10)
            self._cleanup_old_backups(backup_dir, keep_count=10)
            
            return str(backup_file)
        except Exception as e:
            console.print(f"[red]Error: Could not backup Claude directory: {e}[/red]")
            return None

    def _cleanup_old_backups(self, backup_dir: Path, keep_count: int = 10) -> None:
        """Remove old backups, keeping only the most recent ones"""
        try:
            # Find all backup files (both essential and full)
            backups = sorted(backup_dir.glob("claude_backup_*.tar.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
            
            if len(backups) > keep_count:
                for old_backup in backups[keep_count:]:
                    old_backup.unlink()
                    console.print(f"[dim]Removed old backup: {old_backup.name}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not clean up old backups: {e}[/yellow]")

    def _restore_claude_settings(self, backup_path: Optional[str] = None) -> bool:
        """Restore Claude Code settings from backup"""
        try:
            # If no backup path provided, use the most recent one
            if backup_path is None:
                backup_path = self.settings_backup_path
            
            if not backup_path:
                console.print("[red]No backup path available[/red]")
                return False
            
            backup_file = Path(backup_path)
            if not backup_file.exists():
                console.print(f"[red]Backup file not found: {backup_path}[/red]")
                return False
            
            claude_dir = Path.home() / ".claude"
            
            # For partial backups, we don't need to remove the entire directory
            # Just extract over existing files
            try:
                # Ensure claude directory exists
                claude_dir.mkdir(exist_ok=True)
                
                # Save original metadata of existing files
                existing_metadata = {}
                for item in claude_dir.rglob("*"):
                    if item.exists():
                        stat = item.stat()
                        existing_metadata[str(item)] = {
                            'mode': stat.st_mode,
                            'mtime': stat.st_mtime,
                            'atime': stat.st_atime,
                        }
                
                # Extract backup
                import tarfile
                console.print(f"[dim]Restoring from {backup_file.name}...[/dim]")
                
                with tarfile.open(backup_file, "r:gz") as tar:
                    # Extract with numeric owner to preserve permissions
                    tar.extractall(path=claude_dir.parent, numeric_owner=True)
                    
                    # Get list of extracted files to preserve their times
                    for member in tar.getmembers():
                        if member.isfile():
                            extracted_path = claude_dir.parent / member.name
                            if extracted_path.exists():
                                # Preserve the modification time from the archive
                                os.utime(extracted_path, (member.mtime, member.mtime))
                
                # Ensure proper permissions on sensitive files
                settings_file = claude_dir / "settings.json"
                if settings_file.exists():
                    # Ensure settings.json has appropriate permissions (readable by user only)
                    os.chmod(settings_file, 0o600)
                
                # Set ownership to current user for all restored files
                uid = os.getuid()
                gid = os.getgid()
                
                for root, dirs, files in os.walk(claude_dir):
                    for d in dirs:
                        path = Path(root) / d
                        with contextlib.suppress(Exception):
                            os.chown(path, uid, gid)
                    for f in files:
                        path = Path(root) / f
                        with contextlib.suppress(Exception):
                            os.chown(path, uid, gid)
                
                console.print("[green]✓ Restored Claude settings from backup[/green]")
                
                # Check and fix permissions after restore
                self._check_claude_permissions()
                
                return True
                
            except Exception as e:
                console.print(f"[red]Error during restore: {e}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]Error restoring Claude settings: {e}[/red]")
            return False

    def _load_prompt(self) -> str:
        """Load prompt from file or use default"""
        if self.prompt_file:
            prompt_path = Path(self.prompt_file)
            if not prompt_path.exists():
                console.print(f"[red]Error: Prompt file not found: {self.prompt_file}[/red]")
                console.print("[yellow]Using default prompt instead[/yellow]")
                return DEFAULT_PROMPT.strip()
            prompt_text = prompt_path.read_text().strip()
            if not prompt_text:
                raise ValueError(f"Prompt file is empty: {self.prompt_file}")
            return prompt_text
        return DEFAULT_PROMPT.strip()

    def regenerate_problems(self) -> None:
        """Regenerate the type-checker and linter problems file"""
        if self.skip_regenerate:
            console.print("[yellow]Skipping problem file regeneration[/yellow]")
            return

        console.rule("[yellow]Regenerating type-check and lint output")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Running type-check and lint...", total=None)

            os.chdir(self.project_path)

            # Use a proper temporary file to avoid conflicts
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.project_path, prefix="combined_", suffix=".tmp", delete=False
            ) as tmpfile:
                tmpfile_path = Path(tmpfile.name)

                tmpfile.write("$ bun run type-check\n")
                tmpfile.flush()

                # Check if we should continue
                if not self.running:
                    tmpfile_path.unlink(missing_ok=True)
                    raise KeyboardInterrupt()

                # Run type-check
                subprocess.run(
                    ["bun", "run", "type-check"], stdout=tmpfile, stderr=subprocess.STDOUT, cwd=self.project_path
                )

                tmpfile.write("\n\n$ bun run lint\n")
                tmpfile.flush()

                # Check again before lint
                if not self.running:
                    tmpfile_path.unlink(missing_ok=True)
                    raise KeyboardInterrupt()

                # Run lint
                subprocess.run(["bun", "run", "lint"], stdout=tmpfile, stderr=subprocess.STDOUT, cwd=self.project_path)

            # Atomic rename (handle cross-filesystem moves)
            try:
                tmpfile_path.replace(self.combined_file)
            except OSError:
                # Fallback for cross-filesystem scenarios
                import shutil

                shutil.move(str(tmpfile_path), str(self.combined_file))
            finally:
                # Clean up temp file if it still exists
                tmpfile_path.unlink(missing_ok=True)

            progress.update(task, completed=True)

        count = line_count(self.combined_file)
        console.print(f"[green]✓ Generated {count} lines of problems[/green]")

    def commit_and_push(self) -> None:
        """Commit and push the updated problem count"""
        if self.skip_commit:
            console.print("[yellow]Skipping git commit/push[/yellow]")
            return

        console.rule("[yellow]Committing updated problem count")

        # Verify we're in a git repository
        ret, _, _ = run("git rev-parse --is-inside-work-tree", capture=True, quiet=True, check=False)
        if ret != 0:
            console.print("[red]Error: Not in a git repository[/red]")
            console.print("[yellow]Skipping git operations[/yellow]")
            return

        # Check for uncommitted changes
        # First, check if the problems file exists and has changes
        if self.combined_file.exists():
            ret, stdout, _ = run(
                f"git diff --name-only {shlex.quote(str(self.combined_file))}", capture=True, quiet=True
            )
            if stdout.strip():
                console.print(
                    f"[yellow]Warning: {self.combined_file.name} has uncommitted changes that will be committed[/yellow]"
                )

        ret, stdout, _ = run("git status --porcelain", capture=True, quiet=True)
        if stdout.strip():
            console.print("[yellow]You have uncommitted changes:[/yellow]")
            console.print(stdout)
            console.print("[yellow]The agent farm will add and commit the problems file.[/yellow]")
            console.print("[yellow]Other uncommitted changes will remain uncommitted.[/yellow]")
            if not interruptible_confirm("Do you want to continue?"):
                raise typer.Exit(1)

        # Ensure we're on a branch (not detached HEAD)
        ret, stdout, _ = run("git symbolic-ref HEAD", capture=True, quiet=True, check=False)
        if ret != 0:
            console.print("[red]Error: You're in a detached HEAD state[/red]")
            console.print("[yellow]Please checkout a branch before running the agent farm[/yellow]")
            raise typer.Exit(1)

        count = line_count(self.combined_file)

        try:
            run(f"git add {shlex.quote(str(self.combined_file))}")

            # Capture commit output to check if anything was actually committed
            commit_cmd = ["git", "commit", "-m", f"Before next round of fixes; currently {count} lines of problems"]
            commit_result = subprocess.run(commit_cmd, capture_output=True, text=True)

            both = (commit_result.stdout or "") + (commit_result.stderr or "")
            if commit_result.returncode != 0:  # noqa: SIM102
                if "nothing to commit" in both or "no changes added" in both:
                    console.print("[yellow]No changes to commit - skipping push[/yellow]")
                    return
                # Other errors still show warning below

            # Determine branch and remote
            branch = self.git_branch
            if not branch:
                _, stdout, _ = run("git rev-parse --abbrev-ref HEAD", capture=True, quiet=True)
                branch = stdout.strip() or "HEAD"

            remote = self.git_remote
            run(f"git push {remote} {branch}", check=False)
            console.print(f"[green]✓ Pushed commit with {count} current problems[/green]")
        except subprocess.CalledProcessError:
            console.print("[yellow]⚠ git commit/push skipped (no changes?)")

    def _wait_for_shell_prompt(self, pane_target: str, timeout: int = 30, ignore_shutdown: bool = False) -> bool:
        """Wait for a shell prompt to appear in the pane
        
        Args:
            pane_target: The tmux pane to check
            timeout: Maximum time to wait in seconds
            ignore_shutdown: If True, don't abort on shutdown signal (for error recovery)
        """
        start_time = time.time()
        content: str = ""  # prevents UnboundLocalError on early exit
        last_content = ""
        debug_shown = False

        # Get the last part of the project path for detection
        project_dir_name = self.project_path.name

        while time.time() - start_time < timeout and (ignore_shutdown or self.running):
            content = tmux_capture(pane_target)
            if content and content != last_content:
                lines = content.strip().splitlines()
                if lines:
                    # Only show debug output once per pane
                    if not debug_shown and len(lines[-1]) > 0:
                        console.print(f"[dim]Pane {pane_target}: ...{lines[-1][-50:]}[/dim]")
                        debug_shown = True

                    # Skip if direnv is still loading
                    if any("direnv:" in line for line in lines[-3:]):
                        continue  # Wait for direnv to finish

                    # Check if we see the project directory in the prompt
                    # This works with most modern shells that show the current directory
                    # Look through ALL lines, not just the last few, since prompts can be complex
                    for line in lines:
                        if project_dir_name in line:
                            console.print(f"[dim]Found project directory '{project_dir_name}' in prompt[/dim]")
                            return True

                    # Fall back to traditional prompt detection
                    # Look for common shell prompt patterns at the end of the last line
                    if re.search(r"[\$%>#]\s*$", lines[-1]):
                        return True
                    # Also check for other common prompt patterns
                    if re.search(r"❯\s*$", lines[-1]):  # Starship/PowerLevel10k style  # noqa: RUF001
                        return True
                    if re.search(r"➜\s*$", lines[-1]):  # Oh-my-zsh default
                        return True
                    if re.search(r"\]\s*$", lines[-1]) and "[" in lines[-1]:  # Bracketed prompts
                        return True
                    # Check for git prompt patterns like "on main"
                    if re.search(r"on\s+\S+.*\s*$", lines[-1]):  # Git branch info
                        console.print("[dim]Detected git prompt pattern[/dim]")
                        return True
                last_content = content
            time.sleep(0.2)

        # Check if we were interrupted
        if not self.running:
            console.print("[yellow]Shell prompt check interrupted by shutdown signal[/yellow]")
            return False

        # Show final content if we timeout
        if content:
            lines = content.strip().splitlines()
            if lines:
                console.print(f"[yellow]Timeout waiting for prompt in {pane_target}[/yellow]")
                console.print(f"[yellow]Last line: {lines[-1]}[/yellow]")
                # Show more context to help debug
                if len(lines) > 1:
                    console.print("[dim]Previous lines:[/dim]")
                    for line in lines[-5:-1]:  # Show up to 4 lines before the last
                        console.print(f"[dim]  {line}[/dim]")
        else:
            console.print(f"[yellow]Timeout waiting for prompt in {pane_target}. No content captured.[/yellow]")

        return False

    def setup_tmux_session(self) -> None:
        """Create tmux session with tiled agent panes"""
        console.rule(f"[yellow]Creating tmux session '{self.session}' with {self.agents} agents")

        # Kill existing session if it exists
        run(f"tmux kill-session -t {self.session}", check=False, quiet=True)
        time.sleep(0.5)

        # Create new session with controller window
        run(f"tmux new-session -d -s {self.session} -n controller")
        # CRITICAL: Set POWERLEVEL9K_INSTANT_PROMPT=off for ALL panes including the first one
        run(f"tmux new-window -t {self.session} -n agents -e 'POWERLEVEL9K_INSTANT_PROMPT=off'", quiet=True)

        # Create agent panes in tiled layout
        for i in range(self.agents):
            if i > 0:
                # CRITICAL: Set POWERLEVEL9K_INSTANT_PROMPT=off when creating the pane
                # This ensures it's disabled BEFORE the shell initializes
                run(f"tmux split-window -t {self.session}:agents -e 'POWERLEVEL9K_INSTANT_PROMPT=off'", quiet=True)
                run(f"tmux select-layout -t {self.session}:agents tiled", quiet=True)

        # Get the actual pane IDs - retry until we have the right count
        console.print("[dim]Waiting for tmux panes to be created...[/dim]")
        pane_ids = []
        start_time = time.time()
        timeout = 10  # 10 seconds max to create panes

        while time.time() - start_time < timeout:
            _, pane_list, _ = run(
                f"tmux list-panes -t {self.session}:agents -F '#{{pane_index}}'", capture=True, quiet=True
            )
            pane_ids = sorted(  # make mapping deterministic
                (pid.strip() for pid in pane_list.strip().splitlines() if pid.strip()),
                key=int,
            )

            if len(pane_ids) == self.agents:
                break

            time.sleep(0.1)

        # Create mapping from agent ID to actual pane ID
        if len(pane_ids) != self.agents:
            console.print(f"[red]Error: Expected {self.agents} panes but found {len(pane_ids)}[/red]")
            console.print(f"[red]Pane IDs found: {pane_ids}[/red]")
            raise RuntimeError("Failed to create the expected number of tmux panes")

        for agent_id, pane_id in enumerate(pane_ids[: self.agents]):
            self.pane_mapping[agent_id] = f"{self.session}:agents.{pane_id}"

        # Wait for all panes to have shell prompts ready
        if not self.fast_start:
            console.print("[dim]Waiting for shell prompts in all panes...[/dim]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Initializing panes", total=self.agents)

                for agent_id, pane_target in self.pane_mapping.items():
                    if not self.running:
                        console.print("[yellow]Initialization interrupted by shutdown signal[/yellow]")
                        raise KeyboardInterrupt()

                    if self._wait_for_shell_prompt(pane_target):
                        progress.update(task, advance=1)
                    else:
                        console.print(f"[yellow]Warning: Pane {agent_id} slow to initialize[/yellow]")
                        progress.update(task, advance=1)
        else:
            console.print("[dim]Fast start enabled - skipping shell prompt checks[/dim]")
            # Just wait a moment for panes to settle
            time.sleep(1.0)

        # Enable mouse support if configured
        if self.tmux_mouse:
            run(f"tmux set-option -t {self.session} -g mouse on", quiet=True)

        # Automatically adjust font size for many panes
        if self.agents >= 10:
            console.print(f"[yellow]Optimizing display for {self.agents} panes...[/yellow]")

            # Calculate zoom level needed
            zoom_steps = 3 if self.agents >= 20 else 2 if self.agents >= 15 else 1

            # If we're already in tmux, we can control the client zoom
            if os.environ.get("TMUX"):
                # We're in tmux - use client commands
                for _ in range(zoom_steps):
                    run("tmux resize-pane -Z", quiet=True, check=False)  # Toggle zoom
                    run("tmux send-keys C--", quiet=True, check=False)  # Send Ctrl+-
                    time.sleep(0.05)
            else:
                # Try to auto-zoom using escape sequences (works in some terminals)
                # This will be ignored by terminals that don't support it
                if sys.platform == "darwin":  # macOS
                    # iTerm2 specific
                    sys.stdout.write(f"\033]1337;SetProfile=FontSize={(14 - zoom_steps)}\007")
                else:
                    # Generic terminals - try zoom out escape sequence
                    for _ in range(zoom_steps):
                        sys.stdout.write("\033[1;2'-'")  # Some terminals recognize this
                sys.stdout.flush()

            # Optimize tmux display
            run(f"tmux set-option -t {self.session} -g pane-border-style 'fg=colour240,bg=colour235'", quiet=True)
            run(
                f"tmux set-option -t {self.session} -g pane-active-border-style 'fg=colour250,bg=colour235'", quiet=True
            )
            run(f"tmux set-option -t {self.session} -g pane-border-lines single", quiet=True)
            run(f"tmux set-option -t {self.session} -g pane-border-status off", quiet=True)

            # Hide status bar to save space
            run(f"tmux set-option -t {self.session} -g status off", quiet=True)

            # Set aggressive resize for better space usage
            run(f"tmux set-option -t {self.session} -g aggressive-resize on", quiet=True)

            # If zoom didn't work automatically, remind user
            console.print("[dim]If text is too large, zoom out with: Ctrl/Cmd + minus (-)[/dim]")
            console.print(f"[dim]Zoom level suggested: {zoom_steps} step{'s' if zoom_steps > 1 else ''}[/dim]")

            self._zoom_adjusted = True

        # Launch monitor in controller window if monitoring is enabled
        if not self.no_monitor:
            # Get the current script path
            script_path = Path(__file__).resolve()
            # Launch monitor mode in controller pane
            monitor_cmd = f"cd {self.project_path} && {sys.executable} {script_path} monitor-only --path {self.project_path} --session {self.session}"
            tmux_send(f"{self.session}:controller", monitor_cmd)

        # Register cleanup handler
        if not self._cleanup_registered:
            import atexit

            atexit.register(self._emergency_cleanup)
            self._cleanup_registered = True

        console.print(f"[green]✓ Created session with {self.agents} panes[/green]")

    def _acquire_claude_lock(self, timeout: float = 5.0) -> bool:
        """Acquire a lock file to prevent concurrent Claude Code launches"""
        lock_file = Path.home() / ".claude" / ".agent_farm_launch.lock"
        lock_file.parent.mkdir(exist_ok=True)

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to create lock file exclusively
                fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()}\n".encode())
                os.close(fd)
                return True
            except FileExistsError:
                # Lock file exists, check if it's stale
                try:
                    if lock_file.exists():  # noqa: SIM102
                        # Check if lock is older than 30 seconds (likely stale)
                        if time.time() - lock_file.stat().st_mtime > 30:
                            lock_file.unlink()
                            continue
                except Exception:
                    pass
                time.sleep(0.1)

        return False

    def _release_claude_lock(self) -> None:
        """Release the Claude Code launch lock"""
        lock_file = Path.home() / ".claude" / ".agent_farm_launch.lock"
        with contextlib.suppress(Exception):
            lock_file.unlink()

    def _check_claude_permissions(self) -> bool:
        """Check and fix permissions on Claude settings files"""
        claude_dir = Path.home() / ".claude"
        settings_file = claude_dir / "settings.json"
        
        try:
            if settings_file.exists():
                # Get current permissions
                current_mode = settings_file.stat().st_mode & 0o777
                
                # Check if permissions are too open
                if current_mode != 0o600:
                    console.print(f"[yellow]Fixing permissions on {settings_file.name} (was {oct(current_mode)})[/yellow]")
                    os.chmod(settings_file, 0o600)
                
                # Ensure we own the file
                stat_info = settings_file.stat()
                if stat_info.st_uid != os.getuid():
                    console.print(f"[yellow]Fixing ownership on {settings_file.name}[/yellow]")
                    try:
                        os.chown(settings_file, os.getuid(), os.getgid())
                    except PermissionError:
                        console.print("[red]Could not change ownership - may need sudo[/red]")
                        return False
            
            # Check directory permissions
            if claude_dir.exists():
                dir_mode = claude_dir.stat().st_mode & 0o777
                if dir_mode not in (0o700, 0o755):
                    console.print(f"[yellow]Fixing permissions on .claude directory (was {oct(dir_mode)})[/yellow]")
                    os.chmod(claude_dir, 0o700)
            
            return True
            
        except Exception as e:
            console.print(f"[red]Error checking permissions: {e}[/red]")
            return False

    def start_agent(self, agent_id: int, restart: bool = False) -> None:
        """Start or restart a single agent"""
        pane_target = self.pane_mapping.get(agent_id)
        if not pane_target:
            console.print(f"[red]Error: No pane mapping found for agent {agent_id}[/red]")
            return

        if restart:
            # Exit current session
            tmux_send(pane_target, "/exit")
            # Wait for shell prompt to appear
            if not self._wait_for_shell_prompt(pane_target, timeout=15):
                console.print(f"[yellow]Warning: Agent {agent_id} pane slow to return to shell prompt[/yellow]")

            if self.monitor:
                self.monitor.agents[agent_id]["cycles"] += 1

        # Navigate and start Claude Code
        # Quote the path so directories with spaces or shell metacharacters work
        tmux_send(pane_target, f"cd {shlex.quote(str(self.project_path))}")
        
        # CRITICAL: Wait a couple seconds for cd to complete and shell to stabilize
        # This prevents race conditions and ensures we're in the right directory
        console.print(f"[dim]Agent {agent_id:02d}: Waiting 2s after cd before launching cc...[/dim]")
        time.sleep(2.0)

        # Acquire lock before launching cc to prevent config corruption
        lock_acquired = False
        if not self._acquire_claude_lock(timeout=10.0):
            console.print(f"[yellow]Agent {agent_id:02d}: Waiting for lock to launch cc...[/yellow]")
            # Try once more with longer timeout
            if not self._acquire_claude_lock(timeout=20.0):
                console.print(f"[red]Agent {agent_id:02d}: Could not acquire lock - aborting launch[/red]")
                if self.monitor:
                    self.monitor.agents[agent_id]["status"] = "error"
                    self.monitor.agents[agent_id]["errors"] += 1
                return

        lock_acquired = True
        try:
            tmux_send(pane_target, "cc")

            if not restart:
                console.print(f"🛠  Agent {agent_id:02d}: launching cc, waiting {self.wait_after_cc}s...")

            # Make wait_after_cc interruptible
            for _ in range(int(self.wait_after_cc * 5)):
                if not self.running:
                    return
                time.sleep(0.2)
        finally:
            # Always release lock
            if lock_acquired:
                self._release_claude_lock()

        # Verify Claude Code started successfully
        max_retries = 5
        claude_started_successfully = False

        # Give Claude Code a bit more time to fully initialize before first check
        time.sleep(2.0)

        for attempt in range(max_retries):
            if not self.running:
                return

            content = tmux_capture(pane_target)

            # Debug log for troubleshooting startup detection
            if attempt == 0 and len(content.strip()) > 0:
                console.print(f"[dim]Agent {agent_id:02d}: Captured {len(content)} chars, checking readiness...[/dim]")
                # Log key indicators for debugging
                if "Welcome to Claude Code!" in content:
                    console.print(f"[dim]Agent {agent_id:02d}: Found welcome message[/dim]")
                if "│ >" in content:
                    console.print(f"[dim]Agent {agent_id:02d}: Found prompt box[/dim]")
                if "? for shortcuts" in content:
                    console.print(f"[dim]Agent {agent_id:02d}: Found shortcuts hint[/dim]")

            # Check for various failure conditions
            if not content or len(content.strip()) < 10:
                # Empty or nearly empty content indicates cc didn't start
                console.print(
                    f"[yellow]Agent {agent_id:02d}: No output from Claude Code yet (attempt {attempt + 1}/{max_retries})[/yellow]"
                )
            elif self.monitor and self.monitor.is_claude_ready(content):
                # Check for readiness FIRST before checking for errors
                claude_started_successfully = True
                break
            elif self.monitor and len(content.strip()) > 100 and (
                self.monitor.has_settings_error(content) or self.monitor.has_welcome_screen(content)
            ):
                # Only check for errors if we have substantial content (>100 chars)
                console.print(
                    f"[red]Agent {agent_id:02d}: Settings error/setup screen detected - attempting restore[/red]"
                )

                # Kill this cc instance more forcefully
                console.print(f"[yellow]Agent {agent_id:02d}: Killing corrupted Claude Code instance...[/yellow]")
                tmux_send(pane_target, "\x03")  # Ctrl+C
                time.sleep(0.5)
                tmux_send(pane_target, "\x03")  # Send Ctrl+C again to be sure
                time.sleep(0.5)
                
                # Send exit command in case it's still in Claude Code
                tmux_send(pane_target, "/exit")
                time.sleep(1.0)
                
                # Wait for shell prompt to ensure Claude Code has fully exited
                if not self._wait_for_shell_prompt(pane_target, timeout=10, ignore_shutdown=True):
                    # If still not at shell, try harder
                    tmux_send(pane_target, "\x03")  # Another Ctrl+C
                    time.sleep(1.0)
                    tmux_send(pane_target, "exit")  # Try shell exit command
                    time.sleep(1.0)
                
                # NEVER EVER kill all claude-code processes! This would kill ALL working agents!
                # Just let this specific instance clean up naturally
                time.sleep(2.0)  # Give time for this instance to fully exit

                # Try to restore from backup
                if hasattr(self, "settings_backup_path") and self.settings_backup_path:  # noqa: SIM102
                    if self._restore_claude_settings():
                        console.print(f"[green]Settings restored for agent {agent_id} - retrying launch[/green]")
                        # Wait a bit more to ensure everything is settled
                        time.sleep(2.0)
                        
                        # Return to shell and retry
                        if self._wait_for_shell_prompt(pane_target, timeout=10, ignore_shutdown=True):
                            # Re-acquire lock before retrying
                            if self._acquire_claude_lock(timeout=10.0):
                                try:
                                    tmux_send(pane_target, "cc")
                                    time.sleep(self.wait_after_cc)
                                finally:
                                    self._release_claude_lock()
                                # Continue to next iteration to check again
                                continue
                            else:
                                console.print(f"[red]Agent {agent_id:02d}: Could not acquire lock for retry[/red]")
                        else:
                            console.print(f"[red]Agent {agent_id:02d}: Could not get shell prompt for retry[/red]")

                if self.monitor:
                    self.monitor.agents[agent_id]["status"] = "error"
                    self.monitor.agents[agent_id]["errors"] += 1
                return
            elif "command not found" in content and "cc" in content:
                # cc command doesn't exist
                console.print(f"[red]Agent {agent_id:02d}: 'cc' command not found[/red]")
                if self.monitor:
                    self.monitor.agents[agent_id]["status"] = "error"
                    self.monitor.agents[agent_id]["errors"] += 1
                return
            elif attempt < max_retries - 1:
                # Make this sleep interruptible too
                for _ in range(25):  # 5 seconds in 0.2s intervals
                    if not self.running:
                        return
                    time.sleep(0.2)
            else:
                console.print(
                    f"[red]Agent {agent_id:02d}: Claude Code failed to start properly after {max_retries} attempts[/red]"
                )
                if self.monitor:
                    self.monitor.agents[agent_id]["status"] = "error"
                    self.monitor.agents[agent_id]["errors"] += 1
                return

        # Only send prompt if Claude Code started successfully
        if not claude_started_successfully:
            console.print(f"[red]Agent {agent_id:02d}: Skipping prompt injection - Claude Code not ready[/red]")
            return

        # Send prompt with unique seed for randomization
        seed = randint(100000, 999999)
        # Use regex to handle variations like "random chunks of 50 lines"
        salted_prompt = re.sub(
            r"random chunks(\b.*?\b)?",
            lambda m: f"{m.group(0)} (instance-seed {seed})",
            self.prompt_text,
            count=1,
            flags=re.IGNORECASE,
        )

        # Send prompt as a single message
        # CRITICAL: Use tmux's literal mode to send the entire prompt correctly
        # This avoids complex line-by-line sending that can cause issues
        console.print(f"[dim]Agent {agent_id:02d}: Sending {len(salted_prompt)} char prompt...[/dim]")
        
        # Send the entire prompt at once using literal mode
        tmux_send(pane_target, salted_prompt, enter=True)

        if not restart:
            console.print(f"[green]✓ Agent {agent_id:02d}: prompt injected[/green]")

        # Verify prompt was received by checking for working state
        time.sleep(2.0)  # Give Claude a moment to start processing
        verify_content = tmux_capture(pane_target)
        if self.monitor and self.monitor.is_claude_working(verify_content):
            console.print(f"[green]✓ Agent {agent_id:02d}: Claude Code is processing the prompt[/green]")
        else:
            console.print(f"[yellow]⚠ Agent {agent_id:02d}: Claude Code may not have received the prompt properly[/yellow]")

        if self.monitor:
            self.monitor.agents[agent_id]["status"] = "starting"
            self.monitor.agents[agent_id]["last_activity"] = datetime.now()

    def launch_agents(self) -> None:
        """Launch all agents with staggered start times"""
        console.rule("[yellow]Launching agents")

        # Track successful launches to detect corruption patterns
        successful_launches = 0
        corruption_detected = False

        for i in range(self.agents):
            if not self.running:
                console.print("[yellow]Agent launch interrupted by shutdown signal[/yellow]")
                break

            self.start_agent(i)

            # Check if last agent started successfully
            if self.monitor and i > 0:
                time.sleep(1)  # Brief pause to let status update
                prev_agent_status = self.monitor.agents[i]["status"]
                if prev_agent_status == "error":
                    corruption_detected = True
                    console.print("[yellow]Detected potential config corruption - increasing launch delay[/yellow]")
                else:
                    successful_launches += 1

            # Stagger starts to avoid config clobbering
            if i < self.agents - 1 and self.running:
                # Dynamic stagger time based on corruption detection
                if corruption_detected:
                    # Use longer delay if we've seen errors
                    stagger_time = self.stagger * 2.0
                    console.print(f"[dim]Using extended stagger time: {stagger_time}s[/dim]")
                else:
                    # Gradually increase stagger time as we launch more agents
                    # This helps prevent pile-up effects
                    stagger_time = self.stagger * (1.0 + i * 0.1)

                # Use smaller sleep intervals to be more responsive to shutdown
                for _ in range(int(stagger_time * 5)):
                    if not self.running:
                        break
                    time.sleep(0.2)

        if successful_launches < self.agents:
            console.print(
                f"[yellow]Warning: Only {successful_launches}/{self.agents} agents launched successfully[/yellow]"
            )
            if self.auto_restart:
                console.print("[yellow]Auto-restart enabled - failed agents will be retried[/yellow]")

    def monitor_loop(self) -> None:
        """Main monitoring loop with auto-restart capability"""
        if self.no_monitor:
            console.print("[yellow]Monitoring disabled. Agents will run without supervision.[/yellow]")
            console.print(f"[cyan]Attach with: tmux attach -t {self.session}[/cyan]")
            return

        console.rule("[green]All agents launched - Monitoring active")
        console.print("[yellow]Monitor dashboard running in tmux controller window[/yellow]")
        console.print(f"[cyan]View with: tmux attach -t {self.session}:controller[/cyan]")
        console.print("[cyan]Or use: ./view_agents.sh[/cyan]")
        console.print("[dim]Press Ctrl+C here for graceful shutdown[/dim]\n")

        if self.monitor:
            check_counter = 0

            while self.running:
                # Check agents every N seconds
                if check_counter % self.check_interval == 0:
                    for agent_id in range(self.agents):
                        self.monitor.check_agent(agent_id)

                        # Auto-restart if needed
                        if self.auto_restart and self.monitor.needs_restart(agent_id):
                            agent = self.monitor.agents[agent_id]

                            # Implement exponential backoff
                            if agent["last_restart"]:
                                time_since_restart = (datetime.now() - agent["last_restart"]).total_seconds()
                                backoff_time = min(300, 10 * (2 ** agent["restart_count"]))  # Max 5 min

                                if time_since_restart < backoff_time:
                                    continue  # Skip restart, still in backoff period

                            console.print(
                                f"[yellow]Restarting agent {agent_id} (attempt #{agent['restart_count'] + 1})...[/yellow]"
                            )
                            self.start_agent(agent_id, restart=True)

                            # Update restart tracking
                            agent["restart_count"] += 1
                            agent["last_restart"] = datetime.now()
                            self.monitor.agents[agent_id] = agent

                # Write state to file for monitor process
                self.write_monitor_state()
                time.sleep(1)
                check_counter += 1

    def run(self) -> None:
        """Main orchestration flow"""
        os.chdir(self.project_path)

        # Display startup banner
        console.print(
            Panel.fit(
                f"[bold cyan]Claude Code Agent Farm[/bold cyan]\n"
                f"Project: {self.project_path}\n"
                f"Agents: {self.agents}\n"
                f"Session: {self.session}\n"
                f"Auto-restart: {'enabled' if self.auto_restart else 'disabled'}",
                border_style="cyan",
            )
        )

        # Backup Claude settings before starting
        self.settings_backup_path = self._backup_claude_settings()
        
        # Check current permissions are correct
        self._check_claude_permissions()

        try:
            # Execute workflow steps
            self.regenerate_problems()
            self.commit_and_push()
            self.setup_tmux_session()

            # Initialize monitor
            self.monitor = AgentMonitor(
                self.session,
                self.agents,
                self.pane_mapping,
                context_threshold=self.context_threshold,
                idle_timeout=self.idle_timeout,
                max_errors=self.max_errors,
            )

            # Launch agents
            self.launch_agents()

            # Start monitoring
            self.monitor_loop()

            # Attach to session if requested
            if self.attach and not self.no_monitor:
                run(f"tmux attach-session -t {self.session}", check=False)

        except KeyboardInterrupt:
            # Don't print anything here - let the main handler deal with it
            raise

    def shutdown(self) -> None:
        """Clean shutdown of all agents"""
        console.print("\n[yellow]Shutting down agents...[/yellow]")

        for i in range(self.agents):
            pane_target = self.pane_mapping.get(i)
            if pane_target:
                tmux_send(pane_target, "/exit")

        time.sleep(2)

        if getattr(self, "tmux_kill_on_exit", True):
            run(f"tmux kill-session -t {self.session}", check=False, quiet=True)
            console.print("[green]✓ tmux session terminated[/green]")
        else:
            console.print("[yellow]tmux left running (tmux_kill_on_exit = false)[/yellow]")

        # Clean up state file
        if hasattr(self, "state_file") and self.state_file.exists():
            self.state_file.unlink()
            console.print("[green]✓ Monitor state file cleaned up[/green]")

        # Restore font size if we changed it
        if hasattr(self, "_zoom_adjusted") and self._zoom_adjusted:
            console.print("[dim]Note: You may want to restore zoom with Ctrl/Cmd + 0[/dim]")

    def _emergency_cleanup(self) -> None:
        """Emergency cleanup handler for unexpected exits"""
        with contextlib.suppress(Exception):
            # Kill the tmux session if it exists
            run(f"tmux kill-session -t {self.session}", check=False, quiet=True)
            # Clean up state file
            if hasattr(self, "state_file") and self.state_file.exists():
                self.state_file.unlink()
            # Clean up lock file
            lock_file = Path.home() / ".claude" / ".agent_farm_launch.lock"
            if lock_file.exists():
                lock_file.unlink()

    def write_monitor_state(self) -> None:
        """Write current monitor state to shared file"""
        if not self.monitor:
            return

        def _serialise_agent(a: dict) -> dict:
            """Convert datetime objects to ISO strings for JSON serialization"""
            return {
                **a,
                "start_time": a["start_time"].isoformat(),
                "last_activity": a["last_activity"].isoformat(),
                "last_restart": a["last_restart"].isoformat() if a.get("last_restart") is not None else None,
            }

        state_data = {
            "session": self.session,
            "num_agents": self.agents,
            "agents": {str(k): _serialise_agent(v) for k, v in self.monitor.agents.items()},
            "start_time": self.monitor.start_time.isoformat(),
            "timestamp": datetime.now().isoformat(),
        }

        # Write atomically with file locking
        tmp_file = self.state_file.with_suffix(".tmp")
        try:
            with tmp_file.open("w") as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(state_data, f)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Atomic rename
            tmp_file.replace(self.state_file)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not write monitor state: {e}[/yellow]")
            tmp_file.unlink(missing_ok=True)


# ─────────────────────────────── CLI Entry Point ──────────────────────────── #


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    path: str = typer.Option(..., "--path", help="Absolute path to project root", rich_help_panel="Required Arguments"),
    agents: int = typer.Option(
        20, "--agents", "-n", help="Number of Claude agents", rich_help_panel="Agent Configuration"
    ),
    session: str = typer.Option(
        "claude_agents", "--session", "-s", help="tmux session name", rich_help_panel="Agent Configuration"
    ),
    stagger: float = typer.Option(
        10.0, "--stagger", help="Seconds between starting agents", rich_help_panel="Timing Configuration"
    ),
    wait_after_cc: float = typer.Option(
        15.0, "--wait-after-cc", help="Seconds to wait after launching cc", rich_help_panel="Timing Configuration"
    ),
    check_interval: int = typer.Option(
        10, "--check-interval", help="Seconds between agent health checks", rich_help_panel="Timing Configuration"
    ),
    skip_regenerate: bool = typer.Option(
        False, "--skip-regenerate", help="Skip regenerating problems file", rich_help_panel="Feature Flags"
    ),
    skip_commit: bool = typer.Option(
        False, "--skip-commit", help="Skip git commit/push", rich_help_panel="Feature Flags"
    ),
    auto_restart: bool = typer.Option(
        False, "--auto-restart", help="Auto-restart agents on errors/completion", rich_help_panel="Feature Flags"
    ),
    no_monitor: bool = typer.Option(
        False, "--no-monitor", help="Disable monitoring (just launch and exit)", rich_help_panel="Feature Flags"
    ),
    attach: bool = typer.Option(
        False, "--attach", help="Attach to tmux session after setup", rich_help_panel="Feature Flags"
    ),
    prompt_file: Optional[str] = typer.Option(
        None, "--prompt-file", help="Path to custom prompt file", rich_help_panel="Advanced Options"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Load settings from JSON config file", rich_help_panel="Advanced Options"
    ),
    context_threshold: int = typer.Option(
        20,
        "--context-threshold",
        help="Restart agent when context ≤ this percentage",
        rich_help_panel="Advanced Options",
    ),
    idle_timeout: int = typer.Option(
        60,
        "--idle-timeout",
        help="Seconds of inactivity before marking agent as idle",
        rich_help_panel="Advanced Options",
    ),
    max_errors: int = typer.Option(
        3, "--max-errors", help="Maximum consecutive errors before disabling agent", rich_help_panel="Advanced Options"
    ),
    tmux_kill_on_exit: bool = typer.Option(
        True,
        "--tmux-kill-on-exit/--no-tmux-kill-on-exit",
        help="Kill tmux session on exit",
        rich_help_panel="Advanced Options",
    ),
    tmux_mouse: bool = typer.Option(
        True, "--tmux-mouse/--no-tmux-mouse", help="Enable tmux mouse support", rich_help_panel="Advanced Options"
    ),
    fast_start: bool = typer.Option(
        False, "--fast-start", help="Skip shell prompt checking", rich_help_panel="Advanced Options"
    ),
    full_backup: bool = typer.Option(
        False, "--full-backup", help="Perform a full backup before starting", rich_help_panel="Advanced Options"
    ),
) -> None:
    """
    Claude Code Agent Farm - Parallel code fixing automation

    This tool orchestrates multiple Claude Code agents working in parallel
    to fix type-checker and linter problems in your codebase.
    """
    # If a subcommand was invoked, don't run the main logic
    if ctx.invoked_subcommand is not None:
        return

    # Validate project path
    project_path = Path(path).expanduser().resolve()
    if not project_path.is_dir():
        console.print(f"[red]✖ {project_path} is not a directory[/red]")
        raise typer.Exit(1)

    # Validate agent count
    if agents < 1:
        console.print("[red]✖ Number of agents must be at least 1[/red]")
        raise typer.Exit(1)

    if agents > 50:
        console.print(f"[yellow]⚠ Running {agents} agents may consume significant resources[/yellow]")
        if not interruptible_confirm("Do you want to continue?"):
            raise KeyboardInterrupt()

    # Create and run the orchestrator
    farm = ClaudeAgentFarm(
        path=path,
        agents=agents,
        session=session,
        stagger=stagger,
        wait_after_cc=wait_after_cc,
        check_interval=check_interval,
        skip_regenerate=skip_regenerate,
        skip_commit=skip_commit,
        auto_restart=auto_restart,
        no_monitor=no_monitor,
        attach=attach,
        prompt_file=prompt_file,
        config=config,
        context_threshold=context_threshold,
        idle_timeout=idle_timeout,
        max_errors=max_errors,
        tmux_kill_on_exit=tmux_kill_on_exit,
        tmux_mouse=tmux_mouse,
        fast_start=fast_start,
        full_backup=full_backup,
    )

    try:
        farm.run()
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted by user[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
    finally:
        farm.shutdown()


@app.command(name="monitor-only", hidden=True)
def monitor_only(
    path: str = typer.Option(..., "--path", help="Absolute path to project root"),
    session: str = typer.Option("claude_agents", "--session", "-s", help="tmux session name"),
) -> None:
    """Run monitor display only - internal command used by the orchestrator"""
    project_path = Path(path).expanduser().resolve()
    state_file = project_path / MONITOR_STATE_FILE

    console.print(
        Panel.fit(
            f"[bold cyan]Claude Agent Farm Monitor[/bold cyan]\nSession: {session}\nPress Ctrl+C to exit monitor view",
            border_style="cyan",
        )
    )

    with Live(console=console, refresh_per_second=1) as live:
        while True:
            try:
                if state_file.exists():
                    # Use file locking when reading to avoid race conditions
                    try:
                        with state_file.open() as f:
                            # Try to acquire shared lock (non-exclusive)
                            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                            try:
                                state_data = json.load(f)
                            finally:
                                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except (IOError, OSError) as e:
                        raise Exception(f"Failed to read state file: {e}") from e

                    # Recreate table from state data
                    table = Table(title=f"Claude Agent Farm - {datetime.now().strftime('%H:%M:%S')}")
                    table.add_column("Agent", style="cyan", width=8)
                    table.add_column("Status", style="green", width=10)
                    table.add_column("Cycles", style="yellow", width=6)
                    table.add_column("Context", style="magenta", width=8)
                    table.add_column("Runtime", style="blue", width=12)
                    table.add_column("Errors", style="red", width=6)

                    agents = state_data.get("agents", {})

                    for agent_id in sorted(int(k) for k in agents):
                        agent = agents[str(agent_id)]
                        agent_start = datetime.fromisoformat(agent["start_time"])
                        runtime = str(datetime.now() - agent_start).split(".")[0]

                        status_style = {
                            "working": "[green]",
                            "ready": "[cyan]",
                            "idle": "[yellow]",
                            "error": "[red]",
                            "starting": "[yellow]",
                            "unknown": "[dim]",
                        }.get(agent["status"], "")

                        table.add_row(
                            f"Pane {agent_id:02d}",
                            f"{status_style}{agent['status']}[/]",
                            str(agent["cycles"]),
                            f"{agent['last_context']}%",
                            runtime,
                            str(agent["errors"]),
                        )

                    live.update(table)
                else:
                    live.update("[yellow]Waiting for monitor data...[/yellow]")

                time.sleep(1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                live.update(f"[red]Error reading state: {e}[/red]")
                time.sleep(1)


if __name__ == "__main__":
    app()
