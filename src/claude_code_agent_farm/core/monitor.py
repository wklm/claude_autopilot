"""Agent monitoring functionality for Claude Code Agent Farm."""

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rich import box
from rich.console import Console
from rich.table import Table

from claude_code_agent_farm.config import constants
from claude_code_agent_farm.integrations.tmux_composer import TmuxComposer

console = Console(stderr=True)


class AgentMonitor:
    """Monitors Claude Code agents for health and performance."""

    def __init__(
        self,
        session: str,
        num_agents: int,
        tmux: TmuxComposer,
        context_threshold: int = constants.DEFAULT_CONTEXT_THRESHOLD,
        idle_timeout: int = constants.DEFAULT_IDLE_TIMEOUT,
        max_errors: int = constants.DEFAULT_MAX_ERRORS,
        project_path: Optional[Path] = None,
    ):
        self.session = session
        self.num_agents = num_agents
        self.tmux = tmux
        self.agents: Dict[int, Dict] = {}
        self.running = True
        self.start_time = datetime.now()
        self.context_threshold = context_threshold
        self.idle_timeout = idle_timeout
        self.base_idle_timeout = idle_timeout
        self.max_errors = max_errors
        self.project_path = project_path
        
        # Cycle time tracking for adaptive timeout
        self.cycle_times: List[float] = []
        self.max_cycle_history = 20
        
        # Setup heartbeats directory
        self.heartbeats_dir: Optional[Path] = None
        if self.project_path:
            self.heartbeats_dir = self.project_path / constants.HEARTBEATS_DIR
            self.heartbeats_dir.mkdir(exist_ok=True)
            # Clean up any old heartbeat files
            for hb_file in self.heartbeats_dir.glob("agent*.heartbeat"):
                hb_file.unlink(missing_ok=True)

        # Initialize agent tracking
        for i in range(num_agents):
            self.agents[i] = {
                "status": constants.STATUS_STARTING,
                "start_time": datetime.now(),
                "cycles": 0,
                "last_context": 100,
                "errors": 0,
                "last_activity": datetime.now(),
                "restart_count": 0,
                "last_restart": None,
                "last_heartbeat": None,
                "cycle_start_time": None,
            }

    def calculate_adaptive_timeout(self) -> int:
        """Calculate adaptive idle timeout based on median cycle time."""
        if len(self.cycle_times) < 3:
            return self.base_idle_timeout
        
        # Calculate median cycle time
        sorted_times = sorted(self.cycle_times)
        median_time = sorted_times[len(sorted_times) // 2]
        
        # Set timeout to 3x median cycle time, but within reasonable bounds
        adaptive_timeout = int(median_time * 3)
        
        # Enforce minimum and maximum bounds
        min_timeout = 30  # At least 30 seconds
        max_timeout = 600  # At most 10 minutes
        
        adaptive_timeout = max(min_timeout, min(adaptive_timeout, max_timeout))
        
        # Only update if significantly different from current (>20% change)
        if abs(adaptive_timeout - self.idle_timeout) / self.idle_timeout > 0.2:
            console.print(f"[dim]Adjusting idle timeout: {self.idle_timeout}s → {adaptive_timeout}s (median cycle: {median_time:.1f}s)[/dim]")
            self.idle_timeout = adaptive_timeout
        
        return self.idle_timeout
    
    def detect_context_percentage(self, content: str) -> Optional[int]:
        """Extract context percentage from pane content."""
        if not content:
            return None

        for pattern in constants.CONTEXT_PATTERNS:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def is_claude_ready(self, content: str) -> bool:
        """Check if Claude Code is ready for input."""
        return any(indicator in content for indicator in constants.CLAUDE_READY_INDICATORS)

    def is_claude_working(self, content: str) -> bool:
        """Check if Claude Code is actively working."""
        return any(indicator in content for indicator in constants.CLAUDE_WORKING_INDICATORS)

    def has_welcome_screen(self, content: str) -> bool:
        """Check if Claude Code is showing the welcome/setup screen."""
        return any(indicator in content for indicator in constants.CLAUDE_WELCOME_INDICATORS)

    def has_settings_error(self, content: str) -> bool:
        """Check for settings corruption."""
        # First check if Claude is actually ready (avoid false positives)
        if self.is_claude_ready(content):
            return False
            
        return any(indicator in content for indicator in constants.CLAUDE_ERROR_INDICATORS)

    def check_agent(self, agent_id: int) -> Dict:
        """Check status of a single agent."""
        content = self.tmux.capture_pane(agent_id)
        agent = self.agents[agent_id]

        # Update context percentage
        context = self.detect_context_percentage(content)
        if context is not None:
            agent["last_context"] = context

        # Check for errors
        if self.has_settings_error(content):
            agent["status"] = constants.STATUS_ERROR
            agent["errors"] += 1
        else:
            # Store previous status to detect transitions
            prev_status = agent.get("status", constants.STATUS_UNKNOWN)
            
            # Update status based on activity
            if self.is_claude_working(content):
                # If transitioning to working, record cycle start time
                if prev_status != constants.STATUS_WORKING and agent["cycle_start_time"] is None:
                    agent["cycle_start_time"] = datetime.now()
                
                agent["status"] = constants.STATUS_WORKING
                agent["last_activity"] = datetime.now()
                self._update_heartbeat(agent_id)
            elif self.is_claude_ready(content):
                # If transitioning from working to ready, record cycle time
                if prev_status == constants.STATUS_WORKING and agent["cycle_start_time"] is not None:
                    cycle_time = (datetime.now() - agent["cycle_start_time"]).total_seconds()
                    self.cycle_times.append(cycle_time)
                    
                    # Keep only recent cycle times
                    if len(self.cycle_times) > self.max_cycle_history:
                        self.cycle_times.pop(0)
                    
                    # Update adaptive timeout
                    self.calculate_adaptive_timeout()
                    
                    # Reset cycle start time
                    agent["cycle_start_time"] = None
                    
                    # Increment cycle count
                    agent["cycles"] += 1
                
                # Check if idle for too long
                idle_time = (datetime.now() - agent["last_activity"]).total_seconds()
                if idle_time > self.idle_timeout:
                    agent["status"] = constants.STATUS_IDLE
                else:
                    agent["status"] = constants.STATUS_READY
                    self._update_heartbeat(agent_id)
            else:
                agent["status"] = constants.STATUS_UNKNOWN
        
        # Update tmux pane title with context information
        self._update_pane_title(agent_id, agent)

        return agent
    
    def _update_pane_title(self, agent_id: int, agent: Dict) -> None:
        """Update tmux pane title with agent status and context percentage."""
        # Build title with context warning
        context = agent["last_context"]
        status = agent["status"]
        
        # Create context indicator with warning colors
        if context <= self.context_threshold:
            context_str = f"⚠️ {context}%"
        elif context <= 30:
            context_str = f"⚡{context}%"
        else:
            context_str = f"{context}%"
        
        # Status emoji
        status_emoji = constants.STATUS_EMOJIS.get(status, "")
        
        # Build title
        title = f"[{agent_id:02d}] {status_emoji} Context: {context_str}"
        
        # Set pane title
        self.tmux.set_pane_title(agent_id, title)

    def _update_heartbeat(self, agent_id: int) -> None:
        """Update heartbeat file for an agent."""
        if not self.heartbeats_dir:
            return
        
        heartbeat_file = self.heartbeats_dir / f"agent{agent_id:02d}.heartbeat"
        try:
            heartbeat_file.write_text(datetime.now().isoformat())
            self.agents[agent_id]["last_heartbeat"] = datetime.now()
        except Exception:
            pass
    
    def _check_heartbeat_age(self, agent_id: int) -> Optional[float]:
        """Check age of heartbeat file in seconds."""
        if not self.heartbeats_dir:
            return None
        
        heartbeat_file = self.heartbeats_dir / f"agent{agent_id:02d}.heartbeat"
        if not heartbeat_file.exists():
            return None
        
        try:
            mtime = heartbeat_file.stat().st_mtime
            age = time.time() - mtime
            return age
        except Exception:
            return None
    
    def needs_restart(self, agent_id: int) -> bool:
        """Determine if an agent needs to be restarted."""
        agent = self.agents[agent_id]
        
        # Check heartbeat age - if older than 2 minutes, agent might be stuck
        heartbeat_age = self._check_heartbeat_age(agent_id)
        if heartbeat_age is not None and heartbeat_age > 120:
            console.print(f"[yellow]Agent {agent_id} heartbeat is {heartbeat_age:.0f}s old[/yellow]")
            return True

        # Restart conditions
        return bool(
            agent["status"] == constants.STATUS_ERROR
            or agent["errors"] >= self.max_errors
            or agent["status"] == constants.STATUS_IDLE
            or agent["last_context"] <= self.context_threshold
        )

    def get_status_table(self) -> Table:
        """Generate status table for all agents."""
        table = Table(
            title=f"Claude Agent Farm - {datetime.now().strftime('%H:%M:%S')}",
            box=box.ROUNDED,
        )

        table.add_column("Agent", style="cyan", width=8)
        table.add_column("Status", style="green", width=10)
        table.add_column("Cycles", style="yellow", width=6)
        table.add_column("Context", style="magenta", width=8)
        table.add_column("Runtime", style="blue", width=12)
        table.add_column("Heartbeat", style="cyan", width=8)
        table.add_column("Errors", style="red", width=6)

        for agent_id in sorted(self.agents.keys()):
            agent = self.agents[agent_id]
            runtime = str(datetime.now() - agent["start_time"]).split(".")[0]

            status_style = {
                constants.STATUS_WORKING: "[green]",
                constants.STATUS_READY: "[cyan]",
                constants.STATUS_IDLE: "[yellow]",
                constants.STATUS_ERROR: "[red]",
                constants.STATUS_USAGE_LIMIT: "[bold red]",
                constants.STATUS_STARTING: "[yellow]",
                constants.STATUS_UNKNOWN: "[dim]",
            }.get(agent["status"], "")
            
            # Get heartbeat age
            heartbeat_age = self._check_heartbeat_age(agent_id)
            if heartbeat_age is None:
                heartbeat_str = "---"
            elif heartbeat_age < 30:
                heartbeat_str = f"[green]{heartbeat_age:.0f}s[/green]"
            elif heartbeat_age < 60:
                heartbeat_str = f"[yellow]{heartbeat_age:.0f}s[/yellow]"
            else:
                heartbeat_str = f"[red]{heartbeat_age:.0f}s[/red]"

            table.add_row(
                f"Pane {agent_id:02d}",
                f"{status_style}{agent['status']}[/]",
                str(agent["cycles"]),
                f"{agent['last_context']}%",
                runtime,
                heartbeat_str,
                str(agent["errors"]),
            )

        return table