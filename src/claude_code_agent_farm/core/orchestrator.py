"""Main orchestrator for Claude Code Agent Farm."""

import contextlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from claude_code_agent_farm.config import constants
from claude_code_agent_farm.config.settings import Settings
from claude_code_agent_farm.core.monitor import AgentMonitor
from claude_code_agent_farm.integrations import ClaudeAutoResume, ClaudeHooks, TmuxComposer
from claude_code_agent_farm.utils import line_count, run

console = Console(stderr=True)


class ClaudeAgentFarm:
    """Orchestrate multiple Claude Code agents for parallel work."""
    
    def __init__(
        self,
        path: str,
        agents: int = constants.DEFAULT_NUM_AGENTS,
        session: str = constants.DEFAULT_SESSION_NAME,
        stagger: float = constants.DEFAULT_STAGGER_TIME,
        wait_after_cc: float = constants.DEFAULT_WAIT_AFTER_CC,
        check_interval: int = constants.DEFAULT_CHECK_INTERVAL,
        skip_regenerate: bool = False,
        skip_commit: bool = False,
        auto_restart: bool = False,
        no_monitor: bool = False,
        attach: bool = False,
        prompt_file: Optional[str] = None,
        config: Optional[str] = None,
        context_threshold: int = constants.DEFAULT_CONTEXT_THRESHOLD,
        idle_timeout: int = constants.DEFAULT_IDLE_TIMEOUT,
        max_errors: int = constants.DEFAULT_MAX_ERRORS,
        tmux_kill_on_exit: bool = True,
        tmux_mouse: bool = True,
        fast_start: bool = False,
        full_backup: bool = False,
        commit_every: Optional[int] = None,
    ):
        # Core configuration
        self.path = path
        self.agents = agents
        self.session = session
        self.settings = Settings()
        
        # Timing configuration
        self.stagger = stagger
        self.wait_after_cc = wait_after_cc
        self.check_interval = check_interval
        
        # Feature flags
        self.skip_regenerate = skip_regenerate
        self.skip_commit = skip_commit
        self.auto_restart = auto_restart
        self.no_monitor = no_monitor
        self.attach = attach
        self.fast_start = fast_start
        self.full_backup = full_backup
        
        # Agent configuration
        self.context_threshold = context_threshold
        self.idle_timeout = idle_timeout
        self.max_errors = max_errors
        
        # tmux configuration
        self.tmux_kill_on_exit = tmux_kill_on_exit
        self.tmux_mouse = tmux_mouse
        
        # Files and paths
        self.prompt_file = prompt_file
        self.commit_every = commit_every
        
        # Apply config file if provided
        if config:
            self.settings.load_from_file(config)
            # Override with settings from config
            for key, value in self.settings.config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        # Initialize paths
        self.project_path = Path(self.path).expanduser().resolve()
        self.combined_file = self.project_path / "combined_typechecker_and_linter_problems.txt"
        self.prompt_text = self._load_prompt()
        
        # Initialize integrations
        self.tmux = TmuxComposer(self.session)
        self.auto_resume = ClaudeAutoResume()
        self.hooks = ClaudeHooks()
        
        # State
        self.monitor: Optional[AgentMonitor] = None
        self.running = True
        self.shutting_down = False
        self.regeneration_cycles = 0
        self.run_start_time = datetime.now()
        self.total_problems_fixed = 0
        self.total_commits_made = 0
        self.agent_restart_count = 0
        
        # Signal handling
        self._last_sigint_time: Optional[float] = None
        self._force_kill_threshold = 3.0
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, sig: Any, frame: Any) -> None:
        """Handle shutdown signals gracefully with force-kill on double tap."""
        current_time = time.time()
        
        if sig == signal.SIGINT:
            if self._last_sigint_time and (current_time - self._last_sigint_time) < self._force_kill_threshold:
                # Force kill on double Ctrl-C
                console.print("\n[red]Force killing tmux session...[/red]")
                with contextlib.suppress(Exception):
                    run(f"tmux kill-session -t {self.session}", check=False, quiet=True)
                os._exit(1)
            else:
                self._last_sigint_time = current_time
                if not self.shutting_down:
                    self.shutting_down = True
                    console.print("\n[yellow]Shutting down gracefully... Press Ctrl-C again to force kill[/yellow]")
                    self.running = False
        else:
            if not self.shutting_down:
                self.shutting_down = True
                console.print("\n[yellow]Shutting down gracefully...[/yellow]")
                self.running = False
                
    def _load_prompt(self) -> str:
        """Load prompt text from file or default."""
        if self.prompt_file:
            prompt_path = Path(self.prompt_file)
            if prompt_path.exists():
                return prompt_path.read_text()
            else:
                console.print(f"[yellow]Warning: Prompt file not found: {self.prompt_file}[/yellow]")
                
        # Default prompt
        return """Please fix the problems in the combined_typechecker_and_linter_problems.txt file.
Focus on making the code work correctly while maintaining good code quality."""
        
    def setup_project(self) -> None:
        """Setup project environment including hooks and auto-resume."""
        console.print("[cyan]Setting up project environment...[/cyan]")
        
        # Create necessary directories
        (self.project_path / ".agent_logs").mkdir(exist_ok=True)
        (self.project_path / ".agent_metrics").mkdir(exist_ok=True)
        (self.project_path / ".claude_hooks").mkdir(exist_ok=True)
        (self.project_path / constants.HEARTBEATS_DIR).mkdir(exist_ok=True)
        
        # Install hooks if not present
        if not self.hooks.hooks_config_dir.exists():
            console.print("[yellow]Installing claude-code-generic-hooks...[/yellow]")
            if self.hooks.install_hooks():
                console.print("[green]✓ Hooks installed successfully[/green]")
            else:
                console.print("[yellow]⚠ Could not install hooks, continuing without them[/yellow]")
                
        # Test auto-resume installation
        if self.auto_resume.test_installation():
            console.print("[green]✓ claude-auto-resume is available[/green]")
        else:
            console.print("[red]✗ claude-auto-resume not found - API quota handling will not work![/red]")
            console.print("[yellow]Install from: https://github.com/terryso/claude-auto-resume[/yellow]")
            
    def regenerate_problems_file(self) -> int:
        """Regenerate the combined problems file.
        
        Returns:
            Number of problems found
        """
        if self.skip_regenerate and self.combined_file.exists():
            problems = line_count(self.combined_file)
            console.print(f"[yellow]Skipping regeneration. Using existing file with {problems} problems[/yellow]")
            return problems
            
        console.print("[cyan]Regenerating combined problems file...[/cyan]")
        
        # Run type checker and linter commands
        # This is a simplified version - in production you'd want to make this configurable
        commands = [
            "npm run typecheck || true",
            "npm run lint || true",
        ]
        
        problems_content = []
        for cmd in commands:
            try:
                _, stdout, stderr = run(cmd, capture=True, check=False)
                if stdout:
                    problems_content.append(stdout)
                if stderr:
                    problems_content.append(stderr)
            except Exception as e:
                console.print(f"[yellow]Warning: Command failed: {cmd} - {e}[/yellow]")
                
        # Write combined output
        self.combined_file.write_text("\n".join(problems_content))
        problems = line_count(self.combined_file)
        
        console.print(f"[green]✓ Found {problems} problems to fix[/green]")
        return problems
        
    def setup_tmux_session(self) -> None:
        """Create and configure tmux session with all agents."""
        console.print(f"[cyan]Setting up tmux session '{self.session}' with {self.agents} agents...[/cyan]")
        
        # Create tmux configuration
        config_file = self.tmux.create_config(self.agents, str(self.project_path))
        
        # Start session
        self.tmux.start_session()
        
        # Configure tmux options
        if self.tmux_mouse:
            run(f"tmux set-option -t {self.session} -g mouse on", quiet=True)
            
        # For many agents, optimize display
        if self.agents >= 10:
            run(f"tmux set-option -t {self.session} -g pane-border-status off", quiet=True)
            run(f"tmux set-option -t {self.session} -g status off", quiet=True)
            
        console.print(f"[green]✓ tmux session '{self.session}' created[/green]")
        
    def start_agents(self) -> None:
        """Start Claude Code agents in their panes."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        )
        
        with progress:
            task = progress.add_task(f"Starting {self.agents} agents...", total=self.agents)
            
            # Create wrapper script with auto-resume
            wrapper_path = self.project_path / ".claude_wrapper.sh"
            self.auto_resume.create_wrapper_script(wrapper_path)
            
            for i in range(self.agents):
                # Setup hooks for this agent
                hooks_config = self.hooks.setup_agent_hooks(i, self.project_path)
                
                # Send command to start Claude with auto-resume
                claude_cmd = f"cd {self.project_path} && CLAUDE_HOOKS_CONFIG={hooks_config} {wrapper_path}"
                self.tmux.send_to_pane(i, claude_cmd)
                
                progress.update(task, advance=1, description=f"Started agent {i + 1}/{self.agents}")
                
                # Stagger starts to avoid settings conflicts
                if i < self.agents - 1:
                    time.sleep(self.stagger)
                    
        console.print(f"[green]✓ All {self.agents} agents started[/green]")
        
    def wait_for_agents_ready(self) -> None:
        """Wait for all agents to be ready."""
        console.print(f"[cyan]Waiting {self.wait_after_cc}s for agents to initialize...[/cyan]")
        time.sleep(self.wait_after_cc)
        
    def send_prompt_to_agents(self) -> None:
        """Send the initial prompt to all agents."""
        console.print("[cyan]Sending prompt to all agents...[/cyan]")
        
        for i in range(self.agents):
            self.tmux.send_to_pane(i, self.prompt_text)
            # Small delay between sends
            time.sleep(0.1)
            
        console.print("[green]✓ Prompt sent to all agents[/green]")
        
    def run_monitoring_loop(self) -> None:
        """Run the main monitoring loop."""
        if self.no_monitor:
            console.print("[yellow]Monitoring disabled. Agents running in background.[/yellow]")
            console.print(f"[cyan]Attach with: tmux attach-session -t {self.session}[/cyan]")
            return
            
        # Initialize monitor
        self.monitor = AgentMonitor(
            session=self.session,
            num_agents=self.agents,
            tmux=self.tmux,
            context_threshold=self.context_threshold,
            idle_timeout=self.idle_timeout,
            max_errors=self.max_errors,
            project_path=self.project_path,
        )
        
        console.print("[green]Starting monitoring loop...[/green]")
        console.print("[dim]Press Ctrl-C to stop[/dim]")
        
        with Live(console=console, refresh_per_second=1) as live:
            while self.running:
                # Check all agents
                for i in range(self.agents):
                    self.monitor.check_agent(i)
                    
                # Display status
                live.update(self.monitor.get_status_table())
                
                # Check for agents needing restart
                if self.auto_restart:
                    for i in range(self.agents):
                        if self.monitor.needs_restart(i):
                            self.restart_agent(i)
                            
                # Periodic problem regeneration and commit
                if self.commit_every and self.regeneration_cycles % self.commit_every == 0:
                    self.regenerate_and_commit()
                    
                time.sleep(self.check_interval)
                
    def restart_agent(self, agent_id: int) -> None:
        """Restart a specific agent."""
        console.print(f"[yellow]Restarting agent {agent_id}...[/yellow]")
        
        # Send Ctrl-C to stop current session
        self.tmux.send_to_pane(agent_id, "\x03", enter=False)
        time.sleep(1)
        
        # Clear pane
        run(f"tmux clear-history -t {self.session}:{constants.TMUX_AGENTS_WINDOW}.{agent_id}", quiet=True)
        
        # Restart with same command
        wrapper_path = self.project_path / ".claude_wrapper.sh"
        hooks_config = self.project_path / f".claude_hooks/agent{agent_id:02d}_hooks.json"
        claude_cmd = f"cd {self.project_path} && CLAUDE_HOOKS_CONFIG={hooks_config} {wrapper_path}"
        self.tmux.send_to_pane(agent_id, claude_cmd)
        
        # Wait and send prompt
        time.sleep(self.wait_after_cc)
        self.tmux.send_to_pane(agent_id, self.prompt_text)
        
        # Update monitor state
        if self.monitor:
            self.monitor.agents[agent_id]["restart_count"] += 1
            self.monitor.agents[agent_id]["last_restart"] = datetime.now()
            self.monitor.agents[agent_id]["status"] = constants.STATUS_STARTING
            self.monitor.agents[agent_id]["errors"] = 0
            
        self.agent_restart_count += 1
        
    def regenerate_and_commit(self) -> None:
        """Regenerate problems file and commit if changes were made."""
        old_problems = line_count(self.combined_file) if self.combined_file.exists() else 0
        new_problems = self.regenerate_problems_file()
        
        if new_problems < old_problems:
            problems_fixed = old_problems - new_problems
            self.total_problems_fixed += problems_fixed
            
            if not self.skip_commit:
                try:
                    run("git add -A", check=True)
                    commit_msg = f"fix: Resolved {problems_fixed} type/lint issues via agent farm"
                    run(f'git commit -m "{commit_msg}"', check=True)
                    self.total_commits_made += 1
                    console.print(f"[green]✓ Committed fixes for {problems_fixed} problems[/green]")
                except subprocess.CalledProcessError:
                    console.print("[yellow]No changes to commit[/yellow]")
                    
        self.regeneration_cycles += 1
        
    def cleanup(self) -> None:
        """Clean up resources on shutdown."""
        console.print("[cyan]Cleaning up...[/cyan]")
        
        # Clean up tmux config
        if hasattr(self.tmux, 'cleanup'):
            self.tmux.cleanup()
            
        # Kill tmux session if configured
        if self.tmux_kill_on_exit:
            with contextlib.suppress(Exception):
                run(f"tmux kill-session -t {self.session}", check=False, quiet=True)
                
        # Print summary
        runtime = datetime.now() - self.run_start_time
        console.print(Panel(
            f"[green]Session Summary[/green]\n\n"
            f"Runtime: {runtime}\n"
            f"Problems fixed: {self.total_problems_fixed}\n"
            f"Commits made: {self.total_commits_made}\n"
            f"Agent restarts: {self.agent_restart_count}",
            title="Claude Agent Farm",
            box=box.ROUNDED,
        ))
        
    def run(self) -> int:
        """Run the agent farm.
        
        Returns:
            Exit code (0 for success)
        """
        try:
            # Setup
            self.setup_project()
            self.regenerate_problems_file()
            self.setup_tmux_session()
            
            # Start agents
            self.start_agents()
            self.wait_for_agents_ready()
            self.send_prompt_to_agents()
            
            # Monitor or attach
            if self.attach:
                console.print(f"[cyan]Attaching to tmux session '{self.session}'...[/cyan]")
                run(f"tmux attach-session -t {self.session}", check=False)
            else:
                self.run_monitoring_loop()
                
            return 0
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            return 130
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return 1
        finally:
            self.cleanup()