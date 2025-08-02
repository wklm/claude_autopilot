"""Flutter Firebase Agent monitoring for Claude Code.

Specialized monitoring for Flutter app development with Firebase backend,
including Firebase emulator integration and Flutter MCP documentation support.
"""

import signal
import subprocess
import time
from datetime import datetime

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from claude_code_agent_farm import constants
from claude_code_agent_farm.flutter_agent_settings import FlutterAgentSettings
from claude_code_agent_farm.models_new.checkpoint import (
    CheckpointManager,
    RecoveryStrategy,
)
from claude_code_agent_farm.models_new.commands import (
    CommandHistory,
)
from claude_code_agent_farm.models_new.events import (
    EventStore,
    StatusChangeEvent,
)
from claude_code_agent_farm.models_new.health_check import (
    AgentHealthMonitor,
    HealthCheckResult,
    HealthCheckType,
    HealthStatus,
    RestartAttemptTracker,
    RestartHealthCheck,
)
from claude_code_agent_farm.models_new.retry_strategy import (
    RetryStrategy,
    UsageLimitRetryInfo,
)
from claude_code_agent_farm.models_new.session import (
    AgentSession,
    AgentStatus,
    UsageLimitInfo,
)
from claude_code_agent_farm.models_new.watchdog import (
    ActivityType,
    HungAgentDetector,
    WatchdogTimer,
)
from claude_code_agent_farm.utils import UsageLimitTimeParser, run

console = Console(stderr=True)


class FlutterAgentMonitor:
    """Monitor and manage Claude agent for Flutter & Firebase development.

    Features:
    - Flutter project detection and configuration
    - Firebase emulator health monitoring
    - Flutter MCP documentation integration
    - Hot reload detection and handling
    """

    def __init__(self, settings: FlutterAgentSettings):
        self.settings = settings
        self.session = AgentSession(prompt=settings.prompt)
        self.time_parser = UsageLimitTimeParser()
        self.running = True
        self.shutting_down = False

        # Event and command tracking
        self.events = EventStore()
        self.command_history = CommandHistory()
        self.last_command_execution = None
        
        # Track last time agent transitioned to ready
        self.last_ready_time = None

        # Retry strategy for usage limits
        self.retry_strategy = RetryStrategy(
            initial_delay_seconds=60,
            max_delay_seconds=3600,
            backoff_factor=2.0,
            jitter_factor=0.1,
            max_retry_attempts=10,
        )
        self.current_usage_limit_info = None

        # Health monitoring
        self.health_monitor = AgentHealthMonitor(check_interval_seconds=30, unhealthy_threshold=3)
        self.restart_tracker = RestartAttemptTracker(max_attempts=5, cooldown_seconds=300, attempt_window_seconds=3600)

        # Watchdog timer for hung detection
        self.watchdog = WatchdogTimer(
            timeout_seconds=300,  # 5 minutes
            grace_period_seconds=60,  # 1 minute grace after restart
            check_interval_seconds=10,
        )
        self.hung_detector = HungAgentDetector()

        # Checkpoint management
        self.checkpoint_manager = CheckpointManager(
            checkpoint_interval_seconds=300,  # 5 minutes
        )
        self.recovery_strategy = RecoveryStrategy()

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig: int, frame: any) -> None:
        """Handle shutdown signals gracefully."""
        if not self.shutting_down:
            self.shutting_down = True
            console.print("\n[yellow]Shutting down gracefully...[/yellow]")
            self.running = False

    def setup_tmux_session(self) -> None:
        """Create or attach to tmux session."""
        # Kill existing session if it exists
        run(f"tmux kill-session -t {self.settings.tmux_session}", check=False, quiet=True)

        # Create new session
        run(f"tmux new-session -d -s {self.settings.tmux_session} -n agent", check=True)

        # Configure tmux
        run(f"tmux set-option -t {self.settings.tmux_session} -g mouse on", quiet=True)

        console.print(f"[green]✓ Created tmux session '{self.settings.tmux_session}'[/green]")

    def start_claude_agent(self) -> None:
        """Start Claude in the tmux session."""
        # Change to project directory
        run(f"tmux send-keys -t {self.settings.tmux_session}:agent 'cd {self.settings.project_path}'", check=True)
        run(f"tmux send-keys -t {self.settings.tmux_session}:agent C-m", check=True)

        # Start Claude with auto-resume if available
        claude_cmd = "claude-auto-resume" if self._has_auto_resume() else "claude"

        # Add flags
        claude_cmd += " --dangerously-skip-permissions"

        run(f"tmux send-keys -t {self.settings.tmux_session}:agent '{claude_cmd}'", check=True)
        run(f"tmux send-keys -t {self.settings.tmux_session}:agent C-m", check=True)

        console.print("[green]✓ Started Claude agent[/green]")

        # Start watchdog timer
        self.watchdog.start()

        # Wait for Claude to be ready
        time.sleep(10)

    def send_prompt(self) -> None:
        """Send the prompt to Claude."""
        # Escape special characters in prompt
        escaped_prompt = self.settings.prompt_text.replace("'", "'\"'\"'")
        
        # Append ultrathink to enable thinking mode
        enhanced_prompt = f"{escaped_prompt} ultrathink"

        # Send the text
        run(f"tmux send-keys -t {self.settings.tmux_session}:agent '{enhanced_prompt}'", check=True)
        
        # Small delay before Enter (like reference implementation)
        time.sleep(0.2)
        
        # Send Enter using C-m (carriage return)
        run(f"tmux send-keys -t {self.settings.tmux_session}:agent C-m", check=True)

        console.print("[green]✓ Sent prompt to Claude[/green]")
        self.session.total_runs += 1

    def capture_pane_content(self) -> str:
        """Capture the current tmux pane content."""
        try:
            result = subprocess.run(
                ["tmux", "capture-pane", "-t", f"{self.settings.tmux_session}:agent", "-p"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError:
            return ""

    def check_agent_status(self) -> AgentStatus:
        """Check the current status of the Claude agent."""
        content = self.capture_pane_content()

        # Feed watchdog if content changed
        if self.watchdog.is_content_changed(content):
            self.watchdog.feed(ActivityType.PANE_OUTPUT, "Pane content changed", content_length=len(content))

        # Check for stuck patterns
        is_stuck, stuck_reason = self.hung_detector.check_stuck_patterns(content)
        if is_stuck:
            console.print(f"[yellow]Warning: Agent may be stuck - {stuck_reason}[/yellow]")
            
        # Debug: log last few lines of content periodically
        if hasattr(self, '_last_debug_log') and (datetime.now() - self._last_debug_log).total_seconds() > 10:
            lines = content.strip().split('\n')
            if lines:
                console.print(f"[dim]Last line: ...{lines[-1][-80:] if lines[-1] else '(empty)'}[/dim]")
            self._last_debug_log = datetime.now()
        elif not hasattr(self, '_last_debug_log'):
            self._last_debug_log = datetime.now()

        # Check for usage limit
        for indicator in constants.USAGE_LIMIT_INDICATORS:
            if indicator.lower() in content.lower():
                # Parse the usage limit message
                self._handle_usage_limit(content)
                return AgentStatus.USAGE_LIMIT

        # Check for errors
        for indicator in constants.CLAUDE_ERROR_INDICATORS:
            if indicator in content:
                return AgentStatus.ERROR

        # Check if working - "esc to interrupt" is the key indicator
        if "esc to interrupt" in content:
            self.watchdog.feed(ActivityType.TASK_PROGRESS, "Agent working: esc to interrupt detected")
            return AgentStatus.WORKING
        
        # If not working (no "esc to interrupt") and no errors/limits, then Claude has finished
        # This is much more reliable than looking for specific UI patterns
        # Only log this occasionally to avoid spam
        if hasattr(self, '_last_ready_log') and (datetime.now() - self._last_ready_log).total_seconds() > 30:
            console.print("[dim]No 'esc to interrupt' found - agent is ready/finished[/dim]")
            self._last_ready_log = datetime.now()
        elif not hasattr(self, '_last_ready_log'):
            console.print("[dim]No 'esc to interrupt' found - agent is ready/finished[/dim]")
            self._last_ready_log = datetime.now()
        
        return AgentStatus.READY

    def _handle_usage_limit(self, content: str) -> None:
        """Handle usage limit detection and parsing with exponential backoff."""
        # Extract the usage limit message
        usage_lines = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            if any(indicator.lower() in line.lower() for indicator in constants.USAGE_LIMIT_INDICATORS):
                # Get this line and the next few for context
                usage_lines = lines[i : i + 3]
                break

        usage_message = "\n".join(usage_lines)

        # Parse retry time
        retry_time = self.time_parser.parse_usage_limit_message(usage_message)

        # Create enhanced usage limit info with retry strategy
        self.current_usage_limit_info = UsageLimitRetryInfo(message=usage_message, retry_strategy=self.retry_strategy)
        self.current_usage_limit_info.set_parsed_time(retry_time)

        # Record retry attempt
        self.retry_strategy.record_retry_attempt()

        # Get effective retry time
        effective_retry_time = self.current_usage_limit_info.effective_retry_time

        # Create legacy usage limit info for session
        limit_info = UsageLimitInfo(
            message=usage_message,
            retry_time=effective_retry_time,
            wait_duration=effective_retry_time - datetime.now() if effective_retry_time else None,
        )

        # Record in session
        self.session.record_usage_limit(limit_info)

        # Log the retry plan
        if self.current_usage_limit_info.use_fallback:
            console.print(
                f"[yellow]Usage limit hit. {self.current_usage_limit_info.fallback_reason}\n"
                f"Using exponential backoff (attempt {self.retry_strategy.current_attempt}). "
                f"Will retry at {self.time_parser.format_retry_time(effective_retry_time)}[/yellow]",
            )
        else:
            console.print(
                f"[yellow]Usage limit hit. Will retry at "
                f"{self.time_parser.format_retry_time(effective_retry_time)}[/yellow]",
            )

    def restart_agent(self) -> None:
        """Restart the Claude agent with health checks."""
        # Check if restart is allowed
        can_restart, reason = self.restart_tracker.can_restart()
        if not can_restart:
            console.print(f"[red]Cannot restart: {reason}[/red]")
            return

        console.print("[cyan]Restarting Claude agent...[/cyan]")
        
        # Reset ready time tracking
        self.last_ready_time = None

        # Create health check for this restart
        restart_health = RestartHealthCheck()
        restart_start = datetime.now()

        # Pre-restart health checks
        pre_checks = self._perform_pre_restart_checks()
        for check in pre_checks:
            restart_health.add_pre_check(check)

        if not restart_health.all_pre_checks_healthy:
            console.print("[yellow]Warning: Pre-restart checks indicate issues[/yellow]")

        try:
            # Send Ctrl+C to stop current session
            run(f"tmux send-keys -t {self.settings.tmux_session}:agent C-c", check=True)

            time.sleep(2)

            # Clear the pane
            run(f"tmux clear-history -t {self.settings.tmux_session}:agent", check=True)

            # Record restart
            self.session.record_restart()

            # Start agent again
            self.start_claude_agent()
            time.sleep(5)
            self.send_prompt()

            # Post-restart health checks
            time.sleep(3)  # Give agent time to initialize
            post_checks = self._perform_post_restart_checks()
            for check in post_checks:
                restart_health.add_post_check(check)

            # Mark restart completion
            restart_health.restart_timestamp = restart_start
            restart_health.restart_duration_ms = int((datetime.now() - restart_start).total_seconds() * 1000)
            restart_health.restart_successful = restart_health.all_post_checks_healthy

            if restart_health.restart_successful:
                console.print("[green]✓ Restart completed successfully[/green]")
            else:
                console.print("[yellow]⚠ Restart completed with warnings[/yellow]")

        except Exception as e:
            restart_health.restart_successful = False
            restart_health.failure_reason = str(e)
            console.print(f"[red]✗ Restart failed: {e}[/red]")
            raise
        finally:
            # Record the restart attempt
            self.restart_tracker.record_attempt(restart_health)

    def wait_for_usage_limit(self) -> None:
        """Wait until the usage limit period expires."""
        if not self.session.wait_until:
            return

        wait_seconds = (self.session.wait_until - datetime.now()).total_seconds()
        if wait_seconds <= 0:
            self.session.is_waiting_for_limit = False
            self.session.wait_until = None
            return

        console.print(f"[yellow]Waiting {int(wait_seconds)} seconds until usage limit expires...[/yellow]")

        # Wait with periodic status updates
        start_wait = time.time()
        while time.time() - start_wait < wait_seconds and self.running:
            remaining = int(wait_seconds - (time.time() - start_wait))

            # Update status every 60 seconds
            if remaining % 60 == 0 and remaining > 0:
                console.print(f"[dim]Still waiting... {remaining} seconds remaining[/dim]")

            time.sleep(1)

        self.session.is_waiting_for_limit = False
        self.session.wait_until = None

    def get_status_display(self) -> Table:
        """Get a status table for display."""
        table = Table(
            title=f"Claude Flutter Agent Monitor - {datetime.now().strftime('%H:%M:%S')}",
            box=box.ROUNDED,
        )

        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        # Add session info
        table.add_row("Session ID", self.session.session_id)
        table.add_row("Status", f"[{self._get_status_color()}]{self.session.status.value}[/]")
        table.add_row("Runtime", self.session.runtime)
        table.add_row("Total Runs", str(self.session.total_runs))
        table.add_row("Restarts", str(self.session.restart_count))
        table.add_row("Usage Limit Hits", str(self.session.usage_limit_hits))

        if self.session.is_waiting_for_limit and self.session.wait_until:
            wait_remaining = (self.session.wait_until - datetime.now()).total_seconds()
            if wait_remaining > 0:
                table.add_row("Waiting Until", self.time_parser.format_retry_time(self.session.wait_until))
                table.add_row("Wait Remaining", f"{int(wait_remaining)} seconds")

        table.add_row("Project", str(self.settings.project_path))

        return table

    def _get_status_color(self) -> str:
        """Get color for status display."""
        return {
            AgentStatus.WORKING: "green",
            AgentStatus.READY: "cyan",
            AgentStatus.IDLE: "yellow",
            AgentStatus.ERROR: "red",
            AgentStatus.USAGE_LIMIT: "bold red",
            AgentStatus.STARTING: "yellow",
            AgentStatus.UNKNOWN: "dim",
        }.get(self.session.status, "")

    def _has_auto_resume(self) -> bool:
        """Check if claude-auto-resume is available."""
        try:
            result = subprocess.run(["which", "claude-auto-resume"], capture_output=True, check=False)
            return result.returncode == 0
        except Exception:
            return False

    def _create_checkpoint(self) -> None:
        """Create a checkpoint of current session state."""
        try:
            # Gather checkpoint data
            checkpoint_data = {
                "session_id": self.session.session_id,
                "prompt": self.session.prompt,
                "status": self.session.status,
                "last_activity": self.session.last_activity,
                "total_runs": self.session.total_runs,
                "restart_count": self.session.restart_count,
                "usage_limit_hits": self.session.usage_limit_hits,
                "is_waiting_for_limit": self.session.is_waiting_for_limit,
                "wait_until": self.session.wait_until,
                "last_usage_limit": self.session.last_usage_limit.model_dump()
                if self.session.last_usage_limit
                else None,
                "retry_strategy_state": self.retry_strategy.model_dump(),
                "health_monitor_state": self.health_monitor.model_dump(),
                "restart_tracker_state": self.restart_tracker.model_dump(),
                "watchdog_state": self.watchdog.model_dump(),
                "events_summary": self.events.get_summary(),
                "last_pane_content": self.capture_pane_content()[:1000],  # First 1000 chars
                "metadata": {"settings": self.settings.model_dump(), "timestamp": datetime.now().isoformat()},
            }

            checkpoint = self.checkpoint_manager.create_checkpoint(checkpoint_data)
            console.print(f"[dim]Created checkpoint {checkpoint.checkpoint_id}[/dim]")

        except Exception as e:
            console.print(f"[yellow]Warning: Failed to create checkpoint: {e}[/yellow]")

    def _try_recover_from_checkpoint(self) -> bool:
        """Try to recover from the latest checkpoint."""
        try:
            checkpoint = self.checkpoint_manager.restore_latest_checkpoint(self.session.session_id)

            should_recover, reason = self.recovery_strategy.should_recover(checkpoint)
            if not should_recover:
                console.print(f"[yellow]Not recovering from checkpoint: {reason}[/yellow]")
                return False

            console.print(f"[cyan]Recovering from checkpoint {checkpoint.checkpoint_id}[/cyan]")

            # Restore session state
            if self.recovery_strategy.recover_state:
                self.session.total_runs = checkpoint.total_runs
                self.session.restart_count = checkpoint.restart_count
                self.session.usage_limit_hits = checkpoint.usage_limit_hits
                self.session.is_waiting_for_limit = checkpoint.is_waiting_for_limit
                self.session.wait_until = checkpoint.wait_until

            # Restore retry strategy
            if self.recovery_strategy.recover_retry_strategy and checkpoint.retry_strategy_state:
                self.retry_strategy = RetryStrategy(**checkpoint.retry_strategy_state)

            console.print("[green]✓ Successfully recovered from checkpoint[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Failed to recover from checkpoint: {e}[/red]")
            return False

    def _perform_pre_restart_checks(self) -> list[HealthCheckResult]:
        """Perform health checks before restart."""
        checks = []

        # Check tmux session exists
        try:
            result = subprocess.run(
                ["tmux", "has-session", "-t", self.settings.tmux_session], capture_output=True, check=False,
            )
            checks.append(
                HealthCheckResult(
                    check_type=HealthCheckType.TMUX_SESSION,
                    status=HealthStatus.HEALTHY if result.returncode == 0 else HealthStatus.UNHEALTHY,
                    message="Tmux session exists" if result.returncode == 0 else "Tmux session not found",
                ),
            )
        except Exception as e:
            checks.append(
                HealthCheckResult(
                    check_type=HealthCheckType.TMUX_SESSION,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Failed to check tmux session: {e}",
                ),
            )

        # Check disk space
        try:
            import shutil

            stat = shutil.disk_usage(self.settings.project_path)
            free_gb = stat.free / (1024**3)
            checks.append(
                HealthCheckResult(
                    check_type=HealthCheckType.DISK_SPACE,
                    status=HealthStatus.HEALTHY if free_gb > 1 else HealthStatus.DEGRADED,
                    message=f"Free disk space: {free_gb:.1f}GB",
                    details={"free_gb": free_gb},
                ),
            )
        except Exception as e:
            checks.append(
                HealthCheckResult(
                    check_type=HealthCheckType.DISK_SPACE,
                    status=HealthStatus.UNKNOWN,
                    message=f"Could not check disk space: {e}",
                ),
            )

        return checks

    def _perform_post_restart_checks(self) -> list[HealthCheckResult]:
        """Perform health checks after restart."""
        checks = []

        # Check if agent is responsive
        try:
            content = self.capture_pane_content()
            has_prompt = any(indicator in content for indicator in constants.CLAUDE_READY_INDICATORS)
            checks.append(
                HealthCheckResult(
                    check_type=HealthCheckType.AGENT_RESPONSIVE,
                    status=HealthStatus.HEALTHY if has_prompt else HealthStatus.UNHEALTHY,
                    message="Agent is responsive" if has_prompt else "Agent not showing ready prompt",
                ),
            )
        except Exception as e:
            checks.append(
                HealthCheckResult(
                    check_type=HealthCheckType.AGENT_RESPONSIVE,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Failed to check agent responsiveness: {e}",
                ),
            )

        # Check tmux pane is alive
        try:
            result = subprocess.run(
                ["tmux", "list-panes", "-t", f"{self.settings.tmux_session}:agent", "-F", "#{pane_dead}"],
                capture_output=True,
                text=True,
                check=False,
            )
            pane_alive = result.returncode == 0 and result.stdout.strip() == "0"
            checks.append(
                HealthCheckResult(
                    check_type=HealthCheckType.PROCESS_ALIVE,
                    status=HealthStatus.HEALTHY if pane_alive else HealthStatus.UNHEALTHY,
                    message="Tmux pane is alive" if pane_alive else "Tmux pane is dead",
                ),
            )
        except Exception as e:
            checks.append(
                HealthCheckResult(
                    check_type=HealthCheckType.PROCESS_ALIVE,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Failed to check tmux pane: {e}",
                ),
            )

        return checks

    def run(self) -> int:
        """Run the monitoring loop."""
        try:
            # Display startup info
            console.print(
                Panel(
                    f"[bold cyan]Claude Flutter Agent Monitor[/bold cyan]\n\n"
                    f"Project: {self.settings.project_path}\n"
                    f"Prompt: {self.settings.prompt_text[:100]}{'...' if len(self.settings.prompt_text) > 100 else ''}\n"
                    f"Wait on Limit: {self.settings.wait_on_limit}\n"
                    f"Restart on Complete: {self.settings.restart_on_complete}\n"
                    f"Idle Timeout: {self.settings.idle_timeout}s\n"
                    f"Check Interval: {self.settings.check_interval}s",
                    title="Starting Monitor",
                    box=box.ROUNDED,
                ),
            )

            # Setup tmux session
            self.setup_tmux_session()

            # Start initial agent
            self.start_claude_agent()
            time.sleep(5)
            self.send_prompt()

            # Main monitoring loop
            with Live(console=console, refresh_per_second=1) as live:
                last_status = AgentStatus.STARTING

                while self.running:
                    # Check if waiting for usage limit
                    if self.session.is_waiting_for_limit and self.settings.wait_on_limit:
                        live.update(self.get_status_display())
                        self.wait_for_usage_limit()
                        if self.running:  # If not shutdown during wait
                            self.restart_agent()
                        continue

                    # Check agent status
                    current_status = self.check_agent_status()
                    self.session.status = current_status

                    # Check watchdog timeout
                    timed_out, timeout_reason = self.watchdog.check_timeout()
                    if timed_out:
                        console.print(f"[red]Watchdog timeout: {timeout_reason}[/red]")
                        console.print("[yellow]Agent appears to be hung. Forcing restart...[/yellow]")
                        self.restart_agent()
                        self.watchdog.start()  # Reset watchdog after restart
                        continue

                    # Handle status changes
                    if current_status != last_status:
                        console.print(f"[dim]Status changed: {last_status.value} → {current_status.value}[/dim]")

                        # Feed watchdog on status change
                        self.watchdog.feed(
                            ActivityType.STATUS_CHANGE, f"Status: {last_status.value} → {current_status.value}",
                        )

                        # Record status change event
                        self.events.add(
                            StatusChangeEvent(
                                previous_status=last_status.value,
                                new_status=current_status.value,
                                source="monitor",
                            ),
                        )
                        
                        # Track when agent becomes ready (task completed)
                        if current_status == AgentStatus.READY and last_status == AgentStatus.WORKING:
                            self.last_ready_time = datetime.now()
                            console.print("[dim]Agent task completed, now ready[/dim]")

                        # Handle different statuses
                        if current_status == AgentStatus.ERROR:
                            if self.settings.restart_on_error:
                                console.print("[yellow]Error detected, restarting agent...[/yellow]")
                                self.restart_agent()

                        elif current_status == AgentStatus.USAGE_LIMIT and not self.settings.wait_on_limit:
                            console.print("[yellow]Usage limit hit but waiting disabled. Stopping.[/yellow]")
                            self.running = False
                    
                    # Handle READY state - ensure we track ready time and restart appropriately
                    if current_status == AgentStatus.READY:
                        # If we don't have a ready time, set it now
                        if self.last_ready_time is None:
                            self.last_ready_time = datetime.now()
                            console.print("[dim]Agent is ready (setting ready time)[/dim]")
                            
                            # If restart_on_complete is true, restart immediately
                            if self.settings.restart_on_complete:
                                console.print("[yellow]Agent ready, restarting due to restart_on_complete setting...[/yellow]")
                                self.restart_agent()
                                self.last_ready_time = None  # Reset
                                continue
                        
                        # If we already have a ready time, check for timeout
                        else:
                            idle_seconds = (datetime.now() - self.last_ready_time).total_seconds()
                            
                            # Log current idle time periodically
                            if int(idle_seconds) % 30 == 0:  # Log every 30 seconds
                                console.print(f"[dim]Agent idle for {int(idle_seconds)}s[/dim]")
                            
                            # If idle for too long, always restart
                            if idle_seconds > self.settings.idle_timeout:
                                console.print(f"[yellow]Agent idle for {int(idle_seconds)}s (>{self.settings.idle_timeout}s). Restarting...[/yellow]")
                                self.restart_agent()
                                self.last_ready_time = None  # Reset

                    # Reset ready time if agent is no longer ready
                    if current_status != AgentStatus.READY and self.last_ready_time is not None:
                        console.print(f"[dim]Agent no longer ready (now {current_status.value}), resetting ready time[/dim]")
                        self.last_ready_time = None
                    
                    last_status = current_status

                    # Create checkpoint if needed
                    if self.checkpoint_manager.should_checkpoint():
                        self._create_checkpoint()

                    # Update display
                    live.update(self.get_status_display())

                    # Wait before next check
                    time.sleep(self.settings.check_interval)

            return 0

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            return 130
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback

            traceback.print_exc()
            return 1
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources on shutdown."""
        console.print("[cyan]Cleaning up...[/cyan]")

        # Stop watchdog
        self.watchdog.stop()

        # Kill tmux session
        run(f"tmux kill-session -t {self.settings.tmux_session}", check=False, quiet=True)

        # Print summary
        console.print(
            Panel(
                f"[green]Session Summary[/green]\n\n"
                f"Runtime: {self.session.runtime}\n"
                f"Total Runs: {self.session.total_runs}\n"
                f"Restarts: {self.session.restart_count}\n"
                f"Usage Limit Hits: {self.session.usage_limit_hits}",
                title="Claude Flutter Agent Monitor",
                box=box.ROUNDED,
            ),
        )
