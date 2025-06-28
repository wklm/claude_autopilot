#!/usr/bin/env python3
"""
Claude Code Agent Farm - Hybrid Orchestrator
Combines simplicity with robust monitoring and automatic agent management
"""

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from random import randint
from typing import Dict, Optional, Tuple

try:
    import typer
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
except ImportError:
    print("Please install required libraries: pip install typer rich")
    sys.exit(1)

app = typer.Typer(
    help="Claude Code Agent Farm - Parallel code fixing automation",
    rich_markup_mode="rich"
)
console = Console()

# ─────────────────────────────── Configuration ────────────────────────────── #

DEFAULT_PROMPT = textwrap.dedent("""\
    I need you to start going through combined_typechecker_and_linter_problems.txt (just pick random chunks of 50 lines at a time from anywhere within the file, starting with a random starting line; since I have multiple agents working on this task, I want each agent to work on different problems!)

    As you select your chosen problems, mark them as such by prepending the line with [COMPLETED] so we can keep track of which ones have already been processed-- do this up front so there's no risk of forgetting to do it later and wasting time and compute on errors that are already being worked on or which were previously worked on. (Obviously, when selecting your random lines to work on, you should first filter out any rows that have "[COMPLETED]" in them so you don't accidentally work on already in-progress or completed tasks!)

    I want you to be SMART about fixing the problems. For example, if it's a type related problem, never try to use a stupid "band aid" fix and set the type to be Unknown or something dumb like that. If there's an unused variable or import, instead of just deleting it, figure out what we originally intended and whether that import or variable could be usefully and productively employed in the code to improve it so that it's no longer unused or unreferenced.

    Make all edits to the existing code files-- don't ever create a duplicative code file with the changes and give it some silly name; for instance, don't correct a problem in ComponentXYZ.tsx in a newly created file called ComponentXYZFixed.tsx or ComponentXYZNew.tsx-- always just revise ComponentXYZ.tsx in place!

    CRITICALLY IMPORTANT: You must adhere to ALL guidelines and advice in the NEXTJS15_BEST_PRACTICES.md document. I want to avoid technical debt and endless compatibility shims and workarounds and just fix things once and for all the RIGHT WAY. This code is still in development so we don't care at all about backwards compatibility. Note that we only use bun in this project, never npm. And you MUST check each proposed change against the @NEXT_BEST_PRACTICES_IMPLEMENTATION_PROGRESS.md guide!

    When you're done fixing the entire batch of selected problems, you can commit your progress to git with a detailed commit message (but don't go overboard making the commit message super long). Try to complete as much work as possible before coming back to me for more instructions-- what I've already asked you to do should keep you very busy for a while!
""")

# ─────────────────────────────── Helper Functions ─────────────────────────── #

def run(cmd: str, *, check: bool = True, quiet: bool = False, capture: bool = False) -> Tuple[int, str, str]:
    """Execute shell command with optional output capture"""
    if not quiet:
        console.log(cmd, style="cyan")

    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    else:
        result = subprocess.run(cmd, shell=True, check=check)
        return result.returncode, "", ""

def line_count(file_path: Path) -> int:
    """Count lines in a file"""
    try:
        with file_path.open("r", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def tmux_send(target: str, data: str, enter: bool = True) -> None:
    """Send keystrokes to a tmux pane"""
    escaped = data.replace("'", r"'\''")
    run(f"tmux send-keys -t {target} '{escaped}'", quiet=True)
    if enter:
        run(f"tmux send-keys -t {target} C-m", quiet=True)

def tmux_capture(target: str) -> str:
    """Capture content from a tmux pane"""
    _, stdout, _ = run(f"tmux capture-pane -t {target} -p", quiet=True, capture=True)
    return stdout

# ─────────────────────────────── Agent Monitor ────────────────────────────── #

class AgentMonitor:
    """Monitors Claude Code agents for health and performance"""

    def __init__(self, session: str, num_agents: int):
        self.session = session
        self.num_agents = num_agents
        self.agents: Dict[int, Dict] = {}
        self.running = True
        self.start_time = datetime.now()

        # Initialize agent tracking
        for i in range(num_agents):
            self.agents[i] = {
                'status': 'starting',
                'start_time': datetime.now(),
                'cycles': 0,
                'last_context': 100,
                'errors': 0,
                'last_activity': datetime.now()
            }

    def detect_context_percentage(self, content: str) -> Optional[int]:
        """Extract context percentage from pane content"""
        match = re.search(r'Context left until\s*auto-compact:\s*(\d+)%', content)
        return int(match.group(1)) if match else None

    def is_claude_ready(self, content: str) -> bool:
        """Check if Claude Code is ready for input"""
        return '│ >' in content and 'for shortcuts' in content

    def is_claude_working(self, content: str) -> bool:
        """Check if Claude Code is actively working"""
        indicators = ['✻ Pontificating', '● Bash(', '✻ Running', '✻ Thinking', 'esc to interrupt']
        return any(indicator in content for indicator in indicators)

    def has_settings_error(self, content: str) -> bool:
        """Check for settings corruption"""
        error_indicators = ['API key', 'Enter your API key', 'Configuration error', 'Settings corrupted']
        return any(indicator in content for indicator in error_indicators)

    def check_agent(self, agent_id: int) -> Dict:
        """Check status of a single agent"""
        pane_target = f"{self.session}:agents.{agent_id}"
        content = tmux_capture(pane_target)

        agent = self.agents[agent_id]

        # Update context percentage
        context = self.detect_context_percentage(content)
        if context is not None:
            agent['last_context'] = context

        # Check for errors
        if self.has_settings_error(content):
            agent['status'] = 'error'
            agent['errors'] += 1
            return agent

        # Update status based on activity
        if self.is_claude_working(content):
            agent['status'] = 'working'
            agent['last_activity'] = datetime.now()
        elif self.is_claude_ready(content):
            # Check if idle for too long
            idle_time = (datetime.now() - agent['last_activity']).total_seconds()
            if idle_time > 60:  # 1 minute idle
                agent['status'] = 'idle'
            else:
                agent['status'] = 'ready'
        else:
            agent['status'] = 'unknown'

        return agent

    def needs_restart(self, agent_id: int) -> bool:
        """Determine if an agent needs to be restarted"""
        agent = self.agents[agent_id]

        # Restart conditions
        return (
            agent['status'] == 'error' or
            agent['status'] == 'idle' or
            agent['last_context'] <= 20  # Context threshold
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
            runtime = str(datetime.now() - agent['start_time']).split('.')[0]

            status_style = {
                'working': '[green]',
                'ready': '[cyan]',
                'idle': '[yellow]',
                'error': '[red]',
                'starting': '[yellow]',
                'unknown': '[dim]'
            }.get(agent['status'], '')

            table.add_row(
                f"Pane {agent_id:02d}",
                f"{status_style}{agent['status']}[/]",
                str(agent['cycles']),
                f"{agent['last_context']}%",
                runtime,
                str(agent['errors'])
            )

        return table

# ─────────────────────────────── Main Orchestrator ────────────────────────── #

class ClaudeAgentFarm:
    def __init__(self,
                 path: str,
                 agents: int = 20,
                 session: str = "claude_agents",
                 stagger: float = 4.0,
                 wait_after_cc: float = 3.0,
                 check_interval: int = 10,
                 skip_regenerate: bool = False,
                 skip_commit: bool = False,
                 auto_restart: bool = False,
                 no_monitor: bool = False,
                 attach: bool = False,
                 prompt_file: Optional[str] = None,
                 config: Optional[str] = None):

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

        # Apply config file if provided
        if config:
            self._load_config(config)

        # Initialize other attributes
        self.project_path = Path(self.path).expanduser().resolve()
        self.combined_file = self.project_path / "combined_typechecker_and_linter_problems.txt"
        self.prompt_text = self._load_prompt()
        self.monitor = None
        self.running = True

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self, config_path: str):
        """Load settings from JSON config file"""
        config_file = Path(config_path)
        if config_file.exists():
            with config_file.open() as f:
                config_data = json.load(f)
                # Update instance attributes with config values
                for key, value in config_data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

    def _signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully"""
        console.print("\n[yellow]Received interrupt signal. Shutting down gracefully...[/yellow]")
        self.running = False

    def _load_prompt(self) -> str:
        """Load prompt from file or use default"""
        if self.prompt_file:
            return Path(self.prompt_file).read_text().strip()
        return DEFAULT_PROMPT.strip()

    def regenerate_problems(self):
        """Regenerate the type-checker and linter problems file"""
        if self.skip_regenerate:
            console.print("[yellow]Skipping problem file regeneration[/yellow]")
            return

        console.rule("[yellow]Regenerating type-checker + linter output")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Running type-check and lint...", total=None)

            os.chdir(self.project_path)
            tmpfile = self.combined_file.with_suffix(".tmp")

            with tmpfile.open("w") as out:
                out.write("$ bun run type-check\n")
                subprocess.run("bun run type-check", shell=True, stdout=out, stderr=subprocess.STDOUT)
                out.write("\n\n$ bun run lint\n")
                subprocess.run("bun run lint", shell=True, stdout=out, stderr=subprocess.STDOUT)

            tmpfile.replace(self.combined_file)
            progress.update(task, completed=True)

        count = line_count(self.combined_file)
        console.print(f"[green]✓ Generated {count} lines of problems[/green]")

    def commit_and_push(self):
        """Commit and push the updated problem count"""
        if self.skip_commit:
            console.print("[yellow]Skipping git commit/push[/yellow]")
            return

        console.rule("[yellow]Committing updated problem count")
        count = line_count(self.combined_file)

        try:
            run(f"git add {shlex.quote(str(self.combined_file))}")
            run(f"git commit -m 'Before next round of fixes; currently {count} lines of problems'", check=False)
            run("git push", check=False)
            console.print(f"[green]✓ Pushed commit with {count} current problems[/green]")
        except subprocess.CalledProcessError:
            console.print("[yellow]⚠ git commit/push skipped (no changes?)")

    def setup_tmux_session(self):
        """Create tmux session with tiled agent panes"""
        console.rule(f"[yellow]Creating tmux session '{self.session}' with {self.agents} agents")

        # Kill existing session if it exists
        run(f"tmux kill-session -t {self.session} 2>/dev/null", check=False, quiet=True)
        time.sleep(0.5)

        # Create new session with controller window
        run(f"tmux new-session -d -s {self.session} -n controller")
        run(f"tmux new-window -t {self.session} -n agents")

        # Create agent panes in tiled layout
        for i in range(self.agents):
            if i > 0:
                run(f"tmux split-window -t {self.session}:agents", quiet=True)
                run(f"tmux select-layout -t {self.session}:agents tiled", quiet=True)

        console.print(f"[green]✓ Created session with {self.agents} panes[/green]")

    def start_agent(self, agent_id: int, restart: bool = False):
        """Start or restart a single agent"""
        pane_target = f"{self.session}:agents.{agent_id}"

        if restart:
            # Exit current session
            tmux_send(pane_target, "/exit")
            time.sleep(1)
            self.monitor.agents[agent_id]['cycles'] += 1

        # Navigate and start Claude Code
        tmux_send(pane_target, f"cd {self.project_path}")
        tmux_send(pane_target, "cc")

        if not restart:
            console.print(f"🛠  Agent {agent_id:02d}: launching cc, waiting {self.wait_after_cc}s...")

        time.sleep(self.wait_after_cc)

        # Send prompt with unique seed for randomization
        seed = randint(100000, 999999)
        salted_prompt = self.prompt_text.replace(
            "random chunks",
            f"random chunks (instance-seed {seed})"
        )

        # Send prompt line by line
        for line in salted_prompt.splitlines():
            tmux_send(pane_target, line, enter=False)
            tmux_send(pane_target, "", enter=True)

        if not restart:
            console.print(f"[green]✓ Agent {agent_id:02d}: prompt injected[/green]")

        self.monitor.agents[agent_id]['status'] = 'starting'
        self.monitor.agents[agent_id]['last_activity'] = datetime.now()

    def launch_agents(self):
        """Launch all agents with staggered start times"""
        console.rule("[yellow]Launching agents")

        for i in range(self.agents):
            if not self.running:
                break

            self.start_agent(i)

            # Stagger starts to avoid config clobbering
            if i < self.agents - 1:
                time.sleep(self.stagger)

    def monitor_loop(self):
        """Main monitoring loop with auto-restart capability"""
        if self.no_monitor:
            console.print("[yellow]Monitoring disabled. Agents will run without supervision.[/yellow]")
            console.print(f"[cyan]Attach with: tmux attach -t {self.session}[/cyan]")
            return

        console.rule("[green]All agents launched - Monitoring active")
        console.print("[dim]Press Ctrl+C for graceful shutdown[/dim]\n")

        with Live(self.monitor.get_status_table(), refresh_per_second=1) as live:
            check_counter = 0

            while self.running:
                # Check agents every N seconds
                if check_counter % self.check_interval == 0:
                    for agent_id in range(self.agents):
                        self.monitor.check_agent(agent_id)

                        # Auto-restart if needed
                        if self.auto_restart and self.monitor.needs_restart(agent_id):
                            console.print(f"\n[yellow]Restarting agent {agent_id}...[/yellow]")
                            self.start_agent(agent_id, restart=True)

                live.update(self.monitor.get_status_table())
                time.sleep(1)
                check_counter += 1

    def run(self):
        """Main orchestration flow"""
        os.chdir(self.project_path)

        # Display startup banner
        console.print(Panel.fit(
            f"[bold cyan]Claude Code Agent Farm[/bold cyan]\n"
            f"Project: {self.project_path}\n"
            f"Agents: {self.agents}\n"
            f"Session: {self.session}\n"
            f"Auto-restart: {'enabled' if self.auto_restart else 'disabled'}",
            border_style="cyan"
        ))

        # Execute workflow steps
        self.regenerate_problems()
        self.commit_and_push()
        self.setup_tmux_session()

        # Initialize monitor
        self.monitor = AgentMonitor(self.session, self.agents)

        # Launch agents
        self.launch_agents()

        # Start monitoring
        self.monitor_loop()

        # Attach to session if requested
        if self.attach and not self.no_monitor:
            run(f"tmux attach-session -t {self.session}", check=False)

    def shutdown(self):
        """Clean shutdown of all agents"""
        if not self.monitor:
            return

        console.print("\n[yellow]Shutting down agents...[/yellow]")

        for i in range(self.agents):
            pane_target = f"{self.session}:agents.{i}"
            tmux_send(pane_target, "/exit")

        time.sleep(2)
        run(f"tmux kill-session -t {self.session}", check=False, quiet=True)
        console.print("[green]✓ Shutdown complete[/green]")

# ─────────────────────────────── CLI Entry Point ──────────────────────────── #

@app.command()
def main(
    path: str = typer.Option(
        ...,
        "--path",
        help="Absolute path to project root",
        rich_help_panel="Required Arguments"
    ),
    agents: int = typer.Option(
        20,
        "--agents", "-n",
        help="Number of Claude agents",
        rich_help_panel="Agent Configuration"
    ),
    session: str = typer.Option(
        "claude_agents",
        "--session", "-s",
        help="tmux session name",
        rich_help_panel="Agent Configuration"
    ),
    stagger: float = typer.Option(
        4.0,
        "--stagger",
        help="Seconds between starting agents",
        rich_help_panel="Timing Configuration"
    ),
    wait_after_cc: float = typer.Option(
        3.0,
        "--wait-after-cc",
        help="Seconds to wait after launching cc",
        rich_help_panel="Timing Configuration"
    ),
    check_interval: int = typer.Option(
        10,
        "--check-interval",
        help="Seconds between agent health checks",
        rich_help_panel="Timing Configuration"
    ),
    skip_regenerate: bool = typer.Option(
        False,
        "--skip-regenerate",
        help="Skip regenerating problems file",
        rich_help_panel="Feature Flags"
    ),
    skip_commit: bool = typer.Option(
        False,
        "--skip-commit",
        help="Skip git commit/push",
        rich_help_panel="Feature Flags"
    ),
    auto_restart: bool = typer.Option(
        False,
        "--auto-restart",
        help="Auto-restart agents on errors/completion",
        rich_help_panel="Feature Flags"
    ),
    no_monitor: bool = typer.Option(
        False,
        "--no-monitor",
        help="Disable monitoring (just launch and exit)",
        rich_help_panel="Feature Flags"
    ),
    attach: bool = typer.Option(
        False,
        "--attach",
        help="Attach to tmux session after setup",
        rich_help_panel="Feature Flags"
    ),
    prompt_file: Optional[str] = typer.Option(
        None,
        "--prompt-file",
        help="Path to custom prompt file",
        rich_help_panel="Advanced Options"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Load settings from JSON config file",
        rich_help_panel="Advanced Options"
    )
):
    """
    Claude Code Agent Farm - Parallel code fixing automation

    This tool orchestrates multiple Claude Code agents working in parallel
    to fix type-checker and linter problems in your codebase.
    """

    # Validate project path
    project_path = Path(path).expanduser().resolve()
    if not project_path.is_dir():
        console.print(f"[red]✖ {project_path} is not a directory[/red]")
        raise typer.Exit(1)

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
        config=config
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

if __name__ == "__main__":
    app()
