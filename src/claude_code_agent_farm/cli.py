"""Command-line interface for Claude Code Agent Farm."""

from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel

from claude_code_agent_farm import __version__
from claude_code_agent_farm.config import constants
from claude_code_agent_farm.core.orchestrator import ClaudeAgentFarm

app = typer.Typer(
    rich_markup_mode="rich",
    help="Orchestrate multiple Claude Code agents for parallel work",
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console(stderr=True)


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"Claude Code Agent Farm v{__version__}")
        raise typer.Exit()


@app.command()
def run(
    path: str = typer.Argument(..., help="Path to the project to fix"),
    agents: int = typer.Option(
        constants.DEFAULT_NUM_AGENTS,
        "--agents", "-a",
        help="Number of Claude Code agents to spawn",
        min=1,
        max=100,
    ),
    session: str = typer.Option(
        constants.DEFAULT_SESSION_NAME,
        "--session", "-s",
        help="tmux session name",
    ),
    stagger: float = typer.Option(
        constants.DEFAULT_STAGGER_TIME,
        "--stagger",
        help="Seconds to wait between starting each agent",
        min=0.0,
    ),
    wait_after_cc: float = typer.Option(
        constants.DEFAULT_WAIT_AFTER_CC,
        "--wait-after-cc",
        help="Seconds to wait after starting Claude Code before sending prompt",
        min=0.0,
    ),
    check_interval: int = typer.Option(
        constants.DEFAULT_CHECK_INTERVAL,
        "--check-interval",
        help="Seconds between status checks",
        min=1,
    ),
    skip_regenerate: bool = typer.Option(
        False,
        "--skip-regenerate",
        help="Skip regenerating the problems file",
    ),
    skip_commit: bool = typer.Option(
        False,
        "--skip-commit",
        help="Skip git commits",
    ),
    auto_restart: bool = typer.Option(
        False,
        "--auto-restart",
        help="Automatically restart agents that fail or get stuck",
    ),
    no_monitor: bool = typer.Option(
        False,
        "--no-monitor",
        help="Start agents without monitoring dashboard",
    ),
    attach: bool = typer.Option(
        False,
        "--attach",
        help="Attach to tmux session after starting agents",
    ),
    prompt_file: Optional[str] = typer.Option(
        None,
        "--prompt-file", "-p",
        help="Path to file containing the prompt for agents",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to JSON configuration file",
    ),
    context_threshold: int = typer.Option(
        constants.DEFAULT_CONTEXT_THRESHOLD,
        "--context-threshold",
        help="Context percentage threshold for warnings/restarts",
        min=1,
        max=100,
    ),
    idle_timeout: int = typer.Option(
        constants.DEFAULT_IDLE_TIMEOUT,
        "--idle-timeout",
        help="Seconds before marking an agent as idle",
        min=10,
    ),
    max_errors: int = typer.Option(
        constants.DEFAULT_MAX_ERRORS,
        "--max-errors",
        help="Maximum errors before restarting an agent",
        min=1,
    ),
    tmux_kill_on_exit: bool = typer.Option(
        True,
        "--tmux-kill-on-exit/--no-tmux-kill-on-exit",
        help="Kill tmux session on exit",
    ),
    tmux_mouse: bool = typer.Option(
        True,
        "--tmux-mouse/--no-tmux-mouse",
        help="Enable mouse support in tmux",
    ),
    fast_start: bool = typer.Option(
        False,
        "--fast-start",
        help="Skip some safety checks for faster startup",
    ),
    full_backup: bool = typer.Option(
        False,
        "--full-backup",
        help="Create full backup of Claude settings (including caches)",
    ),
    commit_every: Optional[int] = typer.Option(
        None,
        "--commit-every",
        help="Commit changes every N regeneration cycles",
        min=1,
    ),
    version: bool = typer.Option(
        False,
        "--version", "-v",
        callback=version_callback,
        help="Show version and exit",
    ),
) -> None:
    """Run Claude Code Agent Farm to fix problems in parallel."""
    # Display welcome banner
    console.print(Panel(
        f"[bold cyan]Claude Code Agent Farm v{__version__}[/bold cyan]\n\n"
        f"Project: {path}\n"
        f"Agents: {agents}\n"
        f"Session: {session}",
        title="🤖 Starting Agent Farm",
        box=box.ROUNDED,
    ))
    
    # Create and run orchestrator
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
        commit_every=commit_every,
    )
    
    exit_code = farm.run()
    raise typer.Exit(exit_code)


@app.command()
def stop(
    session: str = typer.Option(
        constants.DEFAULT_SESSION_NAME,
        "--session", "-s",
        help="tmux session name to stop",
    ),
) -> None:
    """Stop a running agent farm session."""
    from claude_code_agent_farm.utils import run
    
    try:
        run(f"tmux kill-session -t {session}", check=True)
        console.print(f"[green]✓ Stopped session '{session}'[/green]")
    except Exception:
        console.print(f"[yellow]Session '{session}' not found or already stopped[/yellow]")


@app.command()
def list() -> None:
    """List active agent farm sessions."""
    from claude_code_agent_farm.utils import run
    
    try:
        _, stdout, _ = run("tmux list-sessions", capture=True, check=True)
        sessions = []
        for line in stdout.strip().split("\n"):
            if line:
                session_name = line.split(":")[0]
                sessions.append(session_name)
                
        if sessions:
            console.print("[cyan]Active sessions:[/cyan]")
            for session in sessions:
                console.print(f"  • {session}")
        else:
            console.print("[yellow]No active sessions[/yellow]")
    except Exception:
        console.print("[yellow]No tmux server running[/yellow]")


@app.command()
def attach(
    session: str = typer.Option(
        constants.DEFAULT_SESSION_NAME,
        "--session", "-s",
        help="tmux session name to attach to",
    ),
) -> None:
    """Attach to a running agent farm session."""
    from claude_code_agent_farm.utils import run
    
    try:
        run(f"tmux attach-session -t {session}", check=True)
    except Exception:
        console.print(f"[red]Failed to attach to session '{session}'[/red]")
        console.print("[yellow]Use 'claude-code-agent-farm list' to see active sessions[/yellow]")


if __name__ == "__main__":
    app()