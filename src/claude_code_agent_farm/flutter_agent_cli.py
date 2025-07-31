"""Command-line interface for Claude Flutter Firebase Agent.

Specialized CLI for Flutter app development with Firebase backend,
including Firebase emulators and Flutter MCP documentation support.
"""

import os
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel

from claude_code_agent_farm import __version__
from claude_code_agent_farm.flutter_agent_monitor import FlutterAgentMonitor
from claude_code_agent_farm.flutter_agent_settings import (
    FlutterAgentSettings,
    load_settings,
)

app = typer.Typer(
    rich_markup_mode="rich",
    help="Claude agent for Flutter app development with Firebase backend - includes emulators & MCP docs",
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console(stderr=True)


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"Claude Flutter Firebase Agent v{__version__}")
        raise typer.Exit()


@app.command()
def run(
    prompt_file: Optional[Path] = typer.Option(
        None,
        "--prompt-file", "-p",
        help="Path to file containing the prompt",
        envvar="CLAUDE_PROMPT_FILE",
    ),
    prompt_text: Optional[str] = typer.Option(
        None,
        "--prompt-text", "-t",
        help="Direct prompt text",
        envvar="CLAUDE_PROMPT_TEXT",
    ),
    project_path: Optional[Path] = typer.Option(
        None,
        "--project-path", "--path",
        help="Path to the project directory",
        envvar="CLAUDE_PROJECT_PATH",
    ),
    wait_on_limit: Optional[bool] = typer.Option(
        None,
        "--wait-on-limit/--no-wait-on-limit",
        help="Wait when usage limit is hit",
        envvar="CLAUDE_WAIT_ON_LIMIT",
    ),
    restart_on_complete: Optional[bool] = typer.Option(
        None,
        "--restart-on-complete/--no-restart-on-complete",
        help="Restart when task completes",
        envvar="CLAUDE_RESTART_ON_COMPLETE",
    ),
    restart_on_error: Optional[bool] = typer.Option(
        None,
        "--restart-on-error/--no-restart-on-error",
        help="Restart when an error occurs",
        envvar="CLAUDE_RESTART_ON_ERROR",
    ),
    check_interval: Optional[int] = typer.Option(
        None,
        "--check-interval",
        help="Seconds between status checks",
        min=1,
        envvar="CLAUDE_CHECK_INTERVAL",
    ),
    idle_timeout: Optional[int] = typer.Option(
        None,
        "--idle-timeout",
        help="Seconds before considering agent idle",
        min=10,
        envvar="CLAUDE_IDLE_TIMEOUT",
    ),
    tmux_session: Optional[str] = typer.Option(
        None,
        "--tmux-session", "-s",
        help="tmux session name",
        envvar="CLAUDE_TMUX_SESSION",
    ),
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
        envvar="CLAUDE_LOG_LEVEL",
    ),
    version: bool = typer.Option(
        False,
        "--version", "-v",
        callback=version_callback,
        help="Show version and exit",
    ),
) -> None:
    """Run Claude Flutter Firebase Agent for Flutter app development.
    
    Features:
    - Flutter app development with hot reload support
    - Firebase emulators (Auth, Firestore, Functions, etc.)
    - Flutter MCP for real-time package documentation
    - Automatic pubspec.yaml detection
    - Firebase seed data management
    
    Configuration supports:
    1. Command-line arguments (highest priority)
    2. Environment variables (CLAUDE_ prefix)
    3. .env file in current directory
    4. Flutter project auto-detection
    """
    
    # Build settings kwargs from CLI arguments
    settings_kwargs = {}
    
    if prompt_file is not None:
        settings_kwargs["prompt_file"] = prompt_file
    if prompt_text is not None:
        settings_kwargs["prompt_text"] = prompt_text
    if project_path is not None:
        settings_kwargs["project_path"] = project_path
    if wait_on_limit is not None:
        settings_kwargs["wait_on_limit"] = wait_on_limit
    if restart_on_complete is not None:
        settings_kwargs["restart_on_complete"] = restart_on_complete
    if restart_on_error is not None:
        settings_kwargs["restart_on_error"] = restart_on_error
    if check_interval is not None:
        settings_kwargs["check_interval"] = check_interval
    if idle_timeout is not None:
        settings_kwargs["idle_timeout"] = idle_timeout
    if tmux_session is not None:
        settings_kwargs["tmux_session"] = tmux_session
    if log_level is not None:
        settings_kwargs["log_level"] = log_level
        
    # Create settings with CLI overrides
    try:
        if settings_kwargs:
            # Load base settings first, then override with CLI
            base_settings = load_settings()
            settings = FlutterAgentSettings(**{
                **base_settings.model_dump(),
                **settings_kwargs
            })
        else:
            # No CLI overrides, use settings as-is
            settings = load_settings()
    except Exception as e:
        # load_settings already handles error display
        raise typer.Exit(1)
        
    # Display welcome banner
    console.print(Panel(
        f"[bold cyan]Claude Flutter Firebase Agent v{__version__}[/bold cyan]\n\n"
        f"Flutter Project: {settings.project_path}\n"
        f"Session: {settings.tmux_session}\n"
        f"Prompt: {settings.prompt_display}\n"
        f"Firebase Emulators: {'Enabled' if settings.firebase_emulators_enabled else 'Disabled'}\n"
        f"Flutter MCP: {'Auto-detect' if settings.mcp_auto_detect else 'Manual'}",
        title="🦋 Starting Flutter Firebase Agent",
        box=box.ROUNDED,
    ))
    
    # Configure logging
    import logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Create and run monitor
    monitor = FlutterAgentMonitor(settings)
    exit_code = monitor.run()
    raise typer.Exit(exit_code)


@app.command()
def attach(
    session: str = typer.Option(
        "claude-agent",
        "--session", "-s",
        help="tmux session name to attach to",
    ),
) -> None:
    """Attach to a running agent session."""
    import subprocess
    
    try:
        subprocess.run(["tmux", "attach-session", "-t", session], check=True)
    except subprocess.CalledProcessError:
        console.print(f"[red]Failed to attach to session '{session}'[/red]")
        console.print("[yellow]Make sure the monitor is running[/yellow]")
        raise typer.Exit(1)


@app.command()
def stop(
    session: str = typer.Option(
        "claude-agent",
        "--session", "-s",
        help="tmux session name to stop",
    ),
) -> None:
    """Stop a running agent session."""
    import subprocess
    
    try:
        subprocess.run(["tmux", "kill-session", "-t", session], check=True)
        console.print(f"[green]✓ Stopped session '{session}'[/green]")
    except subprocess.CalledProcessError:
        console.print(f"[yellow]Session '{session}' not found or already stopped[/yellow]")


@app.command()
def show_config() -> None:
    """Show current configuration from all sources."""
    try:
        settings = load_settings()
        
        # Create config table
        from rich.table import Table
        
        table = Table(
            title="Current Configuration",
            box=box.ROUNDED,
            show_lines=True,
        )
        
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Source", style="dim")
        
        # Get field info with sources
        field_sources = settings.model_fields_set
        
        for field_name, field_info in settings.model_fields.items():
            value = getattr(settings, field_name)
            
            # Determine source
            if field_name in field_sources:
                # Check if it's from env by trying to get the env var
                env_key = f"CLAUDE_{field_name.upper()}"
                if os.getenv(env_key) is not None:
                    source = f"env: {env_key}"
                else:
                    source = "explicit"
            else:
                source = "default"
                
            # Format value for display
            if isinstance(value, Path):
                display_value = str(value)
            elif isinstance(value, list):
                display_value = ", ".join(str(v) for v in value[:3])
                if len(value) > 3:
                    display_value += "..."
            else:
                display_value = str(value)
                
            table.add_row(field_name, display_value, source)
            
        console.print(table)
        
        # Show environment info
        console.print("\n[dim]Environment:[/dim]")
        console.print(f"  Config file: {settings.model_config.get('env_file', '.env')}")
        console.print(f"  Prefix: {settings.model_config.get('env_prefix', 'CLAUDE_')}")
        
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    # When running inside Docker, automatically default to 'run' command
    if os.getenv("CLAUDE_SINGLE_AGENT_DOCKER"):
        import sys
        if len(sys.argv) == 1:
            sys.argv.append("run")
            
    app()