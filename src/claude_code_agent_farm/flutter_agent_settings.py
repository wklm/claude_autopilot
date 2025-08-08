"""Pydantic Settings for Claude Flutter Firebase Agent configuration.

Provides configuration specifically for Flutter & Firebase development
with the carenji healthcare app.
"""

from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FlutterAgentSettings(BaseSettings):
    """Settings for Flutter Firebase Agent with carenji-specific configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CLAUDE_",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "project_path": "/workspace",
                "prompt_text": "Fix all type errors and lint issues",
                "wait_on_limit": True,
                "restart_on_complete": True,
                "check_interval": 5,
                "idle_timeout": 300,
                "tmux_session": "claude-agent",
            },
        },
    )

    # Project configuration
    project_path: Path = Field(
        default=Path("/home/wojtek/dev/carenji"), description="Path to the carenji Flutter project directory",
    )

    # Prompt configuration
    prompt_file: Path | None = Field(default=None, description="Path to file containing the prompt")

    prompt_text: str | None = Field(default=None, description="Direct prompt text")

    # Behavior configuration
    wait_on_limit: bool = Field(default=True, description="Wait when usage limit is hit")

    restart_on_complete: bool = Field(default=True, description="Restart when task completes")

    restart_on_error: bool = Field(default=True, description="Restart when an error occurs")

    # Monitoring configuration
    check_interval: int = Field(default=5, ge=1, le=300, description="Seconds between status checks")

    idle_timeout: int = Field(default=300, ge=10, le=3600, description="Seconds before considering agent idle")

    # tmux configuration
    tmux_session: str = Field(
        default="claude-carenji", description="tmux session name for carenji development", pattern=r"^[a-zA-Z0-9_-]+$",
    )

    # Advanced configuration
    auto_resume_enabled: bool = Field(default=True, description="Use claude-auto-resume if available")

    log_level: str = Field(
        default="INFO", description="Logging level", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )

    status_update_interval: int = Field(default=1, ge=1, le=10, description="Seconds between status display updates")

    max_prompt_display_length: int = Field(
        default=100, ge=50, le=500, description="Maximum characters to display from prompt",
    )

    # Usage limit configuration
    default_wait_hours: int = Field(
        default=1, ge=1, le=24, description="Hours to wait if usage limit time cannot be parsed",
    )

    usage_limit_check_patterns: list[str] = Field(
        default=[
            "Claude usage limit reached",
            "usage limit reached",
            "daily limit exceeded",
            "usage quota exceeded",
            "try again later",
            "rate limit exceeded",
        ],
        description="Patterns to detect usage limits",
    )

    # Claude configuration paths
    claude_config_paths: list[Path] = Field(
        default=[
            Path("/home/claude/.config/claude/.claude.json"),
            Path("/home/claude/.claude.json"),
            Path("~/.config/claude/.claude.json").expanduser(),
            Path("~/.claude.json").expanduser(),
        ],
        description="Paths to check for Claude configuration",
    )

    # Firebase configuration for carenji
    firebase_project_id: str = Field(default="carenji-24ab8", description="Firebase project ID for carenji")

    firebase_emulators_enabled: bool = Field(
        default=True, description="Enable Firebase emulators for local development",
    )

    firebase_emulator_host: str = Field(default="127.0.0.1", description="Firebase emulator host")

    firebase_emulator_ports: dict[str, int] = Field(
        default={
            "auth": 9098,
            "firestore": 8079,
            "functions": 5001,
            "ui": 4001,
        },
        description="Firebase emulator ports matching carenji's firebase.json",
    )

    # Flutter MCP configuration
    mcp_enabled: bool = Field(default=True, description="Enable Flutter MCP for AI assistance")

    mcp_vmservice_port: int = Field(default=8182, description="Flutter VM service port for MCP")

    mcp_dds_port: int = Field(default=8181, description="Flutter DDS port for MCP")

    mcp_auto_detect: bool = Field(default=True, description="Auto-detect Flutter MCP availability")

    # Carenji-specific features
    carenji_features_enabled: list[str] = Field(
        default=[
            "medication_management",
            "vitals_monitoring",
            "staff_scheduling",
            "family_portal",
            "barcode_scanning",
        ],
        description="Enabled carenji features to monitor",
    )

    carenji_test_coverage_threshold: int = Field(
        default=80, ge=0, le=100, description="Minimum test coverage percentage for carenji",
    )

    carenji_lint_on_save: bool = Field(default=True, description="Run Flutter analyzer on file save")

    @field_validator("project_path")
    @classmethod
    def validate_project_path(cls, v: Path) -> Path:
        """Ensure project path exists and is a directory."""
        path = v.expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Project path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Project path is not a directory: {path}")
        return path

    @field_validator("prompt_file")
    @classmethod
    def validate_prompt_file(cls, v: Path | None) -> Path | None:
        """Validate prompt file if provided."""
        if v is None:
            return None
        path = v.expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Prompt file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Prompt file is not a file: {path}")
        return path

    @model_validator(mode="after")
    def validate_prompt_configuration(self) -> "FlutterAgentSettings":
        """Ensure we have a prompt configured."""
        # Check if we have a prompt
        if not self.prompt_file and not self.prompt_text:
            # Check for default prompt.txt in project
            default_prompt = self.project_path / "prompt.txt"
            if default_prompt.exists():
                self.prompt_file = default_prompt
            else:
                raise ValueError(
                    "No prompt configured. Provide either prompt_file, prompt_text, "
                    "or create prompt.txt in the project directory",
                )
        return self

    @property
    def prompt(self) -> str:
        """Get the actual prompt text with carenji-specific enhancements."""
        base_prompt = ""
        if self.prompt_text:
            base_prompt = self.prompt_text.strip()
        elif self.prompt_file:
            base_prompt = self.prompt_file.read_text().strip()
        else:
            raise ValueError("No prompt available")

        # Add carenji context if not already present
        if "carenji" not in base_prompt.lower() and "CLAUDE.md" not in base_prompt:
            return f"{base_prompt}\n\nRemember to follow carenji's CLAUDE.md guidelines for Flutter & Firebase development."
        return base_prompt

    @property
    def prompt_display(self) -> str:
        """Get truncated prompt for display."""
        prompt = self.prompt
        if len(prompt) > self.max_prompt_display_length:
            return f"{prompt[: self.max_prompt_display_length]}..."
        return prompt

    @property
    def has_claude_config(self) -> bool:
        """Check if any Claude configuration exists."""
        return any(path.exists() for path in self.claude_config_paths)

    def get_claude_config_path(self) -> Path | None:
        """Get the first existing Claude configuration path."""
        for path in self.claude_config_paths:
            if path.exists():
                return path
        return None

    def to_flutter_agent_config(self) -> dict:
        """Convert to FlutterAgentConfig compatible dict."""
        return {
            "project_path": self.project_path,
            "prompt_text": self.prompt,
            "wait_on_limit": self.wait_on_limit,
            "restart_on_complete": self.restart_on_complete,
            "restart_on_error": self.restart_on_error,
            "check_interval": self.check_interval,
            "idle_timeout": self.idle_timeout,
            "tmux_session": self.tmux_session,
        }


def load_settings() -> FlutterAgentSettings:
    """Load settings with proper error handling for carenji development."""
    try:
        settings = FlutterAgentSettings()

        # Validate carenji project if path exists
        if settings.project_path.exists():
            from claude_code_agent_farm.utils import check_carenji_project

            if not check_carenji_project(settings.project_path):
                from rich.console import Console

                console = Console(stderr=True)
                console.print(
                    f"[yellow]Warning: {settings.project_path} doesn't appear to be the carenji project[/yellow]",
                )

        return settings
    except Exception as e:
        # Provide helpful error messages
        import sys

        from rich.console import Console

        console = Console(stderr=True)
        console.print(f"[red]Configuration Error:[/red] {e}")

        if "prompt" in str(e).lower():
            console.print("\n[yellow]Hint:[/yellow] Provide a prompt using one of:")
            console.print("  - CLAUDE_PROMPT_TEXT environment variable")
            console.print("  - CLAUDE_PROMPT_FILE environment variable")
            console.print("  - Create prompt.txt in your project directory")
            console.print("\n[dim]Example prompts for carenji:[/dim]")
            console.print('  - "Fix all Flutter analyzer errors following carenji standards"')
            console.print('  - "Implement medication tracking feature with tests"')
            console.print('  - "Review and optimize vitals monitoring performance"')

        sys.exit(1)
