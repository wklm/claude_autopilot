"""Command-related Pydantic models for Claude Single Agent Monitor."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, computed_field

from claude_code_agent_farm.models.base import CommandModel, TimestampedModel


class CommandType(str, Enum):
    """Types of commands."""
    
    TMUX = "tmux"
    CLAUDE = "claude"
    SHELL = "shell"
    SYSTEM = "system"


class CommandStatus(str, Enum):
    """Command execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TmuxCommand(CommandModel):
    """tmux-specific command model."""
    
    command_type: CommandType = Field(default=CommandType.TMUX, const=True)
    session: str = Field(..., description="tmux session name")
    window: Optional[str] = Field(default=None, description="tmux window name")
    pane: Optional[int] = Field(default=None, ge=0, description="tmux pane index")
    
    # tmux-specific options
    send_keys: bool = Field(default=False, description="Use send-keys instead of command")
    enter: bool = Field(default=True, description="Send Enter after command")
    clear_first: bool = Field(default=False, description="Clear pane before command")
    
    @field_validator("session")
    @classmethod
    def validate_session_name(cls, v: str) -> str:
        """Validate tmux session name."""
        if not v or not v.strip():
            raise ValueError("Session name cannot be empty")
        # tmux session names should be alphanumeric with - and _
        if not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError("Session name can only contain letters, numbers, hyphens, and underscores")
        return v
    
    @computed_field
    @property
    def target(self) -> str:
        """Get the tmux target specification."""
        target = self.session
        if self.window:
            target = f"{target}:{self.window}"
        if self.pane is not None:
            target = f"{target}.{self.pane}"
        return target
    
    def to_shell_command(self) -> list[str]:
        """Convert to shell command arguments."""
        if self.send_keys:
            cmd = ["tmux", "send-keys", "-t", self.target]
            cmd.append(self.command)
            if self.enter:
                cmd.append("Enter")
            return cmd
        else:
            return ["tmux", self.command, "-t", self.target] + self.args


class ClaudeCommand(CommandModel):
    """Claude CLI-specific command model."""
    
    command_type: CommandType = Field(default=CommandType.CLAUDE, const=True)
    command: str = Field(default="claude", const=True)
    
    # Claude-specific options
    use_auto_resume: bool = Field(default=True, description="Use claude-auto-resume if available")
    skip_permissions: bool = Field(default=True, description="Skip permissions check")
    enable_background: bool = Field(default=True, description="Enable background tasks")
    
    # Prompt handling
    prompt: Optional[str] = Field(default=None, description="Prompt to send after startup")
    wait_before_prompt: float = Field(default=10.0, ge=0, description="Seconds to wait before sending prompt")
    
    @computed_field
    @property
    def executable(self) -> str:
        """Get the actual executable to use."""
        if self.use_auto_resume:
            return "claude-auto-resume"
        return "claude"
    
    def to_shell_command(self) -> list[str]:
        """Convert to shell command arguments."""
        cmd = [self.executable]
        
        if self.skip_permissions:
            cmd.append("--dangerously-skip-permissions")
            
        cmd.extend(self.args)
        return cmd
    
    def get_env(self) -> dict[str, str]:
        """Get environment variables for Claude."""
        env = self.env.copy()
        
        if self.enable_background:
            env["ENABLE_BACKGROUND_TASKS"] = "1"
        if self.skip_permissions:
            env["CLAUDE_DANGEROUSLY_SKIP_PERMISSIONS"] = "1"
            
        return env


class ShellCommand(CommandModel):
    """General shell command model."""
    
    command_type: CommandType = Field(default=CommandType.SHELL, const=True)
    shell: str = Field(default="/bin/bash", description="Shell to use")
    use_shell: bool = Field(default=True, description="Execute through shell")
    capture_output: bool = Field(default=True, description="Capture command output")
    
    def to_subprocess_args(self) -> dict:
        """Get arguments for subprocess.run()."""
        args = {
            "capture_output": self.capture_output,
            "text": True,
            "env": self.env or None,
            "cwd": self.working_dir,
            "timeout": self.timeout,
        }
        
        if self.use_shell:
            args["shell"] = True
            args["executable"] = self.shell
            
        return args


class CommandExecution(TimestampedModel):
    """Record of a command execution."""
    
    command: CommandModel = Field(..., description="The command that was executed")
    status: CommandStatus = Field(default=CommandStatus.PENDING)
    
    # Execution details
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    duration_seconds: Optional[float] = Field(default=None, ge=0)
    
    # Results
    exit_code: Optional[int] = Field(default=None)
    stdout: Optional[str] = Field(default=None)
    stderr: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    
    # Metadata
    execution_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    retry_count: int = Field(default=0, ge=0)
    
    def start(self) -> None:
        """Mark command as started."""
        self.status = CommandStatus.RUNNING
        self.started_at = datetime.now()
        self.touch()
    
    def complete(self, exit_code: int, stdout: str = "", stderr: str = "") -> None:
        """Mark command as completed."""
        self.status = CommandStatus.SUCCESS if exit_code == 0 else CommandStatus.FAILED
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.completed_at = datetime.now()
        
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
            
        self.touch()
    
    def fail(self, error: str) -> None:
        """Mark command as failed with error."""
        self.status = CommandStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.now()
        
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
            
        self.touch()
    
    def timeout(self) -> None:
        """Mark command as timed out."""
        self.status = CommandStatus.TIMEOUT
        self.error_message = f"Command timed out after {self.command.timeout} seconds"
        self.completed_at = datetime.now()
        
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
            
        self.touch()
    
    @computed_field
    @property
    def is_success(self) -> bool:
        """Check if command succeeded."""
        return self.status == CommandStatus.SUCCESS
    
    @computed_field
    @property
    def is_complete(self) -> bool:
        """Check if command is complete (success or failure)."""
        return self.status in (
            CommandStatus.SUCCESS,
            CommandStatus.FAILED,
            CommandStatus.TIMEOUT,
            CommandStatus.CANCELLED,
        )


class CommandHistory(BaseModel):
    """Track command execution history."""
    
    executions: list[CommandExecution] = Field(default_factory=list)
    max_history: int = Field(default=100, ge=1, description="Maximum executions to keep")
    
    def add(self, execution: CommandExecution) -> None:
        """Add an execution to history."""
        self.executions.append(execution)
        
        # Trim to max size
        if len(self.executions) > self.max_history:
            self.executions = self.executions[-self.max_history:]
    
    def get_recent(self, count: int = 10) -> list[CommandExecution]:
        """Get recent executions."""
        return self.executions[-count:]
    
    def get_by_type(self, command_type: CommandType) -> list[CommandExecution]:
        """Get executions by command type."""
        return [
            ex for ex in self.executions
            if hasattr(ex.command, "command_type") and ex.command.command_type == command_type
        ]
    
    def get_failed(self) -> list[CommandExecution]:
        """Get failed executions."""
        return [ex for ex in self.executions if ex.status == CommandStatus.FAILED]
    
    @computed_field
    @property
    def total_count(self) -> int:
        """Get total execution count."""
        return len(self.executions)
    
    @computed_field
    @property
    def success_count(self) -> int:
        """Get successful execution count."""
        return sum(1 for ex in self.executions if ex.status == CommandStatus.SUCCESS)
    
    @computed_field
    @property
    def failure_count(self) -> int:
        """Get failed execution count."""
        return sum(1 for ex in self.executions if ex.status == CommandStatus.FAILED)
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.success_count / self.total_count) * 100.0