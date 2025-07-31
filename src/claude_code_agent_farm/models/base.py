"""Base models and mixins for Claude Single Agent Monitor."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TimestampedModel(BaseModel):
    """Base model with timestamp fields."""
    
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default=None)
    
    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()


class SerializableModel(BaseModel):
    """Base model with enhanced serialization support."""
    
    model_config = ConfigDict(
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
        },
        populate_by_name=True,
    )
    
    def to_json(self, **kwargs) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(**kwargs)
    
    @classmethod
    def from_json(cls, json_str: str) -> "SerializableModel":
        """Create from JSON string."""
        return cls.model_validate_json(json_str)
    
    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(**kwargs)


class EventModel(TimestampedModel, SerializableModel):
    """Base model for all events."""
    
    event_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    event_type: str = Field(..., description="Type of event")
    source: str = Field(default="system", description="Source of the event")
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        """Ensure event type is not empty."""
        if not v or not v.strip():
            raise ValueError("Event type cannot be empty")
        return v.strip().lower()


class CommandModel(SerializableModel):
    """Base model for commands."""
    
    command: str = Field(..., description="The command to execute")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    working_dir: Optional[str] = Field(default=None, description="Working directory")
    timeout: Optional[int] = Field(default=None, ge=1, description="Command timeout in seconds")
    
    @property
    def full_command(self) -> str:
        """Get the full command with arguments."""
        if self.args:
            return f"{self.command} {' '.join(self.args)}"
        return self.command
    
    @field_validator("command")
    @classmethod
    def validate_command(cls, v: str) -> str:
        """Ensure command is not empty."""
        if not v or not v.strip():
            raise ValueError("Command cannot be empty")
        return v.strip()


class StateModel(SerializableModel):
    """Base model for state tracking."""
    
    state_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    version: int = Field(default=1, ge=1)
    last_modified: datetime = Field(default_factory=datetime.now)
    
    def increment_version(self) -> None:
        """Increment the version and update timestamp."""
        self.version += 1
        self.last_modified = datetime.now()
    
    def to_checkpoint(self) -> dict[str, Any]:
        """Create a checkpoint of the current state."""
        return {
            "state_id": self.state_id,
            "version": self.version,
            "timestamp": self.last_modified.isoformat(),
            "data": self.model_dump(exclude={"state_id", "version", "last_modified"}),
        }


class MetricModel(BaseModel):
    """Base model for metrics."""
    
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    unit: Optional[str] = Field(default=None, description="Unit of measurement")
    timestamp: datetime = Field(default_factory=datetime.now)
    tags: dict[str, str] = Field(default_factory=dict)
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure metric name follows convention."""
        if not v or not v.strip():
            raise ValueError("Metric name cannot be empty")
        # Convert to lowercase with underscores
        return v.strip().lower().replace(" ", "_").replace("-", "_")
    
    def with_tag(self, key: str, value: str) -> "MetricModel":
        """Add a tag to the metric."""
        self.tags[key] = value
        return self


class ValidatedPathModel(BaseModel):
    """Model for validated file/directory paths."""
    
    path: str = Field(..., description="File or directory path")
    exists: bool = Field(default=False, description="Whether the path exists")
    is_file: bool = Field(default=False, description="Whether it's a file")
    is_dir: bool = Field(default=False, description="Whether it's a directory")
    is_absolute: bool = Field(default=False, description="Whether it's an absolute path")
    size: Optional[int] = Field(default=None, description="Size in bytes")
    
    @classmethod
    def from_path(cls, path: str) -> "ValidatedPathModel":
        """Create from a path string with validation."""
        from pathlib import Path
        
        p = Path(path)
        return cls(
            path=str(p),
            exists=p.exists(),
            is_file=p.is_file() if p.exists() else False,
            is_dir=p.is_dir() if p.exists() else False,
            is_absolute=p.is_absolute(),
            size=p.stat().st_size if p.exists() and p.is_file() else None,
        )