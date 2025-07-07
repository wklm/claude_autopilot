"""Configuration management for Claude Code Agent Farm."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console

console = Console(stderr=True)


class Settings:
    """Manage configuration settings for the agent farm."""
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Raises:
            ValueError: If the config file is invalid
        """
        path = Path(config_path)
        if not path.exists():
            raise ValueError(f"Config file not found: {config_path}")
            
        try:
            with path.open() as f:
                config_data = json.load(f)
                
            # Validate required fields
            if "path" not in config_data:
                raise ValueError("Config must include 'path' field")
                
            self.config = config_data
            console.print(f"[green]Loaded configuration from {config_path}[/green]")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
        
    def update(self, **kwargs: Any) -> None:
        """Update configuration with keyword arguments.
        
        Args:
            **kwargs: Configuration key-value pairs
        """
        self.config.update(kwargs)
        
    def validate(self) -> None:
        """Validate configuration settings.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required fields
        if not self.config.get("path"):
            raise ValueError("Configuration must include 'path' field")
            
        # Validate numeric ranges
        agents = self.config.get("agents", 1)
        if not 1 <= agents <= 100:
            raise ValueError(f"Number of agents must be between 1 and 100, got {agents}")
            
        # Validate session name
        session = self.config.get("session", "")
        if session and not session.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid session name: {session}. Use only letters, numbers, hyphens, and underscores.")