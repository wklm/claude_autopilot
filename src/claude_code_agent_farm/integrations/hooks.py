"""Integration with claude-code-generic-hooks for enhanced agent monitoring."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from claude_code_agent_farm.utils import run


class ClaudeHooks:
    """Manage claude-code-generic-hooks integration."""
    
    def __init__(self):
        self.hooks_repo_url = "https://github.com/possibilities/claude-code-generic-hooks"
        self.hooks_config_dir = Path.home() / ".config/claude-code/hooks"
        
    def install_hooks(self, target_dir: Optional[Path] = None) -> bool:
        """Install claude-code-generic-hooks.
        
        Args:
            target_dir: Directory to install hooks (defaults to ~/.config/claude-code/hooks)
            
        Returns:
            True if installation successful
        """
        if target_dir is None:
            target_dir = self.hooks_config_dir
            
        # Create config directory if it doesn't exist
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Clone the hooks repository
        if target_dir.exists():
            # Pull latest changes if already exists
            try:
                run(f"cd {target_dir} && git pull", check=True, quiet=True)
                return True
            except Exception:
                # If pull fails, remove and re-clone
                shutil.rmtree(target_dir)
                
        try:
            run(f"git clone {self.hooks_repo_url} {target_dir}", check=True)
            return True
        except Exception as e:
            print(f"Failed to install hooks: {e}")
            return False
            
    def create_agent_hooks_config(self, agent_id: int, project_path: Path) -> Dict[str, Any]:
        """Create hooks configuration for a specific agent.
        
        Args:
            agent_id: Agent identifier
            project_path: Path to the project
            
        Returns:
            Hooks configuration dictionary
        """
        return {
            "hooks": {
                "pre-command": [
                    {
                        "name": "log-command",
                        "script": f"echo '[Agent {agent_id}] Executing: $CLAUDE_COMMAND' >> {project_path}/.agent_logs/agent{agent_id:02d}.log"
                    }
                ],
                "post-command": [
                    {
                        "name": "log-result",
                        "script": f"echo '[Agent {agent_id}] Completed with status: $CLAUDE_EXIT_CODE' >> {project_path}/.agent_logs/agent{agent_id:02d}.log"
                    }
                ],
                "on-error": [
                    {
                        "name": "log-error",
                        "script": f"echo '[Agent {agent_id}] Error occurred: $CLAUDE_ERROR' >> {project_path}/.agent_logs/agent{agent_id:02d}.log"
                    }
                ],
                "on-context-threshold": [
                    {
                        "name": "auto-save",
                        "script": f"echo '[Agent {agent_id}] Context threshold reached, auto-saving...' >> {project_path}/.agent_logs/agent{agent_id:02d}.log",
                        "threshold": 20
                    }
                ]
            },
            "settings": {
                "log_level": "info",
                "enable_metrics": True,
                "metrics_file": f"{project_path}/.agent_metrics/agent{agent_id:02d}.json"
            }
        }
        
    def write_hooks_config(self, config: Dict[str, Any], config_path: Path) -> None:
        """Write hooks configuration to file.
        
        Args:
            config: Hooks configuration dictionary
            config_path: Path to write configuration
        """
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
    def setup_agent_hooks(self, agent_id: int, project_path: Path) -> Path:
        """Set up hooks for a specific agent.
        
        Args:
            agent_id: Agent identifier
            project_path: Path to the project
            
        Returns:
            Path to the agent's hooks configuration
        """
        # Create directories for logs and metrics
        logs_dir = project_path / ".agent_logs"
        metrics_dir = project_path / ".agent_metrics"
        logs_dir.mkdir(exist_ok=True)
        metrics_dir.mkdir(exist_ok=True)
        
        # Create hooks configuration
        config = self.create_agent_hooks_config(agent_id, project_path)
        
        # Write configuration
        config_path = project_path / f".claude_hooks/agent{agent_id:02d}_hooks.json"
        self.write_hooks_config(config, config_path)
        
        return config_path
        
    def get_default_hooks(self) -> List[Dict[str, str]]:
        """Get list of default hooks for monitoring.
        
        Returns:
            List of hook definitions
        """
        return [
            {
                "name": "monitor-performance",
                "event": "post-command",
                "description": "Track command execution time and resource usage"
            },
            {
                "name": "auto-commit",
                "event": "on-file-change",
                "description": "Automatically commit changes when files are modified"
            },
            {
                "name": "context-warning",
                "event": "on-context-threshold",
                "description": "Warn when context usage is high"
            },
            {
                "name": "error-recovery",
                "event": "on-error",
                "description": "Attempt to recover from common errors"
            }
        ]