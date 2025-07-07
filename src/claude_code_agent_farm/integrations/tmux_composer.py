"""Integration with tmux-composer-cli for tmux session management."""

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from claude_code_agent_farm.config import constants
from claude_code_agent_farm.utils import run


class TmuxComposer:
    """Wrapper for tmux-composer-cli functionality."""
    
    def __init__(self, session_name: str):
        self.session_name = session_name
        self.config_file: Optional[Path] = None
        
    def create_config(self, num_agents: int, project_path: str) -> Path:
        """Create tmux-composer configuration for the agent farm.
        
        Args:
            num_agents: Number of agents to create
            project_path: Path to the project directory
            
        Returns:
            Path to the created configuration file
        """
        config = {
            "name": self.session_name,
            "windows": [
                {
                    "name": constants.TMUX_CONTROLLER_WINDOW,
                    "commands": ["echo 'Controller window ready'"]
                },
                {
                    "name": constants.TMUX_AGENTS_WINDOW,
                    "layout": "tiled",
                    "panes": []
                }
            ]
        }
        
        # Add panes for each agent
        for i in range(num_agents):
            config["windows"][1]["panes"].append({
                "commands": [f"echo 'Agent {i} ready'"],
                "env": {"POWERLEVEL9K_INSTANT_PROMPT": "off"}
            })
            
        # Save configuration to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            self.config_file = Path(f.name)
            
        return self.config_file
        
    def start_session(self) -> None:
        """Start the tmux session using tmux-composer."""
        if not self.config_file:
            raise ValueError("No configuration file created")
            
        # Kill existing session if it exists
        run(f"tmux kill-session -t {self.session_name}", check=False, quiet=True)
        
        # Start new session with tmux-composer
        run(f"tmux-composer -f {self.config_file} start", check=True)
        
    def send_to_pane(self, pane_id: int, command: str, enter: bool = True) -> None:
        """Send command to a specific pane.
        
        Args:
            pane_id: The pane index (0-based)
            command: Command to send
            enter: Whether to send Enter key after command
        """
        target = f"{self.session_name}:{constants.TMUX_AGENTS_WINDOW}.{pane_id}"
        
        # Use tmux send-keys directly for now
        # TODO: Replace with tmux-composer API when available
        if command:
            run(f"tmux send-keys -t {target} {subprocess.list2cmdline([command])}", quiet=True)
            
        if enter:
            time.sleep(0.2)  # Small delay for Claude Code
            run(f"tmux send-keys -t {target} C-m", quiet=True)
            
    def capture_pane(self, pane_id: int) -> str:
        """Capture content from a specific pane.
        
        Args:
            pane_id: The pane index (0-based)
            
        Returns:
            Captured pane content
        """
        target = f"{self.session_name}:{constants.TMUX_AGENTS_WINDOW}.{pane_id}"
        
        try:
            _, stdout, _ = run(f"tmux capture-pane -t {target} -p", quiet=True, capture=True)
            return stdout
        except subprocess.CalledProcessError:
            return ""
            
    def set_pane_title(self, pane_id: int, title: str) -> None:
        """Set the title of a specific pane.
        
        Args:
            pane_id: The pane index (0-based)
            title: Title to set
        """
        target = f"{self.session_name}:{constants.TMUX_AGENTS_WINDOW}.{pane_id}"
        run(f"tmux select-pane -t {target} -T {subprocess.list2cmdline([title])}", quiet=True)
        
    def list_panes(self) -> List[int]:
        """List all pane IDs in the agents window.
        
        Returns:
            List of pane indices
        """
        try:
            _, stdout, _ = run(
                f"tmux list-panes -t {self.session_name}:{constants.TMUX_AGENTS_WINDOW} -F '#{{pane_index}}'",
                capture=True, quiet=True
            )
            return [int(line.strip()) for line in stdout.strip().split("\n") if line.strip()]
        except subprocess.CalledProcessError:
            return []
            
    def cleanup(self) -> None:
        """Clean up temporary configuration file."""
        if self.config_file and self.config_file.exists():
            self.config_file.unlink()