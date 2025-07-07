"""Claude Code Agent Farm - Orchestrate multiple Claude Code agents for parallel work."""

__version__ = "2.0.0"
__author__ = "Jeffrey Emanuel"
__email__ = "jeffrey.emanuel@gmail.com"

from claude_code_agent_farm.core.orchestrator import ClaudeAgentFarm
from claude_code_agent_farm.core.monitor import AgentMonitor

__all__ = ["ClaudeAgentFarm", "AgentMonitor"]