# Claude Code Agent Farm 🤖🚜

> Orchestrate multiple Claude Code agents working in parallel to fix TypeScript and linting errors in your codebase

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## 🎯 What is this?

Claude Code Agent Farm is a powerful orchestration tool that runs multiple Claude Code (`cc`) sessions in parallel to automatically fix TypeScript type-checking and linting errors in your codebase. It's like having a team of AI developers working simultaneously on different parts of your error list.

### Key Features

- 🚀 **Parallel Processing**: Run 20+ Claude Code agents simultaneously
- 📊 **Smart Monitoring**: Real-time dashboard showing agent status, context usage, and progress
- 🔄 **Auto-Recovery**: Automatically restarts agents when they complete work or encounter errors
- 🎯 **Conflict Prevention**: Agents mark completed tasks to avoid duplicate work
- 🛡️ **Robust Error Handling**: Detects and handles Claude Code settings corruption
- 📈 **Progress Tracking**: Automatic git commits to track error reduction over time
- 🖥️ **Flexible Viewing**: Multiple tmux viewing modes to monitor your agent army

## 📋 Prerequisites

- **Python 3.13+** (managed by `uv`)
- **tmux** (for terminal multiplexing)
- **Claude Code** (`claude` command installed and configured)
- **git** (for version control)
- **bun** (for your TypeScript project)
- **direnv** (optional but recommended for auto-environment activation)
- **uv** (modern Python package manager)

### Important: The `cc` Alias

The agent farm requires a special `cc` alias to launch Claude Code with the necessary permissions:

```bash
alias cc="ENABLE_BACKGROUND_TASKS=1 claude --dangerously-skip-permissions"
```

This alias:
- Enables background tasks for Claude Code agents
- Skips permission prompts that would block automation
- **Will be configured automatically by the setup script**

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Dicklesworthstone/claude_code_agent_farm.git
cd claude_code_agent_farm
```

### 2. Run Automated Setup

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Check for required tools
- Create a Python 3.13 virtual environment
- Install all dependencies
- Set up direnv for auto-activation
- Configure the `cc` alias (with your confirmation)
- Create sample configuration files
- Make scripts executable

### 3. Manual Setup (Alternative)

If you prefer to set up manually:

```bash
# Create virtual environment
uv venv --python 3.13

# Install dependencies
uv lock --upgrade
uv sync --all-extras

# Create .envrc for direnv
echo 'source .venv/bin/activate' > .envrc
direnv allow

# Configure the cc alias
alias cc="ENABLE_BACKGROUND_TASKS=1 claude --dangerously-skip-permissions"
# Add to your shell rc file (~/.bashrc or ~/.zshrc):
echo 'alias cc="ENABLE_BACKGROUND_TASKS=1 claude --dangerously-skip-permissions"' >> ~/.bashrc

# Make scripts executable
chmod +x view_agents.sh
```

## 📖 Understanding the Architecture

### The Two-Script System

This project consists of two independent scripts that work together:

#### 1. **Python Script** (`claude_code_agent_farm.py`) - The Brain 🧠

This is the main orchestrator that does all the heavy lifting:

- **Creates and manages tmux sessions** with multiple panes
- **Generates the error file** by running type-check and lint commands
- **Launches Claude Code agents** in each tmux pane
- **Monitors agent health** (context usage, work status, errors)
- **Auto-restarts agents** when they complete tasks or hit issues
- **Runs monitoring dashboard** in the tmux controller window
- **Handles graceful shutdown** with Ctrl+C

**You run this script and it stays running** (unless using `--no-monitor` mode). The monitoring dashboard is displayed in the tmux session's controller window, not in the launching terminal.

#### 2. **Shell Script** (`view_agents.sh`) - The Window 🪟

This is an optional convenience tool for viewing the tmux session:

- **It does NOT interact with the Python script**
- **Run it in a separate terminal** to peek at agent activity
- **Provides different viewing modes** (grid, focus, split)
- **Just a wrapper around tmux commands** for convenience

Think of it like this:
- **Python script** = Your car engine (does all the work)
- **Shell script** = Your dashboard camera (lets you see what's happening)

### Why Two Scripts?

1. **Separation of Concerns**: Core logic (Python) vs viewing utilities (shell)
2. **Flexibility**: You can monitor agents without the viewer script
3. **Independence**: Either script can be used without the other

## 🎮 Usage

### Basic Command

```bash
# Activate environment (if not using direnv)
source .venv/bin/activate

# Run with default settings (20 agents)
claude-agent-farm --path /path/to/your/typescript/project
```

### Common Usage Patterns

```bash
# Quick test with 5 agents
claude-agent-farm --path /your/project -n 5 --skip-regenerate

# Production run with auto-restart
claude-agent-farm --path /your/project --auto-restart --agents 20

# Headless mode (no monitoring)
claude-agent-farm --path /your/project --no-monitor --auto-restart

# Skip git operations for faster iteration
claude-agent-farm --path /your/project --skip-regenerate --skip-commit

# Use custom configuration
claude-agent-farm --path /your/project --config configs/production.json
```

### Viewing Your Agents

While the orchestrator is running, you have several options:

#### Option 1: Built-in Monitoring Dashboard
The monitoring dashboard now runs inside the tmux controller window, allowing you to see both the dashboard and agents in the same tmux session:

```
Claude Agent Farm - 14:32:15
┏━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Agent    ┃ Status     ┃ Cycles ┃ Context  ┃ Runtime      ┃ Errors ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Pane 00  │ working    │ 2      │ 75%      │ 0:05:23      │ 0      │
│ Pane 01  │ working    │ 2      │ 82%      │ 0:05:19      │ 0      │
│ Pane 02  │ idle       │ 3      │ 45%      │ 0:05:15      │ 0      │
└──────────┴────────────┴────────┴──────────┴──────────────┴────────┘
```

To view it:
- `tmux attach -t claude_agents:controller` - View monitoring dashboard
- Use the viewer script (Option 2) for convenient split views

#### Option 2: Tmux Viewer Script
In a separate terminal:

```bash
./view_agents.sh
```

Choose from:
1. **Grid view** - See all agents at once
2. **Focus mode** - Navigate between individual agents
3. **Split view** - Monitor dashboard + agents side by side (recommended!)
4. **Quick attach** - Direct tmux attachment

#### Option 3: Direct tmux Commands

```bash
# List sessions
tmux ls

# Attach to session
tmux attach -t claude_agents

# Navigate panes
# Ctrl+B then arrow keys

# Zoom into a pane
# Ctrl+B then Z

# Detach
# Ctrl+B then D
```

## ⚙️ Configuration

### Command Line Options

```
Required:
  --path PATH               Project root directory

Agent Configuration:
  --agents N, -n N         Number of agents (default: 20)
  --session NAME, -s NAME  tmux session name (default: claude_agents)

Timing:
  --stagger SECONDS        Delay between starting agents (default: 4.0)
  --wait-after-cc SECONDS  Wait time after launching cc (default: 3.0)
  --check-interval SECONDS Health check interval (default: 10)

Features:
  --skip-regenerate        Skip regenerating problems file
  --skip-commit           Skip git commit/push
  --auto-restart          Enable automatic agent restart
  --no-monitor            Just launch agents and exit
  --attach                Attach to tmux after setup

Advanced:
  --prompt-file PATH      Custom prompt file
  --config PATH           JSON configuration file
```

### Configuration Files

Create reusable configurations in JSON:

```json
{
    "path": "/data/projects/smartedgar-frontend",
    "agents": 20,
    "session": "claude_agents",
    "stagger": 4.0,
    "wait_after_cc": 3.0,
    "check_interval": 10,
    "skip_regenerate": false,
    "skip_commit": false,
    "auto_restart": true,
    "no_monitor": false,
    "attach": false,
    "prompt_file": "prompts/bug_fixing_prompt_for_nextjs.txt",
    "context_threshold": 20,
    "idle_timeout": 60,
    "max_errors": 3,
    "verbose": false,
    "log_file": null,
    "git_branch": null,
    "git_remote": "origin",
    "batch_size": 50,
    "max_agents": 50,
    "tmux_kill_on_exit": true,
    "tmux_mouse": true
}
```

#### Configuration Options Explained:

- **path**: Project root directory
- **agents**: Number of Claude Code agents to run
- **session**: tmux session name
- **stagger**: Delay between starting each agent (seconds)
- **wait_after_cc**: Wait time after launching cc before sending prompt
- **check_interval**: How often to check agent health
- **skip_regenerate**: Skip regenerating the problems file
- **skip_commit**: Skip git commit/push step
- **auto_restart**: Automatically restart agents on completion/error
- **no_monitor**: Disable monitoring (just launch and exit)
- **attach**: Attach to tmux session after setup
- **prompt_file**: Path to custom prompt file (null = use default)
- **context_threshold**: Restart when context drops below this percentage
- **idle_timeout**: Seconds before marking an agent as idle
- **max_errors**: Maximum errors before disabling an agent
- **verbose**: Enable verbose logging
- **log_file**: Path to log file (null = console only)
- **git_branch**: Specific branch to use (null = current branch)
- **git_remote**: Git remote to push to
- **batch_size**: Lines per batch in the prompt
- **max_agents**: Safety limit on maximum agents
- **tmux_kill_on_exit**: Kill tmux session on orchestrator exit
- **tmux_mouse**: Enable mouse support in tmux

Use with: `claude-agent-farm --path /your/project --config configs/my-config.json`

## 🔄 How It Works

### 1. **Initialization Phase**
- Changes to your project directory
- Runs `bun run type-check` and `bun run lint` to generate error list
- Commits the current error count to git
- Creates a tmux session with N agent panes

### 2. **Agent Launch Phase**
- Starts Claude Code (`cc`) in each pane
- Waits for it to initialize
- Sends the task prompt with unique randomization seed
- Staggers launches to prevent settings corruption

### 3. **Monitoring Phase** (if enabled)
- Continuously monitors each agent's status
- Tracks context usage percentage
- Detects when agents complete their work
- Identifies settings corruption or errors
- Auto-restarts agents based on configured conditions

### 4. **The Task Prompt**
Each agent receives instructions to:
- Select random 50-line chunks from the error file
- Mark selected lines with `[COMPLETED]` to prevent duplicates
- Fix issues intelligently (not just quick patches)
- Follow Next.js 15 best practices
- Commit changes when done

## 📊 Understanding Agent States

- 🟡 **starting** - Agent is initializing
- 🟢 **working** - Actively processing tasks
- 🔵 **ready** - Waiting for input
- 🟡 **idle** - Completed work, ready for restart
- 🔴 **error** - Settings corruption or other error
- ⚫ **unknown** - State unclear (rare)

## 🚨 Troubleshooting

### Common Issues

#### "Settings corrupted" errors
- Increase `--stagger` time (try 6-8 seconds)
- Reduce number of concurrent agents
- Ensure Claude Code works when run manually

#### Agents not starting
- Verify `cc` alias is configured: `alias | grep cc`
- If not, run: `alias cc="ENABLE_BACKGROUND_TASKS=1 claude --dangerously-skip-permissions"`
- Check Claude Code is configured with valid API key
- Try running `cc` manually first
- Increase `--wait-after-cc` if Claude Code loads slowly

#### High memory usage
- Each agent uses ~500MB RAM
- Reduce agent count or run in batches
- Monitor with `htop` during execution

#### tmux session issues
```bash
# Kill stuck session
tmux kill-session -t claude_agents

# List all sessions
tmux ls

# Kill all tmux sessions (nuclear option)
tmux kill-server
```

## 💡 Pro Tips

1. **Start Small**: Test with 3-5 agents before scaling up
2. **Monitor First Run**: Watch the first cycle to ensure proper behavior
3. **Resource Planning**: Plan for ~500MB RAM per agent
4. **Stagger Timing**: Increase stagger time if you see corruption
5. **Git Workflow**: Always pull before starting (script auto-commits)
6. **Context Management**: Agents restart at 20% context remaining
7. **Batch Processing**: For 50+ agents, consider multiple runs

## 🔧 Advanced Usage

### Running Multiple Projects

```bash
# Terminal 1
claude-agent-farm --path /project1 --session project1-agents

# Terminal 2
claude-agent-farm --path /project2 --session project2-agents
```

### Custom Prompts

Create a custom prompt file:
```bash
echo "Your custom instructions..." > prompts/my-prompt.txt
claude-agent-farm --path /project --prompt-file prompts/my-prompt.txt
```

### Automated/Cron Usage

For unattended operation:
```bash
#!/bin/bash
cd /path/to/agent-farm
source .venv/bin/activate
claude-agent-farm \
  --path /project \
  --agents 10 \
  --no-monitor \
  --auto-restart \
  --config configs/overnight.json
```

## 📁 Project Structure

```
claude_code_agent_farm/
├── claude_code_agent_farm.py   # Main orchestrator (Python)
├── view_agents.sh               # Tmux viewer utility (Bash)
├── setup.sh                     # Automated setup script
├── __init__.py                  # Python package file
├── pyproject.toml              # Project configuration
├── .envrc                      # Direnv configuration
├── configs/                    # Configuration files
│   └── sample.json            # Example configuration
├── prompts/                    # Custom prompt files
└── README.md                   # This file
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- Built for use with [Claude Code](https://www.anthropic.com/)
- Inspired by the need to parallelize AI-assisted code improvement
- Thanks to the tmux and Python communities

## ⚠️ Disclaimer

This tool runs multiple AI agents that will modify your code. Always:
- Back up your code before running
- Review changes before committing
- Start with a small number of agents
- Monitor the first few cycles

---

*Happy farming! 🚜 May your type errors be few and your agents be many.*