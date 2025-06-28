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
- **Claude Code** (`cc` command installed and configured)
- **git** (for version control)
- **bun** (for your TypeScript project)
- **direnv** (optional but recommended for auto-environment activation)
- **uv** (modern Python package manager)

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
- **Displays real-time dashboard** showing all agent statuses
- **Handles graceful shutdown** with Ctrl+C

**You run this script and it stays running** (unless using `--no-monitor` mode).

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

#### Option 1: Built-in Monitoring Dashboard (Default)
The Python script shows a real-time status table:

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

#### Option 2: Tmux Viewer Script
In a separate terminal:

```bash
./view_agents.sh
```

Choose from:
1. **Grid view** - See all agents at once
2. **Focus mode** - Navigate between individual agents
3. **Split view** - Controller + agents side by side
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

Create reusable configurations in JSON (note that comments aren't permitted in JSON, so remove those!)

```json
{
    // Project Configuration
    "path": "/data/projects/smartedgar-frontend",  // Project root directory
    
    // Agent Configuration
    "agents": 20,                    // Number of Claude Code agents to run
    "session": "claude_agents",      // tmux session name
    
    // Timing Configuration (in seconds)
    "stagger": 4.0,                  // Delay between starting each agent
    "wait_after_cc": 3.0,           // Wait time after launching cc before sending prompt
    "check_interval": 10,            // How often to check agent health
    
    // Feature Flags
    "skip_regenerate": false,        // Skip regenerating the problems file
    "skip_commit": false,           // Skip git commit/push step
    "auto_restart": true,           // Automatically restart agents on completion/error
    "no_monitor": false,            // Disable monitoring (just launch and exit)
    "attach": false,                // Attach to tmux session after setup
    
    // Advanced Options
    "prompt_file": null,            // Path to custom prompt file (null = use default)
    
    // You can also override the default prompt inline (not recommended for long prompts)
    // "prompt_text": "Your custom prompt here...",
    
    // Additional settings for fine-tuning
    "context_threshold": 20,        // Restart when context drops below this percentage
    "idle_timeout": 60,             // Seconds before marking an agent as idle
    "max_errors": 3,                // Maximum errors before disabling an agent
    
    // Logging and debugging
    "verbose": false,               // Enable verbose logging
    "log_file": null,              // Path to log file (null = console only)
    
    // Git configuration
    "git_branch": null,            // Specific branch to use (null = current branch)
    "git_remote": "origin",        // Git remote to push to
    
    // Performance tuning
    "batch_size": 50,              // Lines per batch in the prompt
    "max_agents": 50,              // Safety limit on maximum agents
    
    // tmux specific settings
    "tmux_kill_on_exit": true,     // Kill tmux session on orchestrator exit
    "tmux_mouse": true             // Enable mouse support in tmux
  }
```

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
- Verify `cc` command works: `which cc`
- Check Claude Code is configured with valid API key
- Try running `cc` manually first
- Increase `--wait-after-cc` if CC loads slowly

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