# Claude Code Agent Farm 🤖🚜

> Orchestrate multiple Claude Code agents working in parallel to improve your codebase through automated bug fixing or systematic best practices implementation

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## 🎯 What is this?

Claude Code Agent Farm is a powerful orchestration framework that runs multiple Claude Code (`cc`) sessions in parallel to systematically improve your codebase. It supports multiple technology stacks and workflow types, allowing teams of AI agents to work together on large-scale code improvements.

### Key Features

- 🚀 **Parallel Processing**: Run 20+ Claude Code agents simultaneously
- 🎯 **Multiple Workflows**: Bug fixing or best practices implementation
- 🌐 **Multi-Stack Support**: Next.js, Python, Rust, Go, Java, and 10+ more stacks
- 📊 **Smart Monitoring**: Real-time dashboard showing agent status and progress
- 🔄 **Auto-Recovery**: Automatically restarts agents when needed
- 📈 **Progress Tracking**: Git commits and structured progress documents
- ⚙️ **Highly Configurable**: JSON configs with variable substitution
- 🖥️ **Flexible Viewing**: Multiple tmux viewing modes
- 🔒 **Safe Operation**: Automatic settings backup/restore, file locking, atomic operations
- 🛠️ **Development Setup**: Integrated tool installation scripts for complete environments

## 📋 Prerequisites

- **Python 3.13+** (managed by `uv`)
- **tmux** (for terminal multiplexing)
- **Claude Code** (`claude` command installed and configured)
- **git** (for version control)
- **Your project's tools** (e.g., `bun` for Next.js, `mypy`/`ruff` for Python)
- **direnv** (optional but recommended for automatic environment activation)
- **uv** (modern Python package manager)

### Important: The `cc` Alias

The agent farm requires a special `cc` alias to launch Claude Code with the necessary permissions:

```bash
alias cc="ENABLE_BACKGROUND_TASKS=1 claude --dangerously-skip-permissions"
```

This alias will be configured automatically by the setup script.

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/Dicklesworthstone/claude_code_agent_farm.git
cd claude_code_agent_farm
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Check and install missing prerequisites
- Create a Python 3.13 virtual environment
- Install all dependencies
- Configure the `cc` alias
- Set up direnv for automatic environment activation
- Handle both bash and zsh shells automatically

### 2. Choose Your Workflow

#### For Bug Fixing (Traditional)
```bash
# Next.js project
claude-code-agent-farm --path /path/to/project --config configs/nextjs_config.json

# Python project
claude-code-agent-farm --path /path/to/project --config configs/python_config.json
```

#### For Best Practices Implementation
```bash
# Ensure you have a best practices guide in place
cp best_practices_guides/NEXTJS15_BEST_PRACTICES.md /path/to/project/best_practices_guides/

# Run with best practices config
claude-code-agent-farm --path /path/to/project --config configs/nextjs_best_practices_config.json
```

## 🛠️ Tool Setup Scripts

The project includes a comprehensive modular system for setting up development environments:

### Available Setup Scripts

Run the interactive menu:
```bash
cd tool_setup_scripts
./setup.sh
```

Or run specific setups directly:

1. **Python FastAPI** (`setup_python_fastapi.sh`)
   - Python 3.12+, uv, ruff, mypy, pre-commit, ipython

2. **Go Web Apps** (`setup_go_webapps.sh`)
   - Go 1.23+, golangci-lint, air, migrate, mockery, Task, swag

3. **Next.js** (`setup_nextjs.sh`)
   - Node.js 22+, Bun, pnpm, TypeScript, ESLint, Prettier

4. **SvelteKit/Remix/Astro** (`setup_sveltekit_remix_astro.sh`)
   - Extends Next.js setup with Vite, Playwright, Vitest, Biome

5. **Rust Development** (`setup_rust.sh`)
   - Rust toolchain, cargo tools, web & system programming tools

6. **Java Enterprise** (`setup_java_enterprise.sh`)
   - Java 21 LTS, SDKMAN, Gradle 8.11+, Maven 3.9+, JBang

7. **Bash/Zsh Scripting** (`setup_bash_zsh.sh`)
   - Shell development tools and best practices

8. **Cloud Native DevOps** (`setup_cloud_native_devops.sh`)
   - Docker, Kubernetes, Terraform, cloud tools

9. **GenAI/LLM Ops** (`setup_genai_llm_ops.sh`)
   - ML/AI development tools and frameworks

10. **Data Engineering** (`setup_data_engineering.sh`)
    - Data processing and analytics tools

11. **Serverless Edge** (`setup_serverless_edge.sh`)
    - Serverless and edge computing tools

### Setup Features

- 🎨 **Interactive & Safe**: Colorful prompts, always asks before installing
- 🔍 **Smart Detection**: Checks existing installations to avoid conflicts
- 🛡️ **Non-Destructive**: Won't overwrite configurations without permission
- 🐚 **Shell Agnostic**: Works with both bash and zsh
- 📊 **Progress Tracking**: Shows what's installed and what's pending

## 📖 Understanding the Architecture

### The Two-Script System

This project consists of two independent scripts that work together:

#### 1. **Python Script** (`claude_code_agent_farm.py`) - The Brain 🧠

This is the main orchestrator that does all the heavy lifting:

- **Creates and manages tmux sessions** with multiple panes
- **Generates the problems file** by running configured commands
- **Launches Claude Code agents** in each tmux pane
- **Monitors agent health** (context usage, work status, errors)
- **Auto-restarts agents** when they complete tasks or hit issues
- **Runs monitoring dashboard** in the tmux controller window
- **Handles graceful shutdown** with Ctrl+C
- **Manages settings backup/restore** to prevent corruption
- **Implements file locking** for concurrent access safety
- **Writes monitor state** to JSON file for external monitoring

**You run this script and it stays running** (unless using `--no-monitor` mode). The monitoring dashboard is displayed in the tmux session's controller window, not in the launching terminal.

#### 2. **Shell Script** (`view_agents.sh`) - The Window 🪟

This is an optional convenience tool for viewing the tmux session:

- **It does NOT interact with the Python script**
- **Run it in a separate terminal** to peek at agent activity
- **Provides different viewing modes** (grid, focus, split)
- **Just a wrapper around tmux commands** for convenience
- **Automatically suggests font size adjustments** for many agents

Think of it like this:
- **Python script** = Your car engine (does all the work)
- **Shell script** = Your dashboard camera (lets you see what's happening)

### Hidden Commands

#### Monitor-Only Mode
There's a hidden command for running just the monitor display:

```bash
claude-code-agent-farm monitor-only --path /project --session claude_agents
```

This reads the monitor state file and displays the dashboard without launching agents.

### Why Two Scripts?

1. **Separation of Concerns**: Core logic (Python) vs viewing utilities (shell)
2. **Flexibility**: You can monitor agents without the viewer script
3. **Independence**: Either script can be used without the other

## 🎮 Supported Workflows

### 1. Bug Fixing Workflow

Agents work through type-checker and linter problems in parallel:
- Runs your configured type-check and lint commands
- Generates a combined problems file
- Agents select random chunks to fix
- Marks completed problems to avoid duplication
- Focuses on fixing existing issues
- Uses instance-specific seeds for better randomization

### 2. Best Practices Implementation Workflow

Agents systematically implement modern best practices:
- Reads a comprehensive best practices guide
- Creates a progress tracking document (`@<STACK>_BEST_PRACTICES_IMPLEMENTATION_PROGRESS.md`)
- Implements improvements in manageable chunks
- Tracks completion percentage for each guideline
- Maintains continuity between sessions
- Supports continuing existing work with special prompts

## 🌐 Technology Stack Support

### Built-in Configurations

The project includes pre-configured support for:

1. **Next.js** - TypeScript, React, modern web development
2. **Python** - FastAPI, Django, data science workflows
3. **Rust** - System programming and web applications
4. **Go** - Web services and cloud-native applications
5. **Java** - Enterprise applications with Spring Boot
6. **SvelteKit** - Modern web framework
7. **Remix/Astro** - Full-stack web frameworks
8. **Bash/Zsh** - Shell scripting and automation
9. **Terraform/Azure** - Infrastructure as Code
10. **Cloud Native DevOps** - Kubernetes, Docker, CI/CD
11. **GenAI/LLM Ops** - AI/ML operations and tooling
12. **Data Engineering** - ETL, analytics, big data
13. **Serverless Edge** - Edge computing and serverless

Each stack includes:
- Optimized configuration file
- Technology-specific prompts
- Comprehensive best practices guide
- Appropriate chunk sizes and timing

### Custom Tech Stacks

Create your own configuration:

```json
{
  "comment": "Custom Rust configuration",
  "tech_stack": "rust",
  "problem_commands": {
    "type_check": ["cargo", "check"],
    "lint": ["cargo", "clippy", "--", "-D", "warnings"]
  },
  "best_practices_files": ["./guides/RUST_BEST_PRACTICES.md"],
  "chunk_size": 30,
  "prompt_file": "prompts/rust_prompt.txt",
  "agents": 15,
  "auto_restart": true,
  "git_branch": "feature/rust-improvements",
  "git_remote": "origin"
}
```

## ⚙️ Configuration System

### Core Configuration Options

```json
{
  "comment": "Human-readable description",
  "tech_stack": "nextjs",
  "problem_commands": {
    "type_check": ["bun", "run", "type-check"],
    "lint": ["bun", "run", "lint"]
  },
  "best_practices_files": ["./best_practices_guides/NEXTJS15_BEST_PRACTICES.md"],
  "chunk_size": 50,
  "agents": 20,
  "session": "claude_agents",
  "prompt_file": "prompts/default_prompt_nextjs.txt",
  "auto_restart": true,
  "context_threshold": 20,
  "idle_timeout": 60,
  "max_errors": 3,
  "git_branch": null,
  "git_remote": "origin",
  "tmux_kill_on_exit": true,
  "tmux_mouse": true
}
```

### Key Parameters

- **tech_stack**: Technology identifier (nextjs, python, rust, etc.)
- **problem_commands**: Commands for type-checking and linting
- **best_practices_files**: Guides to copy to the project
- **chunk_size**: How many lines/changes per agent iteration (varies by stack: 20-75)
- **prompt_file**: Which prompt template to use
- **auto_restart**: Enable automatic agent restart
- **context_threshold**: Restart when context drops below this %
- **git_branch**: Optional specific branch to commit to
- **git_remote**: Remote to push to (default: origin)

### Command Line Options

All configuration options can be overridden via CLI:

```bash
claude-code-agent-farm \
  --path /project \
  --config configs/base.json \
  --agents 10 \
  --chunk-size 30 \
  --auto-restart
```

#### Complete Options Reference

```
Required:
  --path PATH               Project root directory

Agent Configuration:
  --agents N, -n N         Number of agents (default: 20)
  --session NAME, -s NAME  tmux session name (default: claude_agents)
  --chunk-size N           Override config chunk size

Timing:
  --stagger SECONDS        Delay between starting agents (default: 10.0)
  --wait-after-cc SECONDS  Wait time after launching cc (default: 15.0)
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
  --context-threshold N    Restart agent when context ≤ N% (default: 20)
  --idle-timeout SECONDS   Mark agent idle after N seconds (default: 60)
  --max-errors N           Disable agent after N errors (default: 3)
  --tmux-kill-on-exit      Kill tmux session on exit (default: true)
  --no-tmux-kill-on-exit   Keep tmux session running after exit
  --tmux-mouse             Enable tmux mouse support (default: true)
  --no-tmux-mouse          Disable tmux mouse support
  --fast-start             Skip shell prompt detection
  --full-backup            Full backup of Claude settings before start
```

## 📝 Prompt System

### Prompt Types

The system includes 20+ specialized prompts:

#### Bug Fixing Prompts
- `default_prompt.txt` - Generic bug fixing
- `default_prompt_nextjs.txt` - Next.js specific
- `default_prompt_python.txt` - Python specific
- `bug_fixing_prompt_for_nextjs.txt` - Advanced Next.js fixing

#### Best Practices Prompts
- `default_best_practices_prompt.txt` - Generic implementation
- `default_best_practices_prompt_nextjs.txt` - Next.js 15
- `default_best_practices_prompt_python.txt` - Python/FastAPI
- `default_best_practices_prompt_rust_web.txt` - Rust web apps
- `default_best_practices_prompt_rust_system.txt` - Rust systems
- `default_best_practices_prompt_go.txt` - Go applications
- `default_best_practices_prompt_java.txt` - Java enterprise
- `default_best_practices_prompt_sveltekit.txt` - SvelteKit
- `default_best_practices_prompt_remix_astro.txt` - Remix/Astro
- `default_best_practices_prompt_bash_zsh.txt` - Shell scripting
- `default_best_practices_prompt_terraform_azure.txt` - IaC
- `default_best_practices_prompt_cloud_native_devops.txt` - DevOps
- `default_best_practices_prompt_genai_llm_ops.txt` - AI/ML ops
- `default_best_practices_prompt_data_engineering.txt` - Data pipelines
- `default_best_practices_prompt_serverless_edge.txt` - Edge computing
- `continue_best_practices_prompt.txt` - Continue existing work

### Variable Substitution

Prompts support dynamic variables:
- `{chunk_size}` - Replaced with configured chunk size

Example in prompt:
```
Work on approximately {chunk_size} improvements at a time...
```

## 🔄 How It Works

### Bug Fixing Workflow

1. **Problem Generation**: Runs type-check and lint commands
2. **Agent Launch**: Starts N agents in tmux panes
3. **Task Distribution**: Each agent selects random problem chunks
4. **Conflict Prevention**: Marks completed problems with [COMPLETED]
5. **Progress Tracking**: Commits changes and tracks error reduction

### Best Practices Workflow

1. **Guide Distribution**: Copies best practices guides to project
2. **Progress Document**: Agents create/update tracking document
3. **Systematic Implementation**: Works through guidelines incrementally
4. **Accurate Tracking**: Maintains honest completion percentages
5. **Session Continuity**: Progress persists between runs

### Safety Features

1. **Settings Backup**: Automatically backs up Claude settings before starting
2. **Settings Restore**: Restores from backup if corruption detected
3. **File Locking**: Uses file locks to prevent concurrent access issues
4. **Atomic Operations**: Uses atomic file operations for safety
5. **Emergency Cleanup**: Handles unexpected exits gracefully
6. **Launch Locking**: Prevents concurrent Claude launches with lock files
7. **Dynamic Stagger**: Adjusts launch delays based on error detection

## 📊 Monitoring Dashboard

The Python script includes a real-time monitoring dashboard that shows:

- **Agent Status**: Working, Idle, Context Low, Error, Disabled
- **Context Usage**: Percentage of agent's context window used
- **Last Activity**: Time since the agent last did something
- **Last Error**: Most recent error message (if any)
- **Session Stats**: Total restarts, uptime, active agents
- **Cycle Count**: Number of work cycles completed

### Built-in Dashboard

The monitoring dashboard runs in the tmux controller window:

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

### Viewing Options

```bash
# Use the viewer script
./view_agents.sh

# Direct tmux commands
tmux attach -t claude_agents
tmux attach -t claude_agents:controller  # Dashboard only
```

### Agent States

- 🟡 **starting** - Agent initializing
- 🟢 **working** - Actively processing
- 🔵 **ready** - Waiting for input
- 🟡 **idle** - Completed work
- 🔴 **error** - Problem detected
- ⚫ **unknown** - State unclear

### Monitor State File

The system writes monitor state to `.claude_agent_farm_state.json` in the project directory. This file contains:
- Agent statuses and health metrics
- Session information
- Runtime statistics

External tools can read this file to monitor the farm's progress.

## 💡 Usage Examples

### Quick Test Run
```bash
# 5 agents, skip git operations
claude-code-agent-farm --path /project -n 5 --skip-regenerate --skip-commit
```

### Production Bug Fixing
```bash
# Full run with Python project
claude-code-agent-farm \
  --path /python/project \
  --config configs/python_uv_config.json \
  --agents 15 \
  --auto-restart
```

### Best Practices Implementation
```bash
# Systematic improvements
claude-code-agent-farm \
  --path /nextjs/project \
  --config configs/nextjs_best_practices_config.json \
  --agents 10
```

### Custom Configuration
```bash
# Override config settings
claude-code-agent-farm \
  --path /project \
  --config configs/base.json \
  --chunk-size 25 \
  --context-threshold 15 \
  --idle-timeout 120
```

### Headless Operation
```bash
# Run without monitoring (for CI/CD)
claude-code-agent-farm \
  --path /project \
  --config configs/ci-config.json \
  --no-monitor \
  --auto-restart
```

## 🚨 Troubleshooting

### Common Issues

#### Agents not starting
- Verify `cc` alias: `alias | grep cc`
- Test Claude Code manually: `cc`
- Check API key configuration
- Increase `--wait-after-cc` timing
- Use `--full-backup` flag if settings corruption suspected

#### Configuration errors
- Validate JSON syntax
- Ensure all paths are correct
- Check command availability (mypy, ruff, etc.)
- Verify best practices guides exist

#### Resource issues
- Each agent uses ~500MB RAM
- Reduce agent count if needed
- Monitor with `htop`
- Check available disk space for logs

#### Settings corruption
- System automatically backs up settings
- Restores from backup on error detection
- Manual restore: Check `~/.claude/backups/`
- Use `--full-backup` for comprehensive backup

### Debug Features

- **State File**: Check `.claude_agent_farm_state.json` for agent status
- **Lock Files**: Look for `.agent_farm_launch.lock` in `~/.claude/`
- **Backup Directory**: `~/.claude/backups/` contains settings backups
- **Emergency Cleanup**: Ctrl+C triggers graceful shutdown
- **Manual tmux**: `tmux kill-session -t claude_agents` to force cleanup

## 📁 Project Structure

```
claude_code_agent_farm/
├── claude_code_agent_farm.py    # Main orchestrator
├── view_agents.sh               # Tmux viewer utility
├── setup.sh                     # Automated setup
├── configs/                     # Configuration files
│   ├── nextjs_config.json      # Next.js bug fixing
│   ├── python_config.json      # Python bug fixing
│   ├── python_uv_config.json   # Python with uv
│   ├── nextjs_best_practices_config.json
│   ├── rust_system_config.json # Rust systems programming
│   ├── rust_webapps_config.json # Rust web apps
│   ├── go_webapps_config.json  # Go web development
│   ├── java_enterprise_config.json # Java enterprise
│   ├── sveltekit2_config.json  # SvelteKit framework
│   ├── remix_astro_config.json # Remix/Astro frameworks
│   ├── bash_zsh_config.json    # Shell scripting
│   ├── terraform_azure_config.json # Infrastructure as Code
│   ├── cloud_native_devops_config.json # DevOps tools
│   ├── genai_llm_ops_config.json # AI/ML operations
│   ├── data_engineering_config.json # Data pipelines
│   ├── serverless_edge_config.json # Edge computing
│   └── sample.json             # Example configuration
├── prompts/                     # Prompt templates
│   ├── default_prompt_*.txt    # Bug fixing prompts
│   ├── default_best_practices_*.txt # Best practices prompts
│   └── continue_best_practices_prompt.txt
├── best_practices_guides/       # Best practices documents
│   ├── NEXTJS15_BEST_PRACTICES.md
│   ├── PYTHON_FASTAPI_BEST_PRACTICES.md
│   ├── RUST_SYSTEM_PROGRAMMING_BEST_PRACTICES.md
│   ├── RUST_WEBAPPS_BEST_PRACTICES.md
│   ├── GO_WEBAPPS_BEST_PRACTICES.md
│   ├── JAVA_ENTERPRISE_BEST_PRACTICES.md
│   ├── SVELTEKIT2_BEST_PRACTICES.md
│   ├── REMIX_ASTRO_BEST_PRACTICES.md
│   ├── BASH_AND_ZSH_SCRIPTING_FOR_UBUNTU.md
│   ├── TERRAFORM_WITH_AZURE_BEST_PRACTICES.md
│   ├── CLOUD_NATIVE_DEVOPS_BEST_PRACTICES.md
│   ├── GENAI_LLM_OPS_BEST_PRACTICES.md
│   ├── DATA_ENGINEERING_AND_ANALYTICS_BEST_PRACTICES.md
│   └── SERVERLESS_EDGE_BEST_PRACTICES.md
├── tool_setup_scripts/          # Development environment setup
│   ├── setup.sh                # Interactive menu
│   ├── common_utils.sh         # Shared utilities
│   ├── setup_python_fastapi.sh
│   ├── setup_go_webapps.sh
│   ├── setup_nextjs.sh
│   ├── setup_sveltekit_remix_astro.sh
│   ├── setup_rust.sh
│   ├── setup_java_enterprise.sh
│   ├── setup_bash_zsh.sh
│   ├── setup_cloud_native_devops.sh
│   ├── setup_genai_llm_ops.sh
│   ├── setup_data_engineering.sh
│   ├── setup_serverless_edge.sh
│   └── README.md               # Setup scripts documentation
├── pyproject.toml              # Python project configuration
├── uv.lock                     # Locked dependencies
├── .envrc                      # direnv configuration
└── .gitignore                  # Git ignore patterns
```

## 🔧 Advanced Topics

### Creating Custom Workflows

1. **Define your tech stack config**
2. **Create appropriate prompts**
3. **Add best practices guides** (optional)
4. **Configure problem commands**
5. **Set appropriate chunk sizes** (20-75 based on complexity)
6. **Test with small agent counts first**

### Scaling Considerations

- Start small (5-10 agents) and scale up
- Increase stagger time for many agents
- Consider running in batches for 50+ agents
- Use `--no-monitor` for headless operation
- Monitor system resources (RAM, CPU)
- Adjust chunk sizes based on performance

### Integration with CI/CD

```bash
#!/bin/bash
# Automated code improvement script
claude-code-agent-farm \
  --path $PROJECT_PATH \
  --config configs/ci-config.json \
  --no-monitor \
  --auto-restart \
  --skip-commit \
  --agents 10
```

### Custom Git Workflows

Configure custom git branches and remotes in your config:

```json
{
  "git_branch": "feature/ai-improvements",
  "git_remote": "upstream",
  "skip_commit": false
}
```

### Performance Tuning

- **Chunk Size**: Smaller chunks (20-30) for complex tasks, larger (50-75) for simple fixes
- **Stagger Time**: Increase for many agents or slow systems
- **Context Threshold**: Lower values (15-20%) restart agents sooner
- **Idle Timeout**: Adjust based on task complexity
- **Check Interval**: Balance between responsiveness and CPU usage

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests if applicable
4. Update documentation
5. Submit a pull request

### Adding New Tech Stacks

1. Create config file in `configs/`
2. Add prompts in `prompts/`
3. Write best practices guide in `best_practices_guides/`
4. Add setup script in `tool_setup_scripts/`
5. Test thoroughly with various project types

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## ⚠️ Important Notes

- **Always backup your code** before running
- **Review changes** before committing
- **Start with few agents** to test
- **Monitor first runs** to ensure proper behavior
- **Check resource usage** for large agent counts
- **Verify cc alias** is properly configured
- **Ensure git is configured** with proper credentials

---

*Happy farming! 🚜 May your code be clean and your agents productive.*