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
- 🌐 **Multi-Stack Support**: Next.js, Python, Rust, or any custom stack
- 📊 **Smart Monitoring**: Real-time dashboard showing agent status and progress
- 🔄 **Auto-Recovery**: Automatically restarts agents when needed
- 📈 **Progress Tracking**: Git commits and structured progress documents
- ⚙️ **Highly Configurable**: JSON configs with variable substitution
- 🖥️ **Flexible Viewing**: Multiple tmux viewing modes

## 📋 Prerequisites

- **Python 3.13+** (managed by `uv`)
- **tmux** (for terminal multiplexing)
- **Claude Code** (`claude` command installed and configured)
- **git** (for version control)
- **Your project's tools** (e.g., `bun` for Next.js, `mypy`/`ruff` for Python)
- **direnv** (optional but recommended)
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

## 🎮 Supported Workflows

### 1. Bug Fixing Workflow

Agents work through type-checker and linter problems in parallel:
- Runs your configured type-check and lint commands
- Generates a combined problems file
- Agents select random chunks to fix
- Marks completed problems to avoid duplication
- Focuses on fixing existing issues

### 2. Best Practices Implementation Workflow

Agents systematically implement modern best practices:
- Reads a comprehensive best practices guide
- Creates a progress tracking document
- Implements improvements in manageable chunks
- Tracks completion percentage for each guideline
- Maintains continuity between sessions

## 🌐 Technology Stack Support

### Built-in Support

#### Next.js (TypeScript)
- Tools: `bun`, `tsc`, `eslint`
- Config: `configs/nextjs_config.json`
- Best practices guide included

#### Python
- Tools: `mypy`, `ruff`, optional `uv`
- Configs: `configs/python_config.json`, `configs/python_uv_config.json`
- Supports modern Python patterns

### Custom Tech Stacks

Create your own configuration:

```json
{
  "tech_stack": "rust",
  "problem_commands": {
    "type_check": ["cargo", "check"],
    "lint": ["cargo", "clippy", "--", "-D", "warnings"]
  },
  "best_practices_files": ["./guides/RUST_BEST_PRACTICES.md"],
  "chunk_size": 30,
  "prompt_file": "prompts/rust_prompt.txt"
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
  "idle_timeout": 60
}
```

### Key Parameters

- **tech_stack**: Technology identifier (nextjs, python, rust, etc.)
- **problem_commands**: Commands for type-checking and linting
- **best_practices_files**: Guides to copy to the project
- **chunk_size**: How many lines/changes per agent iteration
- **prompt_file**: Which prompt template to use
- **auto_restart**: Enable automatic agent restart
- **context_threshold**: Restart when context drops below this %

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

## 📝 Prompt System

### Prompt Types

#### Bug Fixing Prompts
- `default_prompt.txt` - Generic bug fixing
- `default_prompt_nextjs.txt` - Next.js specific
- `default_prompt_python.txt` - Python specific

#### Best Practices Prompts
- `default_best_practices_prompt.txt` - Generic implementation
- `default_best_practices_prompt_nextjs.txt` - Next.js specific
- `default_best_practices_prompt_python.txt` - Python specific
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
2. **Progress Document**: Agents create tracking document
3. **Systematic Implementation**: Works through guidelines incrementally
4. **Accurate Tracking**: Maintains honest completion percentages
5. **Session Continuity**: Progress persists between runs

## 📊 Monitoring and Viewing

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

## 🚨 Troubleshooting

### Common Issues

#### Agents not starting
- Verify `cc` alias: `alias | grep cc`
- Test Claude Code manually: `cc`
- Check API key configuration
- Increase `--wait-after-cc` timing

#### Configuration errors
- Validate JSON syntax
- Ensure all paths are correct
- Check command availability (mypy, ruff, etc.)

#### Resource issues
- Each agent uses ~500MB RAM
- Reduce agent count if needed
- Monitor with `htop`

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
│   └── nextjs_best_practices_config.json
├── prompts/                     # Prompt templates
│   ├── default_prompt_*.txt    # Bug fixing prompts
│   └── default_best_practices_*.txt
├── best_practices_guides/       # Best practices documents
│   └── NEXTJS15_BEST_PRACTICES.md
└── README_MODULAR_SYSTEM.md     # Detailed configuration guide
```

## 🔧 Advanced Topics

### Creating Custom Workflows

1. **Define your tech stack config**
2. **Create appropriate prompts**
3. **Add best practices guides** (optional)
4. **Configure problem commands**

### Scaling Considerations

- Start small (5-10 agents) and scale up
- Increase stagger time for many agents
- Consider running in batches for 50+ agents
- Use `--no-monitor` for headless operation

### Integration with CI/CD

```bash
#!/bin/bash
# Automated code improvement script
claude-code-agent-farm \
  --path $PROJECT_PATH \
  --config configs/ci-config.json \
  --no-monitor \
  --auto-restart \
  --skip-commit
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests if applicable
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## ⚠️ Important Notes

- **Always backup your code** before running
- **Review changes** before committing
- **Start with few agents** to test
- **Monitor first runs** to ensure proper behavior

---

*Happy farming! 🚜 May your code be clean and your agents productive.*