# Bash Aliases for Claude Flutter Firebase Agent

This document describes the bash aliases and shell integrations available for the Claude Flutter Firebase Agent.

## Installation

```bash
# Run the setup script
./scripts/setup_aliases.sh

# Or manually source the aliases
source /home/wojtek/dev/claude_code_agent_farm/.bash_aliases
```

## Core Aliases

### Agent Management
| Alias | Description | Example |
|-------|-------------|---------|
| `cffa` | Start agent with default settings | `cffa` |
| `cffa-carenji` | Start agent for Carenji development | `cffa-carenji` |
| `cffa-prompt` | Start with specific prompt | `cffa-prompt "Fix navigation bug"` |
| `cffa-stop` | Stop the agent | `cffa-stop` |
| `cffa-restart` | Restart the agent | `cffa-restart` |
| `cffa-status` | Check agent status | `cffa-status` |
| `cffa-logs` | View and follow logs | `cffa-logs` |

### Docker Operations
| Alias | Description | Example |
|-------|-------------|---------|
| `cffa-build` | Build Docker image | `cffa-build` |
| `cffa-docker` | Run agent in Docker | `cffa-docker` |
| `cffa-docker-carenji` | Run in Docker for Carenji | `cffa-docker-carenji` |
| `cffa-shell` | Shell into container | `cffa-shell` |

### Tmux Management
| Alias | Description | Example |
|-------|-------------|---------|
| `cffa-attach` | Attach to tmux session | `cffa-attach` |
| `cffa-capture` | Capture pane content | `cffa-capture` |
| `cffa-send` | Send keys to session | `cffa-send "help"` |
| `cffa-sessions` | List Claude sessions | `cffa-sessions` |

### Testing
| Alias | Description | Example |
|-------|-------------|---------|
| `cffa-test` | Run all tests | `cffa-test` |
| `cffa-test-unit` | Run unit tests | `cffa-test-unit` |
| `cffa-test-quick` | Quick tests (no Docker) | `cffa-test-quick` |
| `cffa-test-cov` | Tests with coverage | `cffa-test-cov` |
| `cffa-test-carenji` | Carenji-specific tests | `cffa-test-carenji` |

### Firebase Emulators
| Alias | Description | Example |
|-------|-------------|---------|
| `fb-start` | Start all emulators | `fb-start` |
| `fb-auth` | Start auth emulator only | `fb-auth` |
| `fb-firestore` | Start Firestore only | `fb-firestore` |
| `fb-ui` | Open emulator UI | `fb-ui` |
| `fb-kill` | Kill all emulators | `fb-kill` |

### Flutter Shortcuts
| Alias | Description | Example |
|-------|-------------|---------|
| `fl` | Flutter command | `fl doctor` |
| `flr` | Flutter run | `flr` |
| `flt` | Flutter test | `flt` |
| `fla` | Flutter analyze | `fla` |
| `flc` | Flutter clean | `flc` |
| `flpg` | Flutter pub get | `flpg` |
| `fltc` | Test with coverage + HTML | `fltc` |

## Advanced Functions

### Custom Prompt
```bash
# Start agent with custom prompt
cffa-custom "Implement user authentication"
```

### Timed Sessions
```bash
# Run agent for 1 hour
cffa-timed 3600

# Run for 30 minutes
cffa-timed 1800
```

### Session Management
```bash
# Save current session
cffa-save my_session

# Restore session
cffa-restore my_session

# List saved sessions
cffa-restore
```

### Workflow Automation
```bash
# Complete Carenji workflow (emulators + agent)
cffa-workflow

# Quick fix for analyzer issues
cffa-fix

# Interactive prompt menu
cffa-menu
```

### Monitoring
```bash
# Live dashboard
cffa-dashboard

# Watch status updates
cffa-watch

# Performance monitoring
cffa-perf

# Colored status
cffa-status-color
```

## Shell Integration Features

### Prompt Integration
Add agent status to your bash prompt:
```bash
# Add to .bashrc
PS1='$(cffa_prompt_status) \u@\h:\w\$ '
```

This shows:
- 🤖 when agent is running
- ❌ when there's an error
- ⏸️ when usage limit is reached

### Auto-Detection
Enable automatic Carenji project detection:
```bash
# Add to .bashrc
alias cd='cffa_cd'
```

This will offer to start the agent when you `cd` into a Carenji project.

### Smart Testing
```bash
# Automatically runs appropriate tests based on current directory
cffa-smart-test
```

## Utility Commands

### Cleanup
```bash
# Remove old sessions and logs
cffa-cleanup

# Backup logs with timestamp
cffa-backup-logs
```

### Information
```bash
# Show quick help
cffa-help

# Show environment info
cffa-info

# List all aliases
alias | grep cffa
```

### Git Integration
```bash
# Commit with AI assistance
cffa-commit

# Show agent-related changes
cffa-git
```

## Environment Variables

Set these in your `.bashrc` or `.zshrc`:

```bash
# Default project path
export CLAUDE_PROJECT_PATH="/home/wojtek/dev/carenji"

# Default configuration file
export CLAUDE_AGENT_CONFIG="$HOME/.claude-flutter-agent/config.yaml"

# Editor for config files
export CLAUDE_EDITOR="vim"  # or "code", "nano", etc.

# Enable debug mode
export CLAUDE_AGENT_DEBUG=1
export CLAUDE_AGENT_LOG_LEVEL=DEBUG
```

## Tips and Tricks

1. **Quick Development Loop**
   ```bash
   cffa-test-quick && cffa-fix && cffa-test-cov
   ```

2. **Monitor Everything**
   ```bash
   # In one terminal
   cffa-dashboard
   
   # In another
   cffa-attach
   ```

3. **Dockerized Testing**
   ```bash
   cffa-build && cffa-docker-carenji
   ```

4. **Save Important Sessions**
   ```bash
   cffa-save important_feature_$(date +%Y%m%d)
   ```

5. **Chain Commands**
   ```bash
   fb-start && sleep 5 && cffa-carenji
   ```

## Troubleshooting

### Aliases Not Working
```bash
# Reload aliases
source ~/.bashrc

# Check if aliases file exists
ls -la /home/wojtek/dev/claude_code_agent_farm/.bash_aliases

# Manually source
source /home/wojtek/dev/claude_code_agent_farm/.bash_aliases
```

### Permission Issues
```bash
# Fix permissions
chmod +x /home/wojtek/dev/claude_code_agent_farm/scripts/*.sh
chmod +x /home/wojtek/dev/claude_code_agent_farm/.bash_aliases
```

### Completion Not Working
```bash
# Source completion file
source /home/wojtek/dev/claude_code_agent_farm/completion/cffa_completion.bash

# Check bash version (needs 4.0+)
bash --version
```

## Customization

To add your own aliases, create `~/.cffa_custom`:
```bash
# Custom aliases
alias cffa-dev='cffa-custom "Continue development where we left off"'
alias cffa-review='cffa-custom "Review and improve the current code"'

# Source in .bashrc
echo "source ~/.cffa_custom" >> ~/.bashrc
```

## Uninstall

To remove the aliases:
```bash
# Remove from .bashrc
sed -i '/Claude Flutter Firebase Agent aliases/,+1d' ~/.bashrc

# Remove from .zshrc if present
sed -i '/Claude Flutter Firebase Agent aliases/,+1d' ~/.zshrc

# Remove completion directory
rm -rf /home/wojtek/dev/claude_code_agent_farm/completion
```