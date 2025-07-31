#!/bin/bash
# Setup script for Claude Flutter Firebase Agent bash aliases

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ALIASES_FILE="$PROJECT_DIR/.bash_aliases"
BASHRC="$HOME/.bashrc"
ZSHRC="$HOME/.zshrc"

echo "Setting up Claude Flutter Firebase Agent aliases..."

# Function to add source line to shell config
add_to_shell_config() {
    local config_file="$1"
    local source_line="source $ALIASES_FILE"
    local marker="# Claude Flutter Firebase Agent aliases"
    
    if [ -f "$config_file" ]; then
        # Check if already added
        if grep -q "$marker" "$config_file"; then
            echo "✓ Aliases already configured in $config_file"
        else
            echo "" >> "$config_file"
            echo "$marker" >> "$config_file"
            echo "$source_line" >> "$config_file"
            echo "✓ Added aliases to $config_file"
        fi
    fi
}

# Make aliases file executable
chmod +x "$ALIASES_FILE"

# Add to bash configuration
if [ -f "$BASHRC" ]; then
    add_to_shell_config "$BASHRC"
fi

# Add to zsh configuration if using zsh
if [ -f "$ZSHRC" ]; then
    add_to_shell_config "$ZSHRC"
fi

# Create completion directory
COMPLETION_DIR="$PROJECT_DIR/completion"
mkdir -p "$COMPLETION_DIR"

# Create bash completion file
cat > "$COMPLETION_DIR/cffa_completion.bash" << 'EOF'
# Bash completion for Claude Flutter Firebase Agent

_cffa_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Main commands
    opts="run status stop logs attach test build docker help"
    
    case "${prev}" in
        cffa|claude-flutter-agent)
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            return 0
            ;;
        run)
            local run_opts="--prompt --project --max-runs --monitor-interval"
            COMPREPLY=( $(compgen -W "${run_opts}" -- ${cur}) )
            return 0
            ;;
        test)
            local test_opts="unit integration e2e docker carenji firebase quick coverage"
            COMPREPLY=( $(compgen -W "${test_opts}" -- ${cur}) )
            return 0
            ;;
        logs)
            local log_opts="--follow --tail --since"
            COMPREPLY=( $(compgen -W "${log_opts}" -- ${cur}) )
            return 0
            ;;
        --project)
            # Complete with directories
            COMPREPLY=( $(compgen -d -- ${cur}) )
            return 0
            ;;
    esac
    
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
}

# Register completions
complete -F _cffa_completion cffa
complete -F _cffa_completion claude-flutter-agent

# Completion for test command
_cffa_test_completion() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local opts="all unit integration e2e docker carenji firebase quick coverage"
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
}

complete -F _cffa_test_completion cffa-test
EOF

echo "✓ Created bash completion file"

# Create a quick reference card
cat > "$PROJECT_DIR/ALIASES.md" << 'EOF'
# Claude Flutter Firebase Agent - Bash Aliases Reference

## Quick Start
```bash
source ~/.bashrc  # or ~/.zshrc
cffa-help        # Show quick reference
```

## Common Commands

### Agent Control
- `cffa` - Start the agent with default settings
- `cffa-carenji` - Start agent for Carenji development
- `cffa-stop` - Stop the agent
- `cffa-restart` - Restart the agent
- `cffa-status` - Check agent status
- `cffa-logs` - View agent logs (with follow)

### Tmux Session
- `cffa-attach` - Attach to agent tmux session
- `cffa-capture` - Capture current pane content
- `cffa-send <text>` - Send text to agent session

### Docker
- `cffa-build` - Build Docker image
- `cffa-docker` - Run agent in Docker
- `cffa-docker-carenji` - Run in Docker for Carenji
- `cffa-shell` - Shell into agent container

### Testing
- `cffa-test` - Run all tests
- `cffa-test-unit` - Run unit tests only
- `cffa-test-quick` - Run quick tests (no Docker/slow)
- `cffa-test-cov` - Run tests with coverage report
- `cffa-test-carenji` - Run Carenji-specific tests

### Firebase Emulators
- `fb-start` - Start all Firebase emulators
- `fb-kill` - Kill all emulator processes
- `fb-ui` - Open emulator UI in browser

### Flutter Shortcuts
- `fl` - flutter
- `flr` - flutter run
- `flt` - flutter test
- `fla` - flutter analyze
- `flc` - flutter clean
- `flpg` - flutter pub get

### Utilities
- `cffa-cleanup` - Clean up old sessions and logs
- `cffa-backup-logs` - Backup agent logs
- `cffa-watch` - Watch agent status (live update)
- `cffa-custom "prompt"` - Start with custom prompt

## Functions

### Start with custom prompt
```bash
cffa-custom "Implement user authentication for Carenji"
```

### Run agent for specific duration
```bash
cffa-timed 3600  # Run for 1 hour
```

### Check colored status
```bash
cffa-status-color
```

## Environment Variables
- `CLAUDE_AGENT_CONFIG` - Path to agent configuration
- `CLAUDE_EDITOR` - Editor for config files
- `CLAUDE_PROJECT_PATH` - Default project path

## Tips
1. Use `cffa-test-quick` for rapid development feedback
2. Run `cffa-cleanup` weekly to remove old logs
3. Use `cffa-docker` for isolated testing environments
4. Check `cffa-status-color` in your prompt for agent status
EOF

echo "✓ Created aliases reference at $PROJECT_DIR/ALIASES.md"

# Summary
echo ""
echo "=========================================="
echo "Setup complete! 🎉"
echo ""
echo "To start using the aliases:"
echo "  source ~/.bashrc    # For bash"
echo "  source ~/.zshrc     # For zsh"
echo ""
echo "Or start a new terminal session."
echo ""
echo "Type 'cffa-help' for quick reference."
echo "See $PROJECT_DIR/ALIASES.md for full documentation."
echo "=========================================="

# Offer to source immediately
read -p "Would you like to load the aliases now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    source "$ALIASES_FILE"
    echo "✓ Aliases loaded! Type 'cffa-help' to get started."
fi