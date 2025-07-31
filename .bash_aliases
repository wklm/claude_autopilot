#!/bin/bash
# Claude Flutter Firebase Agent - Bash Aliases

# ========================================
# Agent Management
# ========================================

# Start the agent with default settings
alias cffa='claude-flutter-agent'

# Start agent with specific prompt
alias cffa-prompt='claude-flutter-agent run --prompt'

# Start agent for Carenji development
alias cffa-carenji='claude-flutter-agent run --project /home/wojtek/dev/carenji --prompt "Help with Carenji healthcare app development"'

# Monitor agent status
alias cffa-status='claude-flutter-agent status'

# View agent logs
alias cffa-logs='claude-flutter-agent logs --follow'

# Stop the agent
alias cffa-stop='claude-flutter-agent stop'

# ========================================
# Docker Management
# ========================================

# Build Docker image
alias cffa-build='docker build -t claude-flutter-firebase-agent:latest .'

# Run agent in Docker
alias cffa-docker='docker run -it --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e CLAUDE_PROJECT_PATH=/workspace \
  claude-flutter-firebase-agent:latest'

# Run agent in Docker with Carenji
alias cffa-docker-carenji='docker run -it --rm \
  -v /home/wojtek/dev/carenji:/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e CLAUDE_PROJECT_PATH=/workspace \
  -e CLAUDE_PROMPT_TEXT="Help with Carenji healthcare app development" \
  claude-flutter-firebase-agent:latest'

# Shell into agent container
alias cffa-shell='docker run -it --rm \
  -v $(pwd):/workspace \
  --entrypoint /bin/bash \
  claude-flutter-firebase-agent:latest'

# View Docker logs
alias cffa-docker-logs='docker logs -f $(docker ps -q -f ancestor=claude-flutter-firebase-agent:latest)'

# ========================================
# Tmux Session Management
# ========================================

# Attach to agent tmux session
alias cffa-attach='tmux attach-session -t claude-flutter-agent'

# List tmux sessions
alias cffa-sessions='tmux list-sessions | grep claude'

# Send command to agent session
alias cffa-send='tmux send-keys -t claude-flutter-agent:0'

# Capture current pane content
alias cffa-capture='tmux capture-pane -t claude-flutter-agent:0 -p'

# Kill agent tmux session
alias cffa-kill='tmux kill-session -t claude-flutter-agent'

# ========================================
# Firebase Emulator Management
# ========================================

# Start Firebase emulators for Carenji
alias fb-start='cd /home/wojtek/dev/carenji && firebase emulators:start'

# Start specific emulators
alias fb-auth='firebase emulators:start --only auth'
alias fb-firestore='firebase emulators:start --only firestore'
alias fb-storage='firebase emulators:start --only storage'

# View emulator UI
alias fb-ui='open http://localhost:4000'

# Kill all Firebase emulator processes
alias fb-kill='pkill -f "firebase.*emulators" || true'

# ========================================
# Testing
# ========================================

# Run all tests
alias cffa-test='pytest tests/ -v'

# Run unit tests only
alias cffa-test-unit='pytest tests/unit -v'

# Run integration tests
alias cffa-test-int='pytest tests/integration -v -m "not docker"'

# Run Docker tests
alias cffa-test-docker='pytest tests/ -v -m docker'

# Run Carenji-specific tests
alias cffa-test-carenji='pytest tests/ -v -m carenji'

# Run tests with coverage
alias cffa-test-cov='pytest tests/ --cov=claude_code_agent_farm --cov-report=html && open htmlcov/index.html'

# Run quick tests (no Docker, no slow tests)
alias cffa-test-quick='pytest tests/ -v -m "not docker and not slow"'

# ========================================
# Development Shortcuts
# ========================================

# CD to project directory
alias cdcffa='cd /home/wojtek/dev/claude_code_agent_farm'

# CD to Carenji project
alias cdcarenji='cd /home/wojtek/dev/carenji'

# Edit agent configuration
alias cffa-config='${EDITOR:-vim} ~/.claude-flutter-agent/config.yaml'

# View agent settings
alias cffa-settings='cat ~/.claude-flutter-agent/config.yaml'

# Edit agent prompts
alias cffa-prompts='${EDITOR:-vim} /home/wojtek/dev/claude_code_agent_farm/prompts/'

# ========================================
# Flutter & Firebase Shortcuts
# ========================================

# Flutter shortcuts
alias fl='flutter'
alias flr='flutter run'
alias flt='flutter test'
alias fla='flutter analyze'
alias flc='flutter clean'
alias flpg='flutter pub get'
alias flpu='flutter pub upgrade'
alias fldr='flutter doctor'

# Flutter with coverage
alias fltc='flutter test --coverage && genhtml coverage/lcov.info -o coverage/html && open coverage/html/index.html'

# Firebase shortcuts
alias fbs='firebase serve'
alias fbd='firebase deploy'
alias fbdo='firebase deploy --only'
alias fbl='firebase list'
alias fbu='firebase use'

# ========================================
# Git Shortcuts for Agent Development
# ========================================

# Git status with agent files highlighted
alias cffa-git='git status --porcelain | grep -E "(agent|monitor|carenji)"'

# Commit agent changes
alias cffa-commit='git add -A && git commit -m "feat(agent): "'

# Push agent branch
alias cffa-push='git push origin $(git branch --show-current)'

# ========================================
# Monitoring & Debugging
# ========================================

# Watch agent status
alias cffa-watch='watch -n 2 "claude-flutter-agent status"'

# Tail Python logs
alias cffa-tail='tail -f ~/.claude-flutter-agent/logs/*.log'

# Check if agent is running
alias cffa-check='ps aux | grep -E "(claude-flutter-agent|flutter_agent_monitor)" | grep -v grep'

# Resource usage
alias cffa-top='htop -F claude-flutter-agent'

# ========================================
# Utility Functions
# ========================================

# Function to start agent with custom prompt
cffa-custom() {
    local prompt="${1:-Help with Flutter development}"
    claude-flutter-agent run --prompt "$prompt"
}

# Function to run agent for specific time
cffa-timed() {
    local duration="${1:-3600}"  # Default 1 hour
    timeout $duration claude-flutter-agent run
}

# Function to restart agent
cffa-restart() {
    claude-flutter-agent stop
    sleep 2
    claude-flutter-agent run
}

# Function to backup agent logs
cffa-backup-logs() {
    local backup_dir="$HOME/claude-agent-logs-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    cp -r ~/.claude-flutter-agent/logs/* "$backup_dir/"
    echo "Logs backed up to: $backup_dir"
}

# Function to clean up old sessions
cffa-cleanup() {
    # Kill any orphaned tmux sessions
    tmux list-sessions 2>/dev/null | grep -E "claude|flutter" | cut -d: -f1 | xargs -I {} tmux kill-session -t {} 2>/dev/null || true
    
    # Clean up old logs (older than 7 days)
    find ~/.claude-flutter-agent/logs -type f -mtime +7 -delete 2>/dev/null || true
    
    echo "Cleanup complete"
}

# Function to show agent help
cffa-help() {
    echo "Claude Flutter Firebase Agent - Quick Reference"
    echo "=============================================="
    echo "Start/Stop:"
    echo "  cffa              - Start agent"
    echo "  cffa-carenji      - Start for Carenji development"
    echo "  cffa-stop         - Stop agent"
    echo "  cffa-restart      - Restart agent"
    echo ""
    echo "Monitoring:"
    echo "  cffa-status       - Check status"
    echo "  cffa-logs         - View logs"
    echo "  cffa-attach       - Attach to tmux session"
    echo "  cffa-watch        - Watch status (live)"
    echo ""
    echo "Testing:"
    echo "  cffa-test         - Run all tests"
    echo "  cffa-test-quick   - Run quick tests"
    echo "  cffa-test-cov     - Run with coverage"
    echo ""
    echo "Docker:"
    echo "  cffa-docker       - Run in Docker"
    echo "  cffa-build        - Build Docker image"
    echo ""
    echo "Firebase:"
    echo "  fb-start          - Start emulators"
    echo "  fb-kill           - Kill emulators"
    echo ""
    echo "For more commands, see: ~/.bash_aliases"
}

# ========================================
# Environment Setup
# ========================================

# Add agent bin to PATH if not already there
if [[ ":$PATH:" != *":/home/wojtek/dev/claude_code_agent_farm/bin:"* ]]; then
    export PATH="$PATH:/home/wojtek/dev/claude_code_agent_farm/bin"
fi

# Set default editor for agent files
export CLAUDE_EDITOR="${EDITOR:-vim}"

# Set default agent configuration
export CLAUDE_AGENT_CONFIG="${CLAUDE_AGENT_CONFIG:-$HOME/.claude-flutter-agent/config.yaml}"

# Enable agent command completion (if completion file exists)
if [ -f /home/wojtek/dev/claude_code_agent_farm/completion/cffa_completion.bash ]; then
    source /home/wojtek/dev/claude_code_agent_farm/completion/cffa_completion.bash
fi

# ========================================
# Colored Output
# ========================================

# Colors for terminal output
export CFFA_COLOR_RED='\033[0;31m'
export CFFA_COLOR_GREEN='\033[0;32m'
export CFFA_COLOR_YELLOW='\033[0;33m'
export CFFA_COLOR_BLUE='\033[0;34m'
export CFFA_COLOR_RESET='\033[0m'

# Colored status function
cffa-status-color() {
    local status=$(claude-flutter-agent status 2>/dev/null)
    case "$status" in
        *"RUNNING"*|*"WORKING"*)
            echo -e "${CFFA_COLOR_GREEN}✓ Agent is running${CFFA_COLOR_RESET}"
            ;;
        *"ERROR"*)
            echo -e "${CFFA_COLOR_RED}✗ Agent error${CFFA_COLOR_RESET}"
            ;;
        *"USAGE_LIMIT"*)
            echo -e "${CFFA_COLOR_YELLOW}⚠ Usage limit reached${CFFA_COLOR_RESET}"
            ;;
        *)
            echo -e "${CFFA_COLOR_BLUE}○ Agent is stopped${CFFA_COLOR_RESET}"
            ;;
    esac
}

# Show welcome message when aliases are loaded
echo "Claude Flutter Firebase Agent aliases loaded. Type 'cffa-help' for quick reference."