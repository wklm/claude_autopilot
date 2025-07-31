#!/bin/bash
set -e

# Function to display help
show_help() {
    echo "Claude Single Agent Monitor - Docker Container"
    echo ""
    echo "Usage:"
    echo "  docker run -v /path/to/project:/workspace claude-single-agent [OPTIONS]"
    echo ""
    echo "Environment Variables:"
    echo "  PROJECT_PATH          Project path (default: /workspace)"
    echo "  PROMPT_FILE          Path to prompt file"
    echo "  PROMPT_TEXT          Direct prompt text"
    echo "  WAIT_ON_LIMIT        Wait when usage limit hit (default: true)"
    echo "  RESTART_ON_COMPLETE  Restart when task completes (default: true)"
    echo "  RESTART_ON_ERROR     Restart on errors (default: true)"
    echo "  CHECK_INTERVAL       Seconds between checks (default: 5)"
    echo "  IDLE_TIMEOUT         Seconds before idle (default: 300)"
    echo "  TMUX_SESSION         tmux session name (default: claude-agent)"
    echo ""
    echo "Examples:"
    echo "  # With prompt file"
    echo "  docker run -v /my/project:/workspace \\"
    echo "    -v /my/prompt.txt:/prompt.txt \\"
    echo "    -e PROMPT_FILE=/prompt.txt claude-single-agent"
    echo ""
    echo "  # With prompt text"
    echo "  docker run -v /my/project:/workspace \\"
    echo "    -e PROMPT_TEXT='Fix all type errors' claude-single-agent"
    echo ""
    echo "  # With Claude config mounted"
    echo "  docker run -v /my/project:/workspace \\"
    echo "    -v ~/.config/claude:/home/claude/.config/claude:ro \\"
    echo "    -e PROMPT_FILE=/workspace/prompt.txt claude-single-agent"
}

# Handle help flag
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Check if Claude configuration is mounted
check_claude_config() {
    if [[ ! -f "/home/claude/.config/claude/.claude.json" ]] && \
       [[ ! -f "/home/claude/.claude.json" ]] && \
       [[ ! -f "$HOME/.claude.json" ]]; then
        echo "Error: Claude configuration not found!"
        echo ""
        echo "Please mount your Claude configuration:"
        echo "  -v ~/.config/claude:/home/claude/.config/claude:ro"
        echo ""
        echo "Or if using legacy config location:"
        echo "  -v ~/.claude.json:/home/claude/.claude.json:ro"
        echo "  -v ~/.claude:/home/claude/.claude:ro"
        echo ""
        echo "First configure Claude on your host system by running 'claude'"
        return 1
    fi
    return 0
}

# Set default values
: "${PROJECT_PATH:=/workspace}"
: "${WAIT_ON_LIMIT:=true}"
: "${RESTART_ON_COMPLETE:=true}"
: "${RESTART_ON_ERROR:=true}"
: "${CHECK_INTERVAL:=5}"
: "${IDLE_TIMEOUT:=300}"
: "${TMUX_SESSION:=claude-agent}"

# Export for the Python CLI to pick up
export PROJECT_PATH
export WAIT_ON_LIMIT
export RESTART_ON_COMPLETE
export RESTART_ON_ERROR
export CHECK_INTERVAL
export IDLE_TIMEOUT
export TMUX_SESSION

# Check if project directory exists
if [[ ! -d "$PROJECT_PATH" ]]; then
    echo "Error: Project directory $PROJECT_PATH does not exist"
    echo "Please mount your project directory with: -v /path/to/project:/workspace"
    exit 1
fi

# Check if project directory is empty (not properly mounted)
if [[ -z "$(ls -A "$PROJECT_PATH" 2>/dev/null)" ]]; then
    echo "Warning: Project directory $PROJECT_PATH is empty"
    echo "Make sure you mounted your project correctly"
fi

# Handle prompt file/text
if [[ -n "$PROMPT_FILE" ]] && [[ -f "$PROMPT_FILE" ]]; then
    echo "Using prompt file: $PROMPT_FILE"
elif [[ -n "$PROMPT_TEXT" ]]; then
    echo "Using prompt text: ${PROMPT_TEXT:0:50}..."
elif [[ -f "$PROJECT_PATH/prompt.txt" ]]; then
    echo "Using default prompt file: $PROJECT_PATH/prompt.txt"
    export PROMPT_FILE="$PROJECT_PATH/prompt.txt"
else
    echo "Error: No prompt provided!"
    echo "Please provide either:"
    echo "  - PROMPT_FILE environment variable"
    echo "  - PROMPT_TEXT environment variable"
    echo "  - A prompt.txt file in your project directory"
    exit 1
fi

# Check Claude configuration
if ! check_claude_config; then
    exit 1
fi

# Display startup information
echo "========================================"
echo "Claude Single Agent Monitor"
echo "========================================"
echo "Project Path: $PROJECT_PATH"
echo "Wait on Limit: $WAIT_ON_LIMIT"
echo "Restart on Complete: $RESTART_ON_COMPLETE"
echo "Restart on Error: $RESTART_ON_ERROR"
echo "Check Interval: ${CHECK_INTERVAL}s"
echo "Idle Timeout: ${IDLE_TIMEOUT}s"
echo "tmux Session: $TMUX_SESSION"
echo "========================================"
echo ""

# Change to project directory
cd "$PROJECT_PATH"

# Create a wrapper function to handle signals
handle_signal() {
    echo "Received shutdown signal, cleaning up..."
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    exit 0
}

# Trap signals
trap handle_signal SIGTERM SIGINT

# Execute the command passed to docker run, or default command
if [[ $# -gt 0 ]]; then
    exec "$@"
else
    # Run the single agent monitor
    exec python -m claude_code_agent_farm.single_agent_cli
fi