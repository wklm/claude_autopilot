#!/bin/bash
set -e

# Function to add timestamp to output
timestamp() {
    while IFS= read -r line; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
    done
}

# Function to display help
show_help() {
    echo "Claude Code Agent Farm - Docker Container"
    echo ""
    echo "Usage:"
    echo "  docker run -v /path/to/project:/workspace claude-farm [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help                Show this help message"
    echo "  --path PATH          Project path (default: /workspace)"
    echo "  --config CONFIG      Config file (default: /app/configs/flutter_config.json)"
    echo "  --agents N           Number of agents (default: 1)"
    echo "  --prompt-file FILE   Path to prompt file"
    echo "  --prompt-text TEXT   Direct prompt text"
    echo "  --auto-restart       Enable auto-restart (default: true)"
    echo ""
    echo "Environment Variables:"
    echo "  PROMPT_FILE          Path to prompt file"
    echo "  PROMPT_TEXT          Direct prompt text"
    echo "  CONFIG_FILE          Config file path"
    echo "  AGENTS               Number of agents"
    echo "  AUTO_RESTART         Enable auto-restart"
    echo ""
    echo "Examples:"
    echo "  # With prompt file mounted"
    echo "  docker run -v /my/project:/workspace -v /my/prompt.txt:/prompt.txt \\"
    echo "    claude-farm --prompt-file /prompt.txt"
    echo ""
    echo "  # With prompt text"
    echo "  docker run -v /my/project:/workspace \\"
    echo "    -e PROMPT_TEXT='Fix all type errors' claude-farm"
    echo ""
    echo "  # With custom config"
    echo "  docker run -v /my/project:/workspace -v /my/config.json:/config.json \\"
    echo "    claude-farm --config /config.json --agents 5"
}

# Check if Claude is available
check_claude() {
    if ! command -v claude &> /dev/null; then
        echo "Error: Claude Code CLI is not installed or not in PATH"
        echo "Please ensure Claude Code is installed in the container"
        exit 1
    fi
    
    # Check if Claude is configured
    if [ ! -f "$HOME/.claude.json" ]; then
        echo "Warning: Claude configuration not found at $HOME/.claude.json"
        echo "Claude may need to be configured before use"
    fi
}

# Activate virtual environment
source /home/claude/.venv/bin/activate

# Fix Flutter git warning
git config --global --add safe.directory /opt/flutter

# Default values
PROJECT_PATH="/workspace"
ARGS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --path)
            PROJECT_PATH="$2"
            ARGS+=("--path" "$2")
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            ARGS+=("--config" "$2")
            shift 2
            ;;
        --agents|-n)
            AGENTS="$2"
            ARGS+=("--agents" "$2")
            shift 2
            ;;
        --prompt-file)
            PROMPT_FILE="$2"
            ARGS+=("--prompt-file" "$2")
            shift 2
            ;;
        --prompt-text)
            # Create temporary prompt file from text
            TEMP_PROMPT="/tmp/prompt_$$.txt"
            echo "$2" > "$TEMP_PROMPT"
            ARGS+=("--prompt-file" "$TEMP_PROMPT")
            shift 2
            ;;
        --auto-restart)
            ARGS+=("--auto-restart")
            shift
            ;;
        --no-auto-restart)
            AUTO_RESTART="false"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Add default path if not specified
if [[ ! " ${ARGS[@]} " =~ " --path " ]]; then
    ARGS+=("--path" "$PROJECT_PATH")
fi

# Handle environment variables
# Add config first so it can be overridden by other arguments
if [[ -n "$CONFIG_FILE" ]] && [[ ! " ${ARGS[@]} " =~ " --config " ]]; then
    ARGS+=("--config" "$CONFIG_FILE")
fi

# Create prompt.txt file in workspace if prompt is provided
if [[ -n "$PROMPT_FILE" ]]; then
    # Copy the provided prompt file to workspace/prompt.txt
    cp "$PROMPT_FILE" "$PROJECT_PATH/prompt.txt"
elif [[ -n "$PROMPT_TEXT" ]]; then
    # Create prompt.txt from the provided text
    echo "$PROMPT_TEXT" > "$PROJECT_PATH/prompt.txt"
fi

# Note: We don't need to pass --prompt-file since the config will use prompt.txt

if [[ -n "$AGENTS" ]] && [[ ! " ${ARGS[@]} " =~ " --agents " ]]; then
    ARGS+=("--agents" "$AGENTS")
fi

if [[ "$AUTO_RESTART" == "true" ]] && [[ ! " ${ARGS[@]} " =~ " --auto-restart " ]]; then
    ARGS+=("--auto-restart")
fi

# Check if workspace exists and is mounted
if [[ ! -d "$PROJECT_PATH" ]]; then
    echo "Error: Project directory $PROJECT_PATH does not exist"
    echo "Please mount your project directory with: -v /path/to/project:/workspace"
    exit 1
fi

# Check if workspace is empty (not mounted)
if [[ -z "$(ls -A $PROJECT_PATH 2>/dev/null)" ]]; then
    echo "Warning: Project directory $PROJECT_PATH is empty"
    echo "Make sure you mounted your project correctly with: -v /path/to/project:/workspace"
fi

# Check Claude installation
check_claude

# Check if we're in background mode
if [[ "${BACKGROUND_MODE}" == "true" ]]; then
    CONTAINER_INFO="[Container ${CONTAINER_NUM:-?}]"
else
    CONTAINER_INFO=""
fi

# Function to run with timestamps in background mode
run_with_timestamps() {
    if [[ "${BACKGROUND_MODE}" == "true" ]]; then
        "$@" 2>&1 | timestamp
    else
        "$@"
    fi
}

# Display startup information
{
    echo "========================================"
    echo "Claude Code Agent Farm - Flutter Edition ${CONTAINER_INFO}"
    echo "========================================"
    echo "Project Path: $PROJECT_PATH"
    echo "Config: ${CONFIG_FILE:-/app/configs/flutter_config.json}"
    echo "Agents: ${AGENTS:-1}"
    echo "Auto-restart: ${AUTO_RESTART:-true}"
    if [[ -n "$PROMPT_FILE" ]]; then
        echo "Prompt File: $PROMPT_FILE"
    elif [[ -n "$PROMPT_TEXT" ]]; then
        echo "Prompt: $PROMPT_TEXT"
    fi
    echo "========================================"
    echo ""
} | { [[ "${BACKGROUND_MODE}" == "true" ]] && timestamp || cat; }

# Run Flutter doctor to show environment status
{
    echo "Flutter Environment Status:"
    flutter doctor -v
    echo "========================================"
    echo ""
} | { [[ "${BACKGROUND_MODE}" == "true" ]] && timestamp || cat; }

# Check if we should attach to existing tmux session
if [[ -t 0 ]] && tmux has-session -t claude_agents 2>/dev/null; then
    # If we're in interactive mode and tmux session exists, attach to it
    echo "Attaching to existing tmux session..." | { [[ "${BACKGROUND_MODE}" == "true" ]] && timestamp || cat; }
    exec tmux attach-session -t claude_agents
fi

# Execute the Claude Code Agent Farm
echo "Starting Claude Code Agent Farm..." | { [[ "${BACKGROUND_MODE}" == "true" ]] && timestamp || cat; }

# Add non-interactive flag for background mode
if [[ "${BACKGROUND_MODE}" == "true" ]]; then
    # Set environment variable to skip confirmations
    export CLAUDE_AGENT_FARM_NON_INTERACTIVE=1
    exec python /app/claude_code_agent_farm.py "${ARGS[@]}" 2>&1 | timestamp
else
    exec python /app/claude_code_agent_farm.py "${ARGS[@]}"
fi