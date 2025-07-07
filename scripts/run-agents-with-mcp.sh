#!/bin/bash

# Script to run Claude Code agents with Flutter MCP support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
NUM_AGENTS=3
PROJECT_PATH="$(pwd)"
PROMPT_FILE=""
PROMPT_TEXT=""
CONFIG_FILE="/app/configs/flutter_config.json"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --agents|-n)
            NUM_AGENTS="$2"
            shift 2
            ;;
        --path)
            PROJECT_PATH="$2"
            shift 2
            ;;
        --prompt-file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        --prompt-text)
            PROMPT_TEXT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --agents N, -n N     Number of agents to run (default: 3)"
            echo "  --path PATH         Project path (default: current directory)"
            echo "  --prompt-file FILE  Path to prompt file"
            echo "  --prompt-text TEXT  Direct prompt text"
            echo "  --config FILE       Config file (default: /app/configs/flutter_config.json)"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure network exists
if ! docker network ls --format '{{.Name}}' | grep -q '^claude-mcp-network$'; then
    echo "Creating Docker network for MCP communication..."
    docker network create claude-mcp-network
fi

# Start MCP server if not running
if ! docker ps --format '{{.Names}}' | grep -q '^flutter-mcp-server$'; then
    echo "Starting Flutter MCP server..."
    "$SCRIPT_DIR/start-mcp-server.sh"
    sleep 5  # Give server time to fully start
fi

# Check MCP server health
"$SCRIPT_DIR/check-mcp-health.sh" || exit 1

echo ""
echo "Starting $NUM_AGENTS Claude Code agents with Flutter MCP support..."
echo "Project path: $PROJECT_PATH"

# Stop and remove existing agent containers
for i in $(seq 1 $NUM_AGENTS); do
    container_name="ccfarm-mcp-$i"
    if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        echo "Stopping existing container: $container_name"
        docker stop "$container_name" >/dev/null 2>&1 || true
        docker rm "$container_name" >/dev/null 2>&1 || true
    fi
done

# Build the agent image if needed
if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q '^claude-code-agent-farm:latest$'; then
    echo "Building Claude Code Agent Farm image..."
    docker build -t claude-code-agent-farm:latest "$PROJECT_ROOT"
fi

# Start agent containers
for i in $(seq 1 $NUM_AGENTS); do
    container_name="ccfarm-mcp-$i"
    echo "Starting agent $i..."
    
    docker_args=(
        "-d"
        "--name" "$container_name"
        "--network" "claude-mcp-network"
        "-v" "${PROJECT_PATH}:${PROJECT_PATH}"
        "-v" "${HOME}/.claude.json:/home/claude/.claude.json:ro"
        "-v" "${HOME}/.claude:/home/claude/.claude:ro"
        "-e" "MCP_ENABLED=true"
        "-e" "MCP_SERVER_URL=http://flutter-mcp:8000"
        "-e" "PROJECT_PATH=${PROJECT_PATH}"
        "-e" "CLAUDE_DANGEROUSLY_SKIP_PERMISSIONS=1"
        "-e" "HOST_UID=$(id -u)"
        "-e" "HOST_GID=$(id -g)"
        "-e" "CONFIG_FILE=${CONFIG_FILE}"
        "-e" "AUTO_RESTART=true"
        "-e" "CONTAINER_NUM=$i"
        "-e" "BACKGROUND_MODE=true"
    )
    
    if [[ -n "$PROMPT_FILE" ]]; then
        docker_args+=("-e" "PROMPT_FILE=${PROMPT_FILE}")
    elif [[ -n "$PROMPT_TEXT" ]]; then
        docker_args+=("-e" "PROMPT_TEXT=${PROMPT_TEXT}")
    fi
    
    docker run "${docker_args[@]}" \
        claude-code-agent-farm:latest \
        --config "${CONFIG_FILE}"
done

echo ""
echo "All agents started with MCP support!"
echo ""
echo "To view agents in tmux grid:"
echo "  $PROJECT_ROOT/ccfarm-agents.sh"
echo ""
echo "To check agent logs:"
echo "  docker logs ccfarm-mcp-1"
echo ""
echo "To stop all agents:"
echo "  docker stop \$(docker ps -q --filter 'name=ccfarm-mcp-')"