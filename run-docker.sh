#!/bin/bash
# Simple wrapper to run Claude Code Agent Farm Docker container
# Usage: ./run-docker.sh "prompt"
#    or: ./run-docker.sh prompt.txt
#    or: ./run-docker.sh -b N "prompt"  (background mode with N containers)

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker image exists
if ! docker images | grep -q "claude-code-agent-farm.*flutter"; then
    echo -e "${RED}Error: Docker image not found!${NC}"
    echo -e "${YELLOW}Please run ./build-docker.sh first${NC}"
    exit 1
fi

# Parse arguments
BACKGROUND_MODE=false
NUM_CONTAINERS=1
PROMPT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--background)
            BACKGROUND_MODE=true
            if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                NUM_CONTAINERS="$2"
                shift 2
            else
                echo -e "${RED}Error: -b/--background requires a number${NC}"
                exit 1
            fi
            ;;
        *)
            if [[ -z "$PROMPT" ]]; then
                PROMPT="$1"
                shift
            else
                echo -e "${RED}Error: Too many arguments${NC}"
                exit 1
            fi
            ;;
    esac
done

# Check if prompt was provided
if [[ -z "$PROMPT" ]]; then
    echo -e "${RED}Error: No prompt provided${NC}"
    echo ""
    echo "Usage:"
    echo "  $0 \"Fix all type errors\""
    echo "  $0 prompt.txt"
    echo "  $0 -b 5 \"Fix all type errors\"  # Run 5 containers in background"
    echo ""
    echo "Examples:"
    echo "  $0 \"Implement Flutter best practices\""
    echo "  $0 flutter_tasks.txt"
    echo "  $0 -b 3 \"Add error handling\"  # 3 background containers"
    exit 1
fi

# Get current directory as project path
PROJECT_PATH="$(pwd)"

# Function to run a single container
run_container() {
    local container_num=$1
    local container_name="ccfarm-$container_num"
    local docker_args=()
    
    if $BACKGROUND_MODE; then
        docker_args+=("-d" "--name" "$container_name")
        docker_args+=("-e" "CONTAINER_NUM=$container_num")
        docker_args+=("-e" "BACKGROUND_MODE=true")
    else
        docker_args+=("-it" "--rm")
    fi
    
    docker_args+=("-v" "$PROJECT_PATH:/workspace")
    
    # Mount Claude binary from host if it exists
    if command -v claude >/dev/null 2>&1; then
        CLAUDE_PATH=$(which claude)
        # If claude is in ~/.local/bin or similar, mount the actual binary
        if [[ -L "$CLAUDE_PATH" ]]; then
            CLAUDE_PATH=$(readlink -f "$CLAUDE_PATH")
        fi
        docker_args+=("-v" "$CLAUDE_PATH:/usr/local/bin/claude:ro")
    fi
    
    # Check if prompt is a file or text
    if [ -f "$PROMPT" ]; then
        # It's a file
        PROMPT_FILE="$(realpath "$PROMPT")"
        docker_args+=("-v" "$PROMPT_FILE:/prompt.txt:ro")
        docker run "${docker_args[@]}" \
            claude-code-agent-farm:flutter \
            --prompt-file /prompt.txt \
            --agents 1
    else
        # It's text
        docker_args+=("-e" "PROMPT_TEXT=$PROMPT")
        docker run "${docker_args[@]}" \
            claude-code-agent-farm:flutter \
            --agents 1
    fi
}

# Background mode
if $BACKGROUND_MODE; then
    echo -e "${BLUE}Claude Code Agent Farm - Flutter (Background Mode)${NC}"
    echo -e "${BLUE}=================================================${NC}"
    echo -e "Project:    ${GREEN}$PROJECT_PATH${NC}"
    echo -e "Containers: ${GREEN}$NUM_CONTAINERS${NC}"
    echo -e "Prompt:     ${GREEN}\"$PROMPT\"${NC}"
    echo -e "${BLUE}=================================================${NC}"
    
    # Stop and remove existing containers
    echo -e "${YELLOW}Cleaning up existing containers...${NC}"
    for i in $(seq 1 20); do
        docker stop ccfarm-$i 2>/dev/null || true
        docker rm ccfarm-$i 2>/dev/null || true
    done
    
    # Start containers
    echo -e "${GREEN}Starting $NUM_CONTAINERS containers...${NC}"
    for i in $(seq 1 $NUM_CONTAINERS); do
        echo -e "Starting container ${BLUE}ccfarm-$i${NC}..."
        run_container $i
    done
    
    echo -e "${GREEN}All containers started!${NC}"
    echo ""
    echo "View logs:"
    for i in $(seq 1 $NUM_CONTAINERS); do
        echo -e "  docker logs -f ccfarm-$i"
    done
    echo ""
    echo "Attach to a container:"
    echo -e "  ${YELLOW}./docker-attach.sh 1${NC}  # Attach to container 1"
    echo ""
    echo "Stop all containers:"
    echo -e "  ${YELLOW}docker stop \$(docker ps -q --filter name=ccfarm-)${NC}"
    
# Interactive mode
else
    echo -e "${BLUE}Claude Code Agent Farm - Flutter${NC}"
    echo -e "${BLUE}================================${NC}"
    echo -e "Project: ${GREEN}$PROJECT_PATH${NC}"
    echo -e "Agent:   ${GREEN}1${NC}"
    
    if [ -f "$PROMPT" ]; then
        echo -e "Prompt:  ${GREEN}$PROMPT${NC} (file)"
    else
        echo -e "Prompt:  ${GREEN}\"$PROMPT\"${NC}"
    fi
    echo -e "${BLUE}================================${NC}"
    
    run_container 1
fi