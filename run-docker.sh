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
MCP_ENABLED=auto  # auto, true, or false

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
        --mcp)
            MCP_ENABLED=true
            shift
            ;;
        --no-mcp)
            MCP_ENABLED=false
            shift
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
    echo "  $0 --no-mcp \"Work without MCP\"  # Disable MCP server"
    echo ""
    echo "Examples:"
    echo "  $0 \"Implement Flutter best practices\""
    echo "  $0 flutter_tasks.txt"
    echo "  $0 -b 3 \"Add error handling\"  # 3 background containers"
    echo "  $0 --mcp \"Use Flutter docs\"  # Force MCP even for non-Flutter projects"
    exit 1
fi

# Get current directory as project path (absolute path)
PROJECT_PATH="$(pwd)"
# Get the real path to handle symlinks
PROJECT_PATH="$(realpath "$PROJECT_PATH")"

# Auto-detect Flutter project if MCP is set to auto
if [[ "$MCP_ENABLED" == "auto" ]]; then
    if [[ -f "$PROJECT_PATH/pubspec.yaml" ]]; then
        MCP_ENABLED=true
        echo -e "${GREEN}Flutter project detected - MCP will be enabled${NC}"
    else
        MCP_ENABLED=false
    fi
fi

# Function to ensure MCP server is running
ensure_mcp_running() {
    if [[ "$MCP_ENABLED" != "true" ]]; then
        return 0
    fi
    
    # Check if MCP network exists, create if not
    if ! docker network ls --format '{{.Name}}' | grep -q '^claude-mcp-network$'; then
        echo -e "${YELLOW}Creating Docker network for MCP...${NC}"
        docker network create claude-mcp-network >/dev/null 2>&1
    fi
    
    # Check if MCP server is already running
    if docker ps --format '{{.Names}}' | grep -q '^flutter-mcp-server$'; then
        echo -e "${GREEN}Flutter MCP server is already running${NC}"
        return 0
    fi
    
    # Check if stopped container exists
    if docker ps -a --format '{{.Names}}' | grep -q '^flutter-mcp-server$'; then
        echo -e "${YELLOW}Starting existing Flutter MCP server...${NC}"
        docker start flutter-mcp-server >/dev/null 2>&1
    else
        # Build and run MCP server
        echo -e "${YELLOW}Building and starting Flutter MCP server...${NC}"
        local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        
        # Build the image
        if ! docker build -t flutter-mcp:latest "$script_dir/docker/flutter-mcp" >/dev/null 2>&1; then
            echo -e "${RED}Failed to build Flutter MCP image${NC}"
            echo -e "${YELLOW}MCP will be disabled for this session${NC}"
            MCP_ENABLED=false
            return 1
        fi
        
        # Run the server
        docker run -d \
            --name flutter-mcp-server \
            --restart unless-stopped \
            --network claude-mcp-network \
            -p 8000:8000 \
            flutter-mcp:latest >/dev/null 2>&1
    fi
    
    # Wait for server to be ready
    echo -e "${YELLOW}Waiting for MCP server to be ready...${NC}"
    for i in {1..10}; do
        if curl -s http://localhost:8000 >/dev/null 2>&1; then
            echo -e "${GREEN}✓ Flutter MCP server is ready${NC}"
            return 0
        fi
        sleep 1
    done
    
    echo -e "${YELLOW}Warning: MCP server may not be fully ready${NC}"
    return 0
}

# Check if Claude configuration exists
CLAUDE_CONFIG_WARNING=false
if [ ! -f "$HOME/.claude.json" ] || [ ! -d "$HOME/.claude" ]; then
    echo -e "${YELLOW}Warning: Claude configuration not found in your home directory${NC}"
    echo -e "${YELLOW}The container may not work properly without Claude configuration${NC}"
    echo -e "${YELLOW}Run 'claude' command on your host to set up configuration first${NC}"
    echo ""
    CLAUDE_CONFIG_WARNING=true
fi

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
    
    # Mount to the same path as host to preserve Claude trust
    docker_args+=("-v" "$PROJECT_PATH:$PROJECT_PATH")
    # Pass the project path as environment variable
    docker_args+=("-e" "PROJECT_PATH=$PROJECT_PATH")
    
    # Add MCP configuration if enabled
    if [[ "$MCP_ENABLED" == "true" ]]; then
        docker_args+=("--network" "claude-mcp-network")
        docker_args+=("-e" "MCP_ENABLED=true")
        docker_args+=("-e" "MCP_SERVER_URL=http://flutter-mcp-server:8000")
    fi
    
    # Create parent directories in container to match host structure
    # This ensures the full path exists
    local parent_dir=$(dirname "$PROJECT_PATH")
    docker_args+=("-e" "PARENT_DIR=$parent_dir")
    
    # Mount Claude configuration from host (read-only for security)
    if [ -f "$HOME/.claude.json" ]; then
        docker_args+=("-v" "$HOME/.claude.json:/host-claude-config/.claude.json:ro")
    fi
    if [ -d "$HOME/.claude" ]; then
        docker_args+=("-v" "$HOME/.claude:/host-claude-config/.claude:ro")
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
    echo -e "MCP:        ${GREEN}$([ "$MCP_ENABLED" == "true" ] && echo "Enabled" || echo "Disabled")${NC}"
    echo -e "${BLUE}=================================================${NC}"
    
    # Start MCP server if needed
    ensure_mcp_running
    
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
    echo -e "MCP:     ${GREEN}$([ "$MCP_ENABLED" == "true" ] && echo "Enabled" || echo "Disabled")${NC}"
    
    if [ -f "$PROMPT" ]; then
        echo -e "Prompt:  ${GREEN}$PROMPT${NC} (file)"
    else
        echo -e "Prompt:  ${GREEN}\"$PROMPT\"${NC}"
    fi
    echo -e "${BLUE}================================${NC}"
    
    # Start MCP server if needed
    ensure_mcp_running
    
    run_container 1
fi