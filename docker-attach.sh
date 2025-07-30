#!/bin/bash
# Attach to a running Claude Code Agent Farm container
# Usage: ./docker-attach.sh [container_number]

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get container number from argument
CONTAINER_NUM="${1:-1}"
CONTAINER_NAME="ccfarm-$CONTAINER_NUM"

# Check if container exists
if ! docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${RED}Error: Container '${CONTAINER_NAME}' is not running${NC}"
    echo ""
    echo "Running containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}" --filter "name=ccfarm-"
    exit 1
fi

echo -e "${BLUE}Attaching to container: ${GREEN}${CONTAINER_NAME}${NC}"
echo ""

# Get the project path from the container's environment
PROJECT_PATH=$(docker exec "$CONTAINER_NAME" printenv PROJECT_PATH 2>/dev/null || echo "/workspace")

# Get the UID of the workspace to exec as the correct user
WORKSPACE_UID=$(docker exec "$CONTAINER_NAME" stat -c %u "$PROJECT_PATH" 2>/dev/null || echo 1000)
WORKSPACE_GID=$(docker exec "$CONTAINER_NAME" stat -c %g "$PROJECT_PATH" 2>/dev/null || echo 1000)

# Set HOME environment variable to match the user
if [ "$WORKSPACE_UID" -eq 1000 ]; then
    USER_HOME="/home/node"
else
    USER_HOME="/home/hostuser"
fi

# Check if tmux session exists (multi-agent mode) or not (single-agent container mode)
TMUX_SESSION_EXISTS=$(docker exec "$CONTAINER_NAME" bash -c "tmux has-session -t claude_agents 2>/dev/null && echo true || echo false")

if [ "$TMUX_SESSION_EXISTS" = "false" ]; then
    echo -e "${YELLOW}Container is running in single-agent mode (no tmux session)${NC}"
    echo -e "${YELLOW}Attaching directly to container shell...${NC}"
    echo ""
    echo -e "${GREEN}Tips:${NC}"
    echo -e "• Use 'docker logs -f $CONTAINER_NAME' to see agent output"
    echo -e "• Exit with: Ctrl+D or 'exit'"
    echo ""
    # Attach to container shell
    docker exec -it -u "$WORKSPACE_UID:$WORKSPACE_GID" -e "HOME=$USER_HOME" "$CONTAINER_NAME" /bin/bash
else
    # Use docker exec to run view_agents.sh for an interactive menu as the correct user
    docker exec -it -u "$WORKSPACE_UID:$WORKSPACE_GID" -e "HOME=$USER_HOME" "$CONTAINER_NAME" /app/view_agents.sh
fi