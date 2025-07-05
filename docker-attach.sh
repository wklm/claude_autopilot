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

# Use docker exec to run view_agents.sh for an interactive menu
docker exec -it "$CONTAINER_NAME" /app/view_agents.sh