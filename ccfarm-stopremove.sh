#!/bin/bash
# Stop and remove all Claude Code Agent Farm Docker containers
# Usage: ./ccfarm-stopremove.sh

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Claude Code Agent Farm - Container Cleanup${NC}"
echo -e "${BLUE}==========================================${NC}"

# Find all ccfarm containers (running and stopped)
RUNNING_CONTAINERS=$(docker ps -q --filter "name=ccfarm-" 2>/dev/null || true)
ALL_CONTAINERS=$(docker ps -aq --filter "name=ccfarm-" 2>/dev/null || true)

# Count containers
RUNNING_COUNT=$(echo "$RUNNING_CONTAINERS" | grep -c . 2>/dev/null || echo 0)
TOTAL_COUNT=$(echo "$ALL_CONTAINERS" | grep -c . 2>/dev/null || echo 0)

if [ "$TOTAL_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No ccfarm containers found.${NC}"
    exit 0
fi

echo -e "Found ${BLUE}$RUNNING_COUNT${NC} running and ${BLUE}$TOTAL_COUNT${NC} total containers"
echo ""

# Stop running containers
if [ "$RUNNING_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}Stopping running containers...${NC}"
    for container_id in $RUNNING_CONTAINERS; do
        container_name=$(docker inspect --format='{{.Name}}' "$container_id" | sed 's/^\///')
        echo -e "  Stopping ${BLUE}$container_name${NC}..."
        docker stop "$container_id" >/dev/null
    done
    echo -e "${GREEN}✓ All running containers stopped${NC}"
else
    echo -e "${YELLOW}No running containers to stop${NC}"
fi

echo ""

# Remove all containers
echo -e "${YELLOW}Removing all containers...${NC}"
for container_id in $ALL_CONTAINERS; do
    container_name=$(docker inspect --format='{{.Name}}' "$container_id" 2>/dev/null | sed 's/^\///' || echo "unknown")
    echo -e "  Removing ${BLUE}$container_name${NC}..."
    docker rm "$container_id" >/dev/null 2>&1 || true
done

echo -e "${GREEN}✓ All containers removed${NC}"
echo ""

# Check for MCP server and stop it
MCP_REMOVED=false
if docker ps -a --format '{{.Names}}' | grep -q '^flutter-mcp-server$'; then
    echo -e "${YELLOW}Stopping Flutter MCP server...${NC}"
    docker stop flutter-mcp-server >/dev/null 2>&1 || true
    docker rm flutter-mcp-server >/dev/null 2>&1 || true
    echo -e "${GREEN}✓ Flutter MCP server removed${NC}"
    MCP_REMOVED=true
    echo ""
fi

# Clean up MCP network if no other containers are using it
if docker network ls --format '{{.Name}}' | grep -q '^claude-mcp-network$'; then
    # Check if any containers are still connected to the network
    CONNECTED=$(docker network inspect claude-mcp-network --format '{{len .Containers}}' 2>/dev/null || echo "0")
    if [ "$CONNECTED" -eq 0 ]; then
        echo -e "${YELLOW}Removing unused MCP network...${NC}"
        docker network rm claude-mcp-network >/dev/null 2>&1 || true
        echo -e "${GREEN}✓ MCP network removed${NC}"
        echo ""
    fi
fi

# Final summary
echo -e "${GREEN}Cleanup complete!${NC}"
echo -e "Removed ${BLUE}$TOTAL_COUNT${NC} container(s)"
if $MCP_REMOVED; then
    echo -e "Stopped Flutter MCP server"
fi

# Optional: Show disk space freed
if command -v docker &> /dev/null; then
    echo ""
    echo -e "${BLUE}Tip:${NC} To free up disk space from unused images, run:"
    echo -e "  ${YELLOW}docker image prune${NC}"
fi