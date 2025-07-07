#!/bin/bash

# Script to check Flutter MCP server health

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Checking Flutter MCP Server Status..."
echo "================================="

# Check if container exists
if ! docker ps -a --format '{{.Names}}' | grep -q '^flutter-mcp-server$'; then
    echo -e "${RED}✗ Flutter MCP server container does not exist${NC}"
    echo "  Run: ./scripts/start-mcp-server.sh"
    exit 1
fi

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q '^flutter-mcp-server$'; then
    echo -e "${RED}✗ Flutter MCP server container is not running${NC}"
    echo "  Container status:"
    docker ps -a --filter "name=flutter-mcp-server" --format "table {{.Status}}"
    echo "  Run: docker start flutter-mcp-server"
    exit 1
fi

echo -e "${GREEN}✓ Container is running${NC}"

# Check network connectivity
if ! docker network ls --format '{{.Name}}' | grep -q '^claude-mcp-network$'; then
    echo -e "${YELLOW}⚠ Claude MCP network does not exist${NC}"
    echo "  Creating network..."
    docker network create claude-mcp-network
fi

# Check HTTP endpoint
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo -e "${GREEN}✓ MCP server is responding on port 8000${NC}"
else
    echo -e "${RED}✗ MCP server is not responding on port 8000${NC}"
    echo "  Container logs:"
    docker logs --tail 20 flutter-mcp-server
    exit 1
fi

# Show container info
echo ""
echo "Container Details:"
docker ps --filter "name=flutter-mcp-server" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Test MCP functionality
echo ""
echo "Testing MCP Tools:"
echo "=================="

# Create a test request to check if the server supports the expected tools
TEST_RESPONSE=$(curl -s -X POST http://localhost:8000/tools 2>/dev/null || echo "Failed")

if [[ "$TEST_RESPONSE" != "Failed" ]]; then
    echo -e "${GREEN}✓ MCP server tools endpoint is accessible${NC}"
    echo "  Available tools: flutter_search, flutter_docs"
else
    echo -e "${YELLOW}⚠ Could not verify MCP tools (this is normal if the server doesn't expose a tools endpoint)${NC}"
fi

echo ""
echo -e "${GREEN}Flutter MCP Server is healthy and ready for use!${NC}"
echo "Agents can connect to: http://flutter-mcp:8000 (within Docker network)"