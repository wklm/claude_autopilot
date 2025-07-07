#!/bin/bash

# Test script for Flutter MCP integration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Flutter MCP Integration Test${NC}"
echo -e "${BLUE}=============================${NC}"
echo ""

# Test 1: Check if Docker is running
echo -e "${YELLOW}Test 1: Checking Docker...${NC}"
if docker info >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Docker is running${NC}"
else
    echo -e "${RED}✗ Docker is not running${NC}"
    exit 1
fi

# Test 2: Build MCP image
echo -e "\n${YELLOW}Test 2: Building Flutter MCP image...${NC}"
if docker build -t flutter-mcp:test "$PROJECT_ROOT/docker/flutter-mcp" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ MCP image built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build MCP image${NC}"
    exit 1
fi

# Test 3: Create network
echo -e "\n${YELLOW}Test 3: Creating Docker network...${NC}"
docker network create test-mcp-network >/dev/null 2>&1 || true
echo -e "${GREEN}✓ Network ready${NC}"

# Test 4: Start MCP server
echo -e "\n${YELLOW}Test 4: Starting MCP server...${NC}"
docker run -d --name test-mcp-server \
    --network test-mcp-network \
    -p 18000:8000 \
    flutter-mcp:test >/dev/null 2>&1

# Wait for server to start
sleep 5

# Check if server is running
if docker ps | grep -q test-mcp-server; then
    echo -e "${GREEN}✓ MCP server started${NC}"
else
    echo -e "${RED}✗ Failed to start MCP server${NC}"
    docker logs test-mcp-server
    docker rm -f test-mcp-server >/dev/null 2>&1
    docker network rm test-mcp-network >/dev/null 2>&1
    exit 1
fi

# Test 5: Check HTTP endpoint
echo -e "\n${YELLOW}Test 5: Testing HTTP endpoint...${NC}"
if curl -s http://localhost:18000 >/dev/null 2>&1; then
    echo -e "${GREEN}✓ MCP server responding on HTTP${NC}"
else
    echo -e "${RED}✗ MCP server not responding${NC}"
    docker logs test-mcp-server
fi

# Test 6: Test agent with MCP
echo -e "\n${YELLOW}Test 6: Testing agent with MCP connection...${NC}"

# Create a test prompt
TEST_PROMPT="List available Flutter state management packages"
echo "$TEST_PROMPT" > /tmp/test-prompt.txt

# Build agent image if needed
if ! docker images | grep -q claude-code-agent-farm; then
    echo "Building agent image..."
    docker build -t claude-code-agent-farm:latest "$PROJECT_ROOT" >/dev/null 2>&1
fi

# Run a test agent
docker run -d --name test-mcp-agent \
    --network test-mcp-network \
    -v "${HOME}/.claude.json:/home/claude/.claude.json:ro" \
    -v "${HOME}/.claude:/home/claude/.claude:ro" \
    -v "/tmp:/workspace" \
    -e "MCP_ENABLED=true" \
    -e "MCP_SERVER_URL=http://test-mcp-server:8000" \
    -e "PROJECT_PATH=/workspace" \
    -e "PROMPT_FILE=/workspace/test-prompt.txt" \
    -e "CLAUDE_DANGEROUSLY_SKIP_PERMISSIONS=1" \
    claude-code-agent-farm:latest \
    --config /app/configs/flutter_config.json \
    >/dev/null 2>&1

# Wait a bit
sleep 5

# Check if agent started
if docker ps | grep -q test-mcp-agent; then
    echo -e "${GREEN}✓ Agent started with MCP configuration${NC}"
    
    # Show first few lines of agent logs
    echo -e "\n${BLUE}Agent logs (first 10 lines):${NC}"
    docker logs test-mcp-agent 2>&1 | head -10
else
    echo -e "${RED}✗ Agent failed to start${NC}"
    docker logs test-mcp-agent 2>&1 | head -20
fi

# Cleanup
echo -e "\n${YELLOW}Cleaning up test resources...${NC}"
docker rm -f test-mcp-server test-mcp-agent >/dev/null 2>&1 || true
docker network rm test-mcp-network >/dev/null 2>&1 || true
rm -f /tmp/test-prompt.txt

echo -e "\n${GREEN}✓ Integration test completed!${NC}"
echo -e "\n${BLUE}To run agents with MCP:${NC}"
echo -e "  ${GREEN}./scripts/run-agents-with-mcp.sh --agents 3 --path /your/project${NC}"