#!/bin/bash
# Start Carenji Development Environment with Claude Agent

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🦋 Starting Carenji Development Environment${NC}"
echo ""

# Check if CARENJI_PATH is set
if [ -z "$CARENJI_PATH" ]; then
    # Try to find carenji relative to this script
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    POSSIBLE_PATH="$SCRIPT_DIR/../../carenji"
    
    if [ -d "$POSSIBLE_PATH" ] && [ -f "$POSSIBLE_PATH/pubspec.yaml" ]; then
        export CARENJI_PATH="$POSSIBLE_PATH"
        echo -e "${YELLOW}Using detected carenji path: $CARENJI_PATH${NC}"
    else
        echo -e "${RED}Error: CARENJI_PATH not set and carenji not found${NC}"
        echo "Please set CARENJI_PATH environment variable:"
        echo "  export CARENJI_PATH=/path/to/carenji"
        exit 1
    fi
fi

# Verify carenji project exists
if [ ! -f "$CARENJI_PATH/pubspec.yaml" ]; then
    echo -e "${RED}Error: No pubspec.yaml found at $CARENJI_PATH${NC}"
    echo "Please ensure CARENJI_PATH points to the carenji project"
    exit 1
fi

# Check if carenji project
if ! grep -q "name: carenji" "$CARENJI_PATH/pubspec.yaml"; then
    echo -e "${YELLOW}Warning: This doesn't appear to be the carenji project${NC}"
fi

echo -e "${GREEN}✓ Carenji project found at: $CARENJI_PATH${NC}"
echo ""

# Start Firebase emulators if available
echo "Checking Firebase emulators..."
if [ -f "$CARENJI_PATH/docker-compose.emulators.yml" ]; then
    echo -e "${GREEN}Starting Firebase emulators...${NC}"
    cd "$CARENJI_PATH"
    docker-compose -f docker-compose.emulators.yml up -d
    cd - > /dev/null
    echo -e "${GREEN}✓ Firebase emulators started${NC}"
else
    echo -e "${YELLOW}No Firebase emulator configuration found${NC}"
fi

echo ""

# Check if agent docker image exists
if ! docker images | grep -q "claude-flutter-firebase-agent"; then
    echo -e "${YELLOW}Building Claude Flutter Firebase Agent image...${NC}"
    docker-compose build
fi

# Start the agent
echo -e "${GREEN}Starting Claude Flutter Firebase Agent...${NC}"
docker-compose up -d claude-carenji-agent

# Wait for container to be ready
echo "Waiting for agent to start..."
sleep 5

# Show status
if docker ps | grep -q "claude-carenji-agent"; then
    echo -e "${GREEN}✓ Agent started successfully${NC}"
    echo ""
    echo "To view the agent:"
    echo "  docker exec -it claude-carenji-agent tmux attach-session -t claude-carenji"
    echo ""
    echo "To view logs:"
    echo "  docker-compose logs -f claude-carenji-agent"
    echo ""
    echo "To run Flutter with MCP:"
    echo "  cd $CARENJI_PATH"
    echo "  flutter run --debug --host-vmservice-port=8182 --dds-port=8181 --enable-vm-service --disable-service-auth-codes"
else
    echo -e "${RED}Error: Agent failed to start${NC}"
    echo "Check logs with: docker-compose logs claude-carenji-agent"
    exit 1
fi