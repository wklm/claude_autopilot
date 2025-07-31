#!/bin/bash
# Build script for Claude Flutter Firebase Agent Docker image

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🦋 Building Claude Flutter Firebase Agent for Carenji${NC}"
echo ""

# Check if Claude is configured on host (just for information)
if [ -f "$HOME/.config/claude/.claude.json" ] || [ -f "$HOME/.claude.json" ]; then
    echo -e "${GREEN}✓ Claude configuration detected on host${NC}"
    echo -e "${GREEN}  It will be mounted at runtime${NC}"
else
    echo -e "${YELLOW}⚠️  Claude configuration not found${NC}"
    echo -e "${YELLOW}  Make sure to configure Claude before running containers${NC}"
    echo -e "${YELLOW}  Run 'claude' command to set up your configuration${NC}"
fi

echo ""

# Build the Docker image
echo -e "${BLUE}Building Docker image...${NC}"
docker build -t claude-flutter-firebase-agent:latest .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build complete!${NC}"
    echo ""
    echo -e "${BLUE}To run the agent:${NC}"
    echo ""
    echo "1. Using docker-compose (recommended):"
    echo -e "   ${YELLOW}export CARENJI_PATH=/path/to/carenji${NC}"
    echo -e "   ${YELLOW}docker-compose up -d${NC}"
    echo ""
    echo "2. Using docker run:"
    echo -e "   ${YELLOW}docker run -it -v /path/to/carenji:/workspace claude-flutter-firebase-agent:latest${NC}"
    echo ""
    echo "3. Using helper scripts:"
    echo -e "   ${YELLOW}./scripts/start-carenji-dev.sh${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi