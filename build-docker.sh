#!/bin/bash
# Build script for Claude Code Agent Farm Docker image

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Claude Code Agent Farm Docker Image with Flutter${NC}"

# Check if Claude is configured on host (just for information)
if [ -f "$HOME/.claude.json" ] && [ -d "$HOME/.claude" ]; then
    echo -e "${GREEN}Claude configuration detected on host${NC}"
    echo -e "${GREEN}It will be mounted at runtime when you run containers${NC}"
else
    echo -e "${YELLOW}Warning: Claude configuration not found in your home directory${NC}"
    echo -e "${YELLOW}Make sure to configure Claude before running containers${NC}"
    echo -e "${YELLOW}Run 'claude' command to set up your configuration${NC}"
fi

# Build the Docker image
echo -e "${GREEN}Building Docker image...${NC}"
docker build -t claude-code-agent-farm:flutter .

echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "To run the container, use the simple wrapper script:"
echo -e "  ${YELLOW}./run-docker.sh \"Fix all Flutter type errors\" 5${NC}"
echo -e "  ${YELLOW}./run-docker.sh prompt.txt 3${NC}"
echo -e "  ${YELLOW}./run-docker.sh \"Add error handling\"${NC}"
echo ""
echo "Or run manually with Docker:"
echo "  docker run -it -v /path/to/project:/workspace claude-code-agent-farm:flutter"