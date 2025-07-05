#!/bin/bash
# Build script for Claude Code Agent Farm Docker image
# This script copies the Claude configuration before building

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Claude Code Agent Farm Docker Image with Flutter${NC}"

# Copy Claude configuration if it exists
CLAUDE_CONFIG_SRC="$HOME/.claude.json"
CLAUDE_CONFIG_DEST="./claude.json"

if [ -f "$CLAUDE_CONFIG_SRC" ]; then
    echo -e "${GREEN}Copying Claude configuration from $CLAUDE_CONFIG_SRC${NC}"
    cp "$CLAUDE_CONFIG_SRC" "$CLAUDE_CONFIG_DEST"
else
    echo -e "${YELLOW}Warning: Claude configuration not found at $CLAUDE_CONFIG_SRC${NC}"
    echo -e "${YELLOW}The container will need manual Claude setup${NC}"
    # Create empty file to prevent build failure
    echo '{}' > "$CLAUDE_CONFIG_DEST"
fi

# Claude will be installed via npm in the container

# Build the Docker image
echo -e "${GREEN}Building Docker image...${NC}"
docker build -t claude-code-agent-farm:flutter .

# Clean up temporary config file
if [ -f "$CLAUDE_CONFIG_DEST" ]; then
    rm "$CLAUDE_CONFIG_DEST"
fi

echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "To run the container, use the simple wrapper script:"
echo -e "  ${YELLOW}./run-docker.sh \"Fix all Flutter type errors\" 5${NC}"
echo -e "  ${YELLOW}./run-docker.sh prompt.txt 3${NC}"
echo -e "  ${YELLOW}./run-docker.sh \"Add error handling\"${NC}"
echo ""
echo "Or run manually with Docker:"
echo "  docker run -it -v /path/to/project:/workspace claude-code-agent-farm:flutter"