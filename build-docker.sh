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

# Copy Claude configuration and session data
CLAUDE_JSON_SRC="$HOME/.claude.json"
CLAUDE_JSON_DEST="./claude.json"
CLAUDE_DIR_SRC="$HOME/.claude"
CLAUDE_DIR_DEST="./.claude"

# Copy .claude.json if it exists
if [ -f "$CLAUDE_JSON_SRC" ]; then
    echo -e "${GREEN}Copying Claude configuration from $CLAUDE_JSON_SRC${NC}"
    cp "$CLAUDE_JSON_SRC" "$CLAUDE_JSON_DEST"
else
    echo -e "${YELLOW}Warning: Claude configuration not found at $CLAUDE_JSON_SRC${NC}"
    # Create empty file to prevent build failure
    echo '{}' > "$CLAUDE_JSON_DEST"
fi

# Copy .claude directory if it exists
if [ -d "$CLAUDE_DIR_SRC" ]; then
    echo -e "${GREEN}Copying Claude session data from $CLAUDE_DIR_SRC${NC}"
    # Remove existing copy if present
    rm -rf "$CLAUDE_DIR_DEST" 2>/dev/null || true
    # Copy preserving permissions
    cp -rp "$CLAUDE_DIR_SRC" "$CLAUDE_DIR_DEST"
else
    echo -e "${YELLOW}Warning: Claude session directory not found at $CLAUDE_DIR_SRC${NC}"
    echo -e "${YELLOW}The container will need manual Claude setup${NC}"
fi

# Claude will be installed via npm in the container

# Build the Docker image
echo -e "${GREEN}Building Docker image...${NC}"
docker build -t claude-code-agent-farm:flutter .

# Clean up temporary files
if [ -f "$CLAUDE_JSON_DEST" ]; then
    rm "$CLAUDE_JSON_DEST"
fi
if [ -d "$CLAUDE_DIR_DEST" ]; then
    rm -rf "$CLAUDE_DIR_DEST"
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