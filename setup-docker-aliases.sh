#!/bin/bash
# Setup Docker aliases for Claude Code Agent Farm

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}Claude Code Agent Farm - Docker Alias Setup${NC}"
echo -e "${BLUE}===========================================${NC}"

# Detect shell
if [[ -n "$ZSH_VERSION" ]]; then
    SHELL_RC="$HOME/.zshrc"
    SHELL_NAME="zsh"
elif [[ -n "$BASH_VERSION" ]]; then
    SHELL_RC="$HOME/.bashrc"
    SHELL_NAME="bash"
else
    echo -e "${RED}Error: Unsupported shell. Only bash and zsh are supported.${NC}"
    exit 1
fi

echo -e "${GREEN}Detected shell: $SHELL_NAME${NC}"
echo -e "${GREEN}Config file: $SHELL_RC${NC}"

# Check if aliases already exist
if grep -q "ccfarm-build" "$SHELL_RC" 2>/dev/null; then
    echo -e "${YELLOW}Docker aliases already exist in $SHELL_RC${NC}"
    echo -e "${YELLOW}Skipping alias setup${NC}"
else
    echo -e "\n${BLUE}Adding Docker aliases...${NC}"
    
    # Add aliases to shell config
    cat >> "$SHELL_RC" << EOF

# Claude Code Agent Farm - Docker aliases
alias ccfarm-build='cd $SCRIPT_DIR && ./build-docker.sh'
alias ccfarm-run='cd $SCRIPT_DIR && ./run-docker.sh'
alias ccfarm-attach='cd $SCRIPT_DIR && ./docker-attach.sh'

# Claude Code Agent Farm - Stop all containers
alias ccfarm-stop='docker stop \$(docker ps -q --filter name=ccfarm-)'
alias ccfarm-stopremove='cd $SCRIPT_DIR && ./ccfarm-stopremove.sh'

# Claude Code Agent Farm - View all containers in tmux grid
alias ccfarm-agents="$SCRIPT_DIR/ccfarm-agents.sh"

# Claude Code Agent Farm - View logs for a specific container
alias ccfarm-logs='docker logs -f'

# Claude Code Agent Farm - List all running containers
alias ccfarm-ps='docker ps --filter name=ccfarm-'
EOF

    echo -e "${GREEN}✓ Docker aliases added to $SHELL_RC${NC}"
fi

# Check if cc alias exists
if ! grep -q 'alias cc=' "$SHELL_RC" 2>/dev/null; then
    echo -e "\n${YELLOW}Adding required cc alias...${NC}"
    echo 'alias cc="ENABLE_BACKGROUND_TASKS=1 claude --dangerously-skip-permissions"' >> "$SHELL_RC"
    echo -e "${GREEN}✓ cc alias added${NC}"
fi

echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "\n${YELLOW}To use the new aliases, either:${NC}"
echo -e "  1. Run: ${BLUE}source $SHELL_RC${NC}"
echo -e "  2. Open a new terminal"

echo -e "\n${GREEN}Available Docker aliases:${NC}"
echo -e "  ${BLUE}ccfarm-build${NC}      - Build/rebuild the Docker image"
echo -e "  ${BLUE}ccfarm-run${NC}        - Run containers (wrapper for run-docker.sh)"
echo -e "  ${BLUE}ccfarm-attach${NC}     - Attach to a running container"
echo -e "  ${BLUE}ccfarm-stop${NC}       - Stop all ccfarm containers"
echo -e "  ${BLUE}ccfarm-stopremove${NC} - Stop and remove all ccfarm containers"
echo -e "  ${BLUE}ccfarm-ps${NC}         - List all running ccfarm containers"
echo -e "  ${BLUE}ccfarm-logs${NC}       - View logs from a specific container"
echo -e "  ${BLUE}ccfarm-agents${NC}     - View all containers in tmux grid"

echo -e "\n${GREEN}Example usage:${NC}"
echo -e "  ${BLUE}ccfarm-build${NC}"
echo -e "  ${BLUE}ccfarm-run \"Fix all type errors\"${NC}"
echo -e "  ${BLUE}ccfarm-attach 1${NC}"
echo -e "  ${BLUE}ccfarm-logs ccfarm-1${NC}"