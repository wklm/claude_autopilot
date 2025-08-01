#!/bin/bash
# Setup script for Claude Flutter Agent bash aliases

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FLUTTER_AGENT_BIN="$SCRIPT_DIR/bin/claude-flutter-agent-python"
CFA_BIN="$SCRIPT_DIR/bin/cfa"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up Claude Flutter Agent aliases...${NC}"

# Aliases to add
ALIASES="
# Claude Flutter Agent aliases - now with simplified syntax!
alias cfa='$CFA_BIN'
alias cfa-attach='$FLUTTER_AGENT_BIN attach'
alias cfa-stop='$FLUTTER_AGENT_BIN stop'
alias cfa-config='$FLUTTER_AGENT_BIN show-config'
alias cfa-help='$FLUTTER_AGENT_BIN --help'

# Quick commands for common tasks - using new simplified syntax
alias cfa-fix='$CFA_BIN \"Fix the Flutter analyzer errors and ensure all tests pass\"'
alias cfa-test='$CFA_BIN \"Write comprehensive tests for the recent changes\"'
alias cfa-feature='$CFA_BIN @'  # Use: cfa-feature prompt.md
alias cfa-review='$CFA_BIN \"Review the code for best practices and suggest improvements\"'

# Carenji-specific shortcuts
alias cfa-carenji='cd /home/wojtek/dev/carenji && $CFA_BIN'
alias cfa-carenji-firebase='cd /home/wojtek/dev/carenji && $CFA_BIN \"Set up Firebase emulators and test the integration\"'
"

# Determine which shell config file to use
if [ -n "$ZSH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
else
    echo "Unsupported shell. Please add aliases manually."
    exit 1
fi

# Check if aliases already exist
if grep -q "Claude Flutter Agent aliases" "$SHELL_CONFIG" 2>/dev/null; then
    echo -e "${GREEN}Aliases already installed in $SHELL_CONFIG${NC}"
    echo "To update, remove the existing aliases section and run this script again."
else
    # Add aliases to shell config
    echo "$ALIASES" >> "$SHELL_CONFIG"
    echo -e "${GREEN}Aliases added to $SHELL_CONFIG${NC}"
fi

echo -e "\n${BLUE}Available aliases:${NC}"
echo "  cfa 'prompt'     - Run Claude Flutter Agent with prompt"
echo "  cfa @file.md     - Run with prompt from file"
echo "  cfa-attach       - Attach to running session"
echo "  cfa-stop         - Stop running session"
echo "  cfa-config       - Show configuration"
echo "  cfa-help         - Show help"
echo "  cfa-fix          - Fix Flutter analyzer errors"
echo "  cfa-test         - Write tests for recent changes"
echo "  cfa-feature file - Run with prompt file (e.g., cfa-feature todo.md)"
echo "  cfa-review       - Review code for best practices"
echo "  cfa-carenji      - Run in carenji directory"
echo "  cfa-carenji-firebase - Set up Firebase for carenji"

echo -e "\n${GREEN}Setup complete!${NC}"
echo "Run 'source $SHELL_CONFIG' or start a new terminal to use the aliases."