#!/bin/bash
# Installation script for Claude Code Agent Farm dependencies

set -e

echo "🚀 Installing dependencies for Claude Code Agent Farm..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo -e "${RED}Please do not run this script as root${NC}"
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install with appropriate package manager
install_package() {
    if command_exists apt-get; then
        sudo apt-get update && sudo apt-get install -y "$@"
    elif command_exists yum; then
        sudo yum install -y "$@"
    elif command_exists brew; then
        brew install "$@"
    else
        echo -e "${RED}No supported package manager found${NC}"
        return 1
    fi
}

# Check and install system dependencies
echo -e "${YELLOW}Checking system dependencies...${NC}"

if ! command_exists git; then
    echo "Installing git..."
    install_package git
fi

if ! command_exists tmux; then
    echo "Installing tmux..."
    install_package tmux
fi

if ! command_exists node; then
    echo -e "${RED}Node.js is required but not installed.${NC}"
    echo "Please install Node.js 18+ from https://nodejs.org/"
    exit 1
fi

if ! command_exists python3; then
    echo -e "${RED}Python 3 is required but not installed.${NC}"
    echo "Please install Python 3.11+ from https://python.org/"
    exit 1
fi

# Install npm packages
echo -e "${YELLOW}Installing npm packages...${NC}"

# Check if tmux-composer is installed
if ! command_exists tmux-composer; then
    echo "Installing tmux-composer-cli..."
    npm install -g tmux-composer
else
    echo -e "${GREEN}✓ tmux-composer already installed${NC}"
fi

# Install claude-auto-resume
echo -e "${YELLOW}Installing claude-auto-resume...${NC}"
CLAUDE_AUTO_RESUME_DIR="$HOME/.local/share/claude-auto-resume"
if [ ! -d "$CLAUDE_AUTO_RESUME_DIR" ]; then
    git clone https://github.com/terryso/claude-auto-resume "$CLAUDE_AUTO_RESUME_DIR"
    mkdir -p "$HOME/.local/bin"
    ln -sf "$CLAUDE_AUTO_RESUME_DIR/claude-auto-resume.sh" "$HOME/.local/bin/claude-auto-resume"
    echo -e "${GREEN}✓ claude-auto-resume installed${NC}"
else
    echo "Updating claude-auto-resume..."
    cd "$CLAUDE_AUTO_RESUME_DIR" && git pull
    echo -e "${GREEN}✓ claude-auto-resume updated${NC}"
fi

# Install claude-code-generic-hooks
echo -e "${YELLOW}Installing claude-code-generic-hooks...${NC}"
HOOKS_DIR="$HOME/.config/claude-code/hooks"
if [ ! -d "$HOOKS_DIR" ]; then
    mkdir -p "$(dirname "$HOOKS_DIR")"
    git clone https://github.com/possibilities/claude-code-generic-hooks "$HOOKS_DIR"
    echo -e "${GREEN}✓ claude-code-generic-hooks installed${NC}"
else
    echo "Updating claude-code-generic-hooks..."
    cd "$HOOKS_DIR" && git pull
    echo -e "${GREEN}✓ claude-code-generic-hooks updated${NC}"
fi

# Check if PATH includes ~/.local/bin
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo -e "${YELLOW}Adding ~/.local/bin to PATH...${NC}"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc" 2>/dev/null || true
    echo -e "${YELLOW}Please restart your shell or run: source ~/.bashrc${NC}"
fi

# Install Python package
echo -e "${YELLOW}Installing Claude Code Agent Farm...${NC}"
pip install -e .

echo -e "${GREEN}✅ All dependencies installed successfully!${NC}"
echo ""
echo "You can now run:"
echo "  claude-code-agent-farm --help"
echo ""
echo "Make sure claude-auto-resume is in your PATH:"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""