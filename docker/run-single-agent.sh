#!/bin/bash
# Quick start script for Claude Single Agent Monitor

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
PROJECT_DIR=""
PROMPT_TEXT=""
PROMPT_FILE=""
BUILD_FIRST=false

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS] PROJECT_DIR"
    echo ""
    echo "Run Claude Single Agent Monitor on a project"
    echo ""
    echo "Arguments:"
    echo "  PROJECT_DIR          Path to your project directory"
    echo ""
    echo "Options:"
    echo "  -p, --prompt TEXT    Prompt text to use"
    echo "  -f, --file FILE      Path to prompt file"
    echo "  -b, --build          Build Docker image first"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  # With prompt.txt in project"
    echo "  $0 /path/to/project"
    echo ""
    echo "  # With custom prompt"
    echo "  $0 -p 'Fix all type errors' /path/to/project"
    echo ""
    echo "  # With prompt file"
    echo "  $0 -f my-prompt.txt /path/to/project"
    echo ""
    echo "  # Build and run"
    echo "  $0 -b /path/to/project"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--prompt)
            PROMPT_TEXT="$2"
            shift 2
            ;;
        -f|--file)
            PROMPT_FILE="$2"
            shift 2
            ;;
        -b|--build)
            BUILD_FIRST=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            ;;
        *)
            PROJECT_DIR="$1"
            shift
            ;;
    esac
done

# Validate project directory
if [[ -z "$PROJECT_DIR" ]]; then
    echo -e "${RED}Error: Project directory not specified${NC}"
    usage
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo -e "${RED}Error: Project directory does not exist: $PROJECT_DIR${NC}"
    exit 1
fi

# Get absolute path
PROJECT_DIR=$(cd "$PROJECT_DIR" && pwd)

# Check for Claude configuration
if [[ ! -d "$HOME/.config/claude" ]] && [[ ! -f "$HOME/.claude.json" ]]; then
    echo -e "${RED}Error: Claude configuration not found!${NC}"
    echo ""
    echo "Please configure Claude first by running: claude"
    echo ""
    echo "Configuration should be in one of:"
    echo "  - ~/.config/claude/ (modern)"
    echo "  - ~/.claude.json and ~/.claude/ (legacy)"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Build if requested
if [[ "$BUILD_FIRST" == "true" ]]; then
    echo -e "${YELLOW}Building Docker image...${NC}"
    bash "$SCRIPT_DIR/build-single.sh"
    echo ""
fi

# Check if image exists
if ! docker image inspect claude-single-agent:latest >/dev/null 2>&1; then
    echo -e "${YELLOW}Docker image not found. Building...${NC}"
    bash "$SCRIPT_DIR/build-single.sh"
    echo ""
fi

# Prepare environment
export PROJECT_DIR

# Handle prompt
if [[ -n "$PROMPT_TEXT" ]]; then
    export PROMPT_TEXT
    echo -e "${GREEN}Using prompt: ${PROMPT_TEXT:0:50}...${NC}"
elif [[ -n "$PROMPT_FILE" ]]; then
    if [[ ! -f "$PROMPT_FILE" ]]; then
        echo -e "${RED}Error: Prompt file not found: $PROMPT_FILE${NC}"
        exit 1
    fi
    # Copy prompt file to a known location
    cp "$PROMPT_FILE" "$PROJECT_DIR/.claude-prompt.txt"
    export PROMPT_FILE="/workspace/.claude-prompt.txt"
    echo -e "${GREEN}Using prompt file: $PROMPT_FILE${NC}"
elif [[ -f "$PROJECT_DIR/prompt.txt" ]]; then
    echo -e "${GREEN}Using default prompt file: prompt.txt${NC}"
else
    echo -e "${RED}Error: No prompt provided!${NC}"
    echo ""
    echo "Please provide one of:"
    echo "  - A prompt.txt file in your project"
    echo "  - Use -p 'your prompt text'"
    echo "  - Use -f /path/to/prompt.txt"
    exit 1
fi

# Display startup info
echo ""
echo "========================================"
echo -e "${GREEN}Starting Claude Single Agent Monitor${NC}"
echo "========================================"
echo "Project: $PROJECT_DIR"
echo "Image: claude-single-agent:latest"
echo ""
echo "To attach to tmux session:"
echo "  docker exec -it claude-single-agent-monitor tmux attach -t claude-agent"
echo ""
echo "To stop:"
echo "  Press Ctrl+C or run: docker-compose -f $SCRIPT_DIR/docker-compose-single.yml down"
echo "========================================"
echo ""

# Run with docker-compose
cd "$SCRIPT_DIR"
docker-compose -f docker-compose-single.yml up

# Cleanup
if [[ -f "$PROJECT_DIR/.claude-prompt.txt" ]]; then
    rm -f "$PROJECT_DIR/.claude-prompt.txt"
fi