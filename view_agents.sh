#!/usr/bin/env bash
# Simple viewer for Claude Agent Farm tmux session
# Compatible with both bash and zsh

SESSION="${1:-claude_agents}"

# Colors - using printf for better portability
print_color() {
    printf '%b' "$@"
}

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

# Check if session exists
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    print_color "${RED}Error: Session '$SESSION' not found${NC}\n"
    print_color "Usage: $0 [session-name]\n"
    exit 1
fi

print_color "${CYAN}Claude Agent Farm Viewer${NC}\n"
print_color "${CYAN}Session: $SESSION${NC}\n"
print_color "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
print_color "\n"
print_color "1) Grid view    - See all agents at once\n"
print_color "2) Focus mode   - Cycle through individual agents\n"
print_color "3) Split view   - Controller + agents side by side\n"
print_color "4) Quick attach - Jump right into the session\n"
print_color "\n"
print_color "${GREEN}Tip: Press Ctrl+B then D to detach${NC}\n"
print_color "\n"

# Portable prompt
if [ -t 0 ]; then
    print_color "Choose mode [1-4]: "
    read -r choice
else
    # Non-interactive mode (e.g., CI/SSH) - default to quick attach
    choice=4
fi

case "$choice" in
    1)
        # Grid view - show all panes at once
        print_color "${GREEN}Entering grid view...${NC}\n"
        tmux attach-session -t "$SESSION:agents"
        ;;
    2)
        # Focus mode - use tmux native navigation
        print_color "${GREEN}Entering focus mode...${NC}\n"
        print_color "${YELLOW}Navigate: Ctrl+B then arrow keys${NC}\n"
        print_color "${YELLOW}Zoom: Ctrl+B then Z${NC}\n"
        sleep 2
        tmux attach-session -t "$SESSION:agents"
        ;;
    3)
        # Split view - controller and agents
        print_color "${GREEN}Creating split view...${NC}\n"
        tmux attach-session -t "$SESSION" \; \
            select-window -t controller \; \
            split-window -h -p 80 \; \
            send-keys "tmux select-window -t agents" C-m
        ;;
    4)
        # Quick attach
        tmux attach-session -t "$SESSION"
        ;;
    *)
        print_color "${RED}Invalid choice${NC}\n"
        exit 1
        ;;
esac
