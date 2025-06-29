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

# Get number of panes in agents window
PANE_COUNT=$(tmux list-panes -t "$SESSION:agents" 2>/dev/null | wc -l)

print_color "${CYAN}Claude Agent Farm Viewer${NC}\n"
print_color "${CYAN}Session: $SESSION${NC}\n"
if [ "$PANE_COUNT" -gt 0 ]; then
    print_color "${CYAN}Agents: $PANE_COUNT${NC}\n"
fi
print_color "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
print_color "\n"
print_color "1) Grid view    - See all agents at once\n"
print_color "2) Focus mode   - Cycle through individual agents\n"
print_color "3) Split view   - Monitor dashboard + agents side by side\n"
print_color "4) Quick attach - Jump right into the session\n"
print_color "\n"

# Font size tips for many panes
if [ "$PANE_COUNT" -ge 10 ]; then
    print_color "${YELLOW}💡 Tip: With $PANE_COUNT panes, reduce font size for better visibility:${NC}\n"
    print_color "   ${YELLOW}• macOS: Cmd + minus (-)${NC}\n"
    print_color "   ${YELLOW}• Linux/Windows: Ctrl + minus (-)${NC}\n"
    print_color "   ${YELLOW}• Reset: Cmd/Ctrl + 0${NC}\n"
    print_color "\n"
fi

print_color "${GREEN}Navigation tips:${NC}\n"
print_color "• Switch panes: Ctrl+B then arrow keys\n"
print_color "• Zoom pane: Ctrl+B then Z\n"
print_color "• Detach: Ctrl+B then D\n"
print_color "\n"

# Function to try adjusting terminal font size
adjust_font_size() {
    local action=$1
    # These escape sequences work on some terminals
    case "$action" in
        "smaller")
            # Try various terminal escape sequences for zooming out
            printf '\033]50;-*-*-*-*-*-*-8-*-*-*-*-*-*-*\007' 2>/dev/null  # xterm
            printf '\033]710;-*-fixed-medium-r-*-*-10-*-*-*-*-*-*-*\007' 2>/dev/null  # rxvt
            print_color "${YELLOW}Attempted to reduce font size (may not work on all terminals)${NC}\n"
            print_color "${YELLOW}Use your terminal's zoom out shortcut if needed${NC}\n"
            ;;
        "reset")
            printf '\033]50;-*-*-*-*-*-*-12-*-*-*-*-*-*-*\007' 2>/dev/null
            printf '\033]710;-*-fixed-medium-r-*-*-13-*-*-*-*-*-*-*\007' 2>/dev/null
            print_color "${GREEN}Attempted to reset font size${NC}\n"
            ;;
    esac
}

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
        
        # Auto-adjust font for many panes
        if [ "$PANE_COUNT" -ge 15 ]; then
            adjust_font_size "smaller"
            sleep 1
        fi
        
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
        # Split view - monitor dashboard and agents
        print_color "${GREEN}Creating split view...${NC}\n"
        # Check if both windows exist
        if tmux list-windows -t "$SESSION" | grep -q "controller:" && \
           tmux list-windows -t "$SESSION" | grep -q "agents:"; then
            tmux attach-session -t "$SESSION" \; \
                select-window -t controller \; \
                split-window -h -p 80 \; \
                select-window -t agents
        else
            print_color "${YELLOW}Warning: Split view requires both controller and agents windows${NC}\n"
            print_color "${YELLOW}Falling back to quick attach${NC}\n"
            tmux attach-session -t "$SESSION"
        fi
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
