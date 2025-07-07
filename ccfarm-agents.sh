#!/bin/bash
# View all Claude Code Agent Farm containers in a tmux grid
# Each pane shows the focused view of agents in that container
# Usage: ./ccfarm-agents.sh [-m] [-f]
#   -m: Monitor mode - automatically recreate session when containers change
#   -f: Force mode - recreate session even if it exists

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Session name for the overview
SESSION="ccfarm-overview"

# Parse command line arguments
MONITOR_MODE=false
FORCE_MODE=false

while getopts "mfh" opt; do
    case $opt in
        m)
            MONITOR_MODE=true
            ;;
        f)
            FORCE_MODE=true
            ;;
        h)
            echo "Usage: $0 [-m] [-f]"
            echo "  -m: Monitor mode - automatically recreate session when containers change"
            echo "  -f: Force mode - recreate session even if it exists"
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

echo -e "${BLUE}Claude Code Agent Farm - Multi-Container Viewer${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""

# Find all running ccfarm containers (including MCP-enabled ones)
RUNNING_CONTAINERS=$(docker ps --format "{{.Names}}" | grep -E "^ccfarm-" 2>/dev/null | sort -V || true)
CONTAINER_COUNT=$(echo "$RUNNING_CONTAINERS" | grep -c . 2>/dev/null || echo 0)

if [ "$CONTAINER_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No running ccfarm containers found.${NC}"
    echo -e "${YELLOW}Start containers first with: ./run-docker.sh -b N${NC}"
    exit 0
fi

echo -e "Found ${GREEN}$CONTAINER_COUNT${NC} running container(s):"
for container in $RUNNING_CONTAINERS; do
    echo -e "  ${CYAN}$container${NC}"
done
echo ""

# Function to get containers from existing session
get_session_containers() {
    # Get the stored container list from tmux environment variable
    tmux show-environment -t "$SESSION" CCFARM_CONTAINERS 2>/dev/null | cut -d= -f2 | tr ',' '\n' | sort -V || echo ""
}

# Check if session already exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
    if [ "$FORCE_MODE" = true ]; then
        echo -e "${YELLOW}Force mode: Killing existing session '$SESSION'...${NC}"
        tmux kill-session -t "$SESSION" 2>/dev/null || true
    else
        # Get containers from existing session
        SESSION_CONTAINERS=$(get_session_containers)
        
        # Compare with current containers
        if [ "$SESSION_CONTAINERS" = "$RUNNING_CONTAINERS" ]; then
            echo -e "${GREEN}Session '$SESSION' exists with matching containers.${NC}"
            echo -e "${GREEN}Attaching to existing session...${NC}"
            sleep 1
            tmux attach-session -t "$SESSION"
            exit 0
        else
            echo -e "${YELLOW}Session '$SESSION' exists but containers have changed.${NC}"
            echo -e "${YELLOW}Previous containers in session:${NC}"
            if [ -z "$SESSION_CONTAINERS" ]; then
                echo -e "  ${CYAN}(unable to determine)${NC}"
            else
                for container in $SESSION_CONTAINERS; do
                    echo -e "  ${CYAN}$container${NC}"
                done
            fi
            echo ""
            echo -e "${YELLOW}Recreating session with current containers...${NC}"
            tmux kill-session -t "$SESSION" 2>/dev/null || true
            sleep 1
        fi
    fi
fi

# Function to create the tmux session with current containers
create_session() {
    local containers="$1"
    
    echo -e "${GREEN}Creating tmux session '$SESSION'...${NC}"
    
    # Store the container list in tmux environment for later comparison
    local container_list=$(echo "$containers" | tr '\n' ',' | sed 's/,$//')
    
    # Create the tmux session with the first container
    FIRST_CONTAINER=$(echo "$containers" | head -n1)
    if [ "$MONITOR_MODE" = true ]; then
        # In monitor mode, respawn the pane when container exits
        tmux new-session -d -s "$SESSION" -n "agents" \
            "while true; do echo -e '${CYAN}Container: $FIRST_CONTAINER${NC}' && docker exec -it $FIRST_CONTAINER bash -c 'echo 1 | ./view_agents.sh claude_agents' || echo -e '${RED}Container $FIRST_CONTAINER has exited. Waiting for cleanup...${NC}'; sleep 2; done"
    else
        tmux new-session -d -s "$SESSION" -n "agents" \
            "echo -e '${CYAN}Container: $FIRST_CONTAINER${NC}' && docker exec -it $FIRST_CONTAINER bash -c 'echo 1 | ./view_agents.sh claude_agents' || echo -e '${RED}Failed to connect to $FIRST_CONTAINER${NC}' && read"
    fi
    
    # Add remaining containers as panes
    PANE_INDEX=1
    for container in $(echo "$containers" | tail -n +2); do
        if [ "$MONITOR_MODE" = true ]; then
            tmux split-window -t "$SESSION:agents" \
                "while true; do echo -e '${CYAN}Container: $container${NC}' && docker exec -it $container bash -c 'echo 1 | ./view_agents.sh claude_agents' || echo -e '${RED}Container $container has exited. Waiting for cleanup...${NC}'; sleep 2; done"
        else
            tmux split-window -t "$SESSION:agents" \
                "echo -e '${CYAN}Container: $container${NC}' && docker exec -it $container bash -c 'echo 1 | ./view_agents.sh claude_agents' || echo -e '${RED}Failed to connect to $container${NC}' && read"
        fi
        
        # Rearrange panes in tiled layout after each split
        tmux select-layout -t "$SESSION:agents" tiled
        ((PANE_INDEX++))
    done
    
    # Final layout adjustment
    tmux select-layout -t "$SESSION:agents" tiled
    
    # Set some helpful tmux options
    tmux set -t "$SESSION" mouse on
    tmux set -t "$SESSION:agents" pane-border-status top
    tmux set -t "$SESSION:agents" pane-border-format " #{pane_index}: Container "
    
    # Store the container list in tmux environment
    tmux set-environment -t "$SESSION" CCFARM_CONTAINERS "$container_list"
}

# Create the initial session
create_session "$RUNNING_CONTAINERS"

echo ""
echo -e "${GREEN}✓ Session created successfully!${NC}"
echo ""

# Display tips based on container count
if [ "$CONTAINER_COUNT" -ge 4 ]; then
    echo -e "${YELLOW}💡 Tip: With $CONTAINER_COUNT containers, consider reducing font size:${NC}"
    echo -e "   ${YELLOW}• macOS: Cmd + minus (-)${NC}"
    echo -e "   ${YELLOW}• Linux/Windows: Ctrl + minus (-)${NC}"
    echo -e "   ${YELLOW}• Reset: Cmd/Ctrl + 0${NC}"
    echo ""
fi

echo -e "${GREEN}Navigation tips:${NC}"
echo -e "• Switch panes: ${CYAN}Ctrl+B then arrow keys${NC}"
echo -e "• Zoom a pane: ${CYAN}Ctrl+B then Z${NC}"
echo -e "• Mouse click: ${CYAN}Click on any pane to focus${NC}"
echo -e "• Detach: ${CYAN}Ctrl+B then D${NC}"
echo ""

echo -e "${GREEN}Attaching to session...${NC}"
sleep 1

# Attach to the session
tmux attach-session -t "$SESSION"