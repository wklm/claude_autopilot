#!/usr/bin/env bash
# setup.sh - Automated setup for Claude Code Agent Farm

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check for required tools
check_requirements() {
    local missing_tools=()
    
    # Check for uv
    if ! command -v uv &> /dev/null; then
        missing_tools+=("uv")
    fi
    
    # Check for direnv
    if ! command -v direnv &> /dev/null; then
        missing_tools+=("direnv")
    fi
    
    # Check for tmux
    if ! command -v tmux &> /dev/null; then
        missing_tools+=("tmux")
    fi
    
    # Check for git
    if ! command -v git &> /dev/null; then
        missing_tools+=("git")
    fi
    
    # Check for bun (required for the TypeScript project)
    if ! command -v bun &> /dev/null; then
        missing_tools+=("bun")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        echo ""
        echo "Installation instructions:"
        
        for tool in "${missing_tools[@]}"; do
            case $tool in
                "uv")
                    echo "  - uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
                    ;;
                "direnv")
                    echo "  - direnv: https://direnv.net/docs/installation.html"
                    ;;
                "tmux")
                    echo "  - tmux: sudo apt install tmux (Ubuntu) or brew install tmux (macOS)"
                    ;;
                "bun")
                    echo "  - bun: curl -fsSL https://bun.sh/install | bash"
                    ;;
                "git")
                    echo "  - git: sudo apt install git (Ubuntu) or brew install git (macOS)"
                    ;;
            esac
        done
        
        exit 1
    fi
}

# Main setup process
main() {
    print_status "Claude Code Agent Farm Setup"
    echo ""
    
    # Check requirements
    print_status "Checking requirements..."
    check_requirements
    print_success "All required tools found"
    
    # Create virtual environment
    print_status "Creating Python 3.13 virtual environment..."
    uv venv --python 3.13
    print_success "Virtual environment created"
    
    # Lock and sync dependencies
    print_status "Installing dependencies..."
    uv lock --upgrade
    uv sync --all-extras
    print_success "Dependencies installed"
    
    # Create .envrc file
    print_status "Creating .envrc file..."
    echo 'source .venv/bin/activate' > .envrc
    print_success ".envrc file created"
    
    # Set up direnv
    print_status "Setting up direnv..."
    direnv allow
    print_success "direnv configured"
    
    # Make scripts executable
    print_status "Making scripts executable..."
    chmod +x claude_code_agent_farm.py 2>/dev/null || true
    chmod +x view_agents.sh 2>/dev/null || true
    print_success "Scripts are now executable"
    
    # Create config directory
    print_status "Creating config directory..."
    mkdir -p configs
    
    # Create sample config if it doesn't exist
    if [ ! -f "configs/sample.json" ]; then
        cat > configs/sample.json << 'EOF'
{
  "agents": 10,
  "auto_restart": true,
  "stagger": 5.0,
  "check_interval": 15,
  "skip_regenerate": false,
  "skip_commit": false
}
EOF
        print_success "Sample config created at configs/sample.json"
    fi
    
    # Final instructions
    echo ""
    echo -e "${GREEN}✨ Setup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Activate the environment:"
    echo "   source .venv/bin/activate"
    echo "   # Or just 'cd' into this directory if using direnv"
    echo ""
    echo "2. Run the agent farm:"
    echo "   claude-agent-farm --path /your/project/path"
    echo ""
    echo "3. View agents in another terminal:"
    echo "   ./view_agents.sh"
    echo ""
    echo "For more information, see README.md"
}

# Run main function
main