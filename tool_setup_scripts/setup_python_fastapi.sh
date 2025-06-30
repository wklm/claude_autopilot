#!/usr/bin/env bash

# Setup script for Python FastAPI development environment
# Installs: Python 3.12+, uv, ruff, mypy, and other Python tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions and requirements
PYTHON_MIN_VERSION="3.12"
UV_VERSION="latest"

main() {
    show_banner "Python FastAPI Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "python3" "--version"
    show_tool_status "uv" "--version"
    show_tool_status "ruff" "--version"
    show_tool_status "mypy" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "python3-dev" "Python 3 Development Headers"
    install_apt_package "python3-pip" "pip"
    install_apt_package "python3-venv" "Python venv"
    
    # Check Python version
    log_step "Checking Python version"
    if command_exists python3; then
        current_python=$(python3 --version | awk '{print $2}')
        log_info "Current Python version: $current_python"
        
        # Compare versions
        if [[ "$(printf '%s\n' "$PYTHON_MIN_VERSION" "$current_python" | sort -V | head -n1)" != "$PYTHON_MIN_VERSION" ]]; then
            log_warning "Python $current_python is older than recommended $PYTHON_MIN_VERSION"
            
            if confirm "Install Python $PYTHON_MIN_VERSION from deadsnakes PPA?"; then
                # Add deadsnakes PPA
                sudo add-apt-repository -y ppa:deadsnakes/ppa
                sudo apt-get update
                install_apt_package "python${PYTHON_MIN_VERSION}" "Python ${PYTHON_MIN_VERSION}"
                install_apt_package "python${PYTHON_MIN_VERSION}-dev" "Python ${PYTHON_MIN_VERSION} Dev"
                install_apt_package "python${PYTHON_MIN_VERSION}-venv" "Python ${PYTHON_MIN_VERSION} venv"
                
                # Update alternatives
                if confirm "Set Python ${PYTHON_MIN_VERSION} as default python3?"; then
                    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_MIN_VERSION} 1
                    sudo update-alternatives --config python3
                fi
            fi
        else
            log_success "Python version is sufficient"
        fi
    else
        log_error "Python 3 is not installed"
        install_apt_package "python3" "Python 3"
    fi
    
    # Install uv
    log_step "Installing uv (Fast Python package manager)"
    if command_exists uv; then
        log_info "uv is already installed"
        if confirm "Update uv to latest version?"; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            log_success "uv updated"
        fi
    else
        if confirm "Install uv (recommended for modern Python development)?"; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            
            # Add uv to PATH for different shells
            uv_path_line='export PATH="$HOME/.local/bin:$PATH"'
            
            if [[ -f "$HOME/.bashrc" ]]; then
                add_to_file_if_missing "$HOME/.bashrc" "$uv_path_line" "uv PATH"
            fi
            
            if [[ -f "$HOME/.zshrc" ]]; then
                add_to_file_if_missing "$HOME/.zshrc" "$uv_path_line" "uv PATH"
            fi
            
            # Source for current session
            export PATH="$HOME/.local/bin:$PATH"
            log_success "uv installed"
            log_info "You may need to restart your shell or run: source ~/.bashrc"
        fi
    fi
    
    # Install Python tools via uv
    if command_exists uv; then
        log_step "Installing Python development tools"
        
        # Install ruff
        if ! command_exists ruff; then
            if confirm "Install ruff (Fast Python linter and formatter)?"; then
                uv tool install ruff
                log_success "ruff installed"
            fi
        else
            log_info "ruff is already installed"
            if confirm "Update ruff to latest version?"; then
                uv tool upgrade ruff
                log_success "ruff updated"
            fi
        fi
        
        # Install mypy
        if ! command_exists mypy; then
            if confirm "Install mypy (Python type checker)?"; then
                uv tool install mypy
                log_success "mypy installed"
            fi
        else
            log_info "mypy is already installed"
            if confirm "Update mypy to latest version?"; then
                uv tool upgrade mypy
                log_success "mypy updated"
            fi
        fi
        
        # Install pre-commit
        if ! command_exists pre-commit; then
            if confirm "Install pre-commit (Git hook framework)?"; then
                uv tool install pre-commit
                log_success "pre-commit installed"
            fi
        else
            log_info "pre-commit is already installed"
        fi
        
        # Install ipython for better REPL
        if ! command_exists ipython; then
            if confirm "Install ipython (Enhanced Python REPL)?"; then
                uv tool install ipython
                log_success "ipython installed"
            fi
        else
            log_info "ipython is already installed"
        fi
    fi
    
    # Setup global gitignore for Python
    log_step "Configuring Git for Python development"
    if confirm "Add Python patterns to global .gitignore?"; then
        gitignore_file="$HOME/.gitignore_global"
        touch "$gitignore_file"
        
        # Python patterns
        python_patterns=(
            "__pycache__/"
            "*.py[cod]"
            "*$py.class"
            "*.so"
            ".Python"
            "venv/"
            "env/"
            ".venv/"
            ".env"
            "*.egg-info/"
            "dist/"
            "build/"
            ".mypy_cache/"
            ".pytest_cache/"
            ".ruff_cache/"
            ".coverage"
            "htmlcov/"
            ".tox/"
            ".hypothesis/"
        )
        
        for pattern in "${python_patterns[@]}"; do
            if ! grep -Fxq "$pattern" "$gitignore_file"; then
                echo "$pattern" >> "$gitignore_file"
            fi
        done
        
        git config --global core.excludesfile "$gitignore_file"
        log_success "Global .gitignore configured for Python"
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "python3" "--version"
    show_tool_status "uv" "--version"
    show_tool_status "ruff" "--version"
    show_tool_status "mypy" "--version"
    show_tool_status "pre-commit" "--version"
    show_tool_status "ipython" "--version"
    
    echo
    log_success "Python FastAPI development environment is ready!"
    log_info "To create a new FastAPI project, run:"
    echo -e "  ${CYAN}uv init my-project${RESET}"
    echo -e "  ${CYAN}cd my-project${RESET}"
    echo -e "  ${CYAN}uv add fastapi uvicorn[standard]${RESET}"
}

# Run main function
main "$@" 