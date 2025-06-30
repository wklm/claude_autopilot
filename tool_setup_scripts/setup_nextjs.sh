#!/usr/bin/env bash

# Setup script for Next.js development environment
# Installs: Node.js 22+, Bun, pnpm, and essential dev tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
NODE_VERSION="22"
BUN_VERSION="latest"

main() {
    show_banner "Next.js Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "node" "--version"
    show_tool_status "npm" "--version"
    show_tool_status "bun" "--version"
    show_tool_status "pnpm" "--version"
    show_tool_status "npx" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    
    # Install Node.js
    log_step "Installing Node.js"
    if command_exists node; then
        current_node=$(node --version | sed 's/v//' | cut -d. -f1)
        log_info "Current Node.js version: v$(node --version | sed 's/v//')"
        
        if [[ "$current_node" -lt "$NODE_VERSION" ]]; then
            log_warning "Node.js v$current_node is older than recommended v$NODE_VERSION"
            if confirm "Install Node.js v$NODE_VERSION via NodeSource?"; then
                install_nodejs
            fi
        else
            log_success "Node.js version is sufficient"
        fi
    else
        if confirm "Install Node.js v$NODE_VERSION?"; then
            install_nodejs
        else
            log_error "Node.js is required to continue"
            exit 1
        fi
    fi
    
    # Install Bun
    log_step "Installing Bun (Fast JavaScript runtime & package manager)"
    if command_exists bun; then
        log_info "Bun is already installed"
        if confirm "Update Bun to latest version?"; then
            bun upgrade
            log_success "Bun updated"
        fi
    else
        if confirm "Install Bun (recommended for Next.js 15)?"; then
            curl -fsSL https://bun.sh/install | bash
            
            # Add Bun to PATH
            bun_path_line='export BUN_INSTALL="$HOME/.bun"; export PATH="$BUN_INSTALL/bin:$PATH"'
            
            if [[ -f "$HOME/.bashrc" ]]; then
                add_to_file_if_missing "$HOME/.bashrc" "$bun_path_line" "Bun PATH"
            fi
            
            if [[ -f "$HOME/.zshrc" ]]; then
                add_to_file_if_missing "$HOME/.zshrc" "$bun_path_line" "Bun PATH"
            fi
            
            # Source for current session
            export BUN_INSTALL="$HOME/.bun"
            export PATH="$BUN_INSTALL/bin:$PATH"
            
            log_success "Bun installed"
            log_info "You may need to restart your shell or run: source ~/.bashrc"
        fi
    fi
    
    # Install pnpm
    log_step "Installing alternative package managers"
    if ! command_exists pnpm; then
        if confirm "Install pnpm (Fast, disk space efficient package manager)?"; then
            npm install -g pnpm
            log_success "pnpm installed"
        fi
    else
        log_info "pnpm is already installed"
        if confirm "Update pnpm to latest version?"; then
            pnpm add -g pnpm
            log_success "pnpm updated"
        fi
    fi
    
    # Install global Node.js tools
    log_step "Installing essential development tools"
    
    # TypeScript
    if ! command_exists tsc; then
        if confirm "Install TypeScript globally?"; then
            npm install -g typescript
            log_success "TypeScript installed"
        fi
    else
        log_info "TypeScript is already installed"
    fi
    
    # ESLint
    if ! command_exists eslint; then
        if confirm "Install ESLint globally?"; then
            npm install -g eslint
            log_success "ESLint installed"
        fi
    else
        log_info "ESLint is already installed"
    fi
    
    # Prettier
    if ! command_exists prettier; then
        if confirm "Install Prettier globally?"; then
            npm install -g prettier
            log_success "Prettier installed"
        fi
    else
        log_info "Prettier is already installed"
    fi
    
    # Vercel CLI
    if ! command_exists vercel; then
        if confirm "Install Vercel CLI (for deployment)?"; then
            npm install -g vercel
            log_success "Vercel CLI installed"
        fi
    else
        log_info "Vercel CLI is already installed"
    fi
    
    # Create Next.js app helper
    if ! command_exists create-next-app; then
        if confirm "Install create-next-app globally?"; then
            npm install -g create-next-app
            log_success "create-next-app installed"
        fi
    else
        log_info "create-next-app is already installed"
    fi
    
    # Setup global gitignore for Node.js/Next.js
    log_step "Configuring Git for Node.js/Next.js development"
    if confirm "Add Node.js/Next.js patterns to global .gitignore?"; then
        gitignore_file="$HOME/.gitignore_global"
        touch "$gitignore_file"
        
        # Node.js/Next.js patterns
        node_patterns=(
            "# Dependencies"
            "node_modules/"
            ".pnp"
            ".pnp.js"
            ".yarn/install-state.gz"
            ""
            "# Testing"
            "coverage/"
            ".nyc_output"
            ""
            "# Next.js"
            ".next/"
            "out/"
            "build/"
            ".vercel"
            ""
            "# Production"
            "dist/"
            ""
            "# Misc"
            ".DS_Store"
            "*.pem"
            ""
            "# Debug"
            "npm-debug.log*"
            "yarn-debug.log*"
            "yarn-error.log*"
            ".pnpm-debug.log*"
            ""
            "# Local env files"
            ".env"
            ".env.local"
            ".env.development.local"
            ".env.test.local"
            ".env.production.local"
            ""
            "# Editor"
            ".vscode/"
            ".idea/"
            "*.swp"
            "*.swo"
            "*~"
            ""
            "# TypeScript"
            "*.tsbuildinfo"
            "next-env.d.ts"
        )
        
        for pattern in "${node_patterns[@]}"; do
            if [[ -n "$pattern" ]] && ! grep -Fxq "$pattern" "$gitignore_file" 2>/dev/null; then
                echo "$pattern" >> "$gitignore_file"
            elif [[ -z "$pattern" ]]; then
                echo "" >> "$gitignore_file"
            fi
        done
        
        git config --global core.excludesfile "$gitignore_file"
        log_success "Global .gitignore configured for Node.js/Next.js"
    fi
    
    # Configure npm/bun
    log_step "Configuring package managers"
    if confirm "Set up npm configuration for better performance?"; then
        # Enable npm progress bar
        npm config set progress true
        # Set npm cache
        npm config set cache "$HOME/.npm"
        # Enable color output
        npm config set color true
        log_success "npm configured"
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "node" "--version"
    show_tool_status "npm" "--version"
    show_tool_status "bun" "--version"
    show_tool_status "pnpm" "--version"
    show_tool_status "tsc" "--version"
    show_tool_status "eslint" "--version"
    show_tool_status "prettier" "--version"
    show_tool_status "vercel" "--version"
    show_tool_status "create-next-app" "--version"
    
    echo
    log_success "Next.js development environment is ready!"
    log_info "To create a new Next.js 15 project with the App Router, run:"
    echo -e "  ${CYAN}bunx create-next-app@latest my-app --typescript --tailwind --app --src-dir${RESET}"
    echo -e "  ${CYAN}cd my-app${RESET}"
    echo -e "  ${CYAN}bun dev${RESET}"
}

install_nodejs() {
    log_info "Installing Node.js v$NODE_VERSION via NodeSource repository..."
    
    # Download and execute NodeSource setup script
    curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | sudo -E bash -
    
    # Install Node.js
    install_apt_package "nodejs" "Node.js"
    
    # Verify installation
    if command_exists node; then
        log_success "Node.js $(node --version) installed successfully"
    else
        log_error "Failed to install Node.js"
        return 1
    fi
}

# Run main function
main "$@" 