#!/usr/bin/env bash

# Setup script for Angular development environment
# Installs: Node.js, npm, Angular CLI, and related tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions and requirements
NODE_VERSION="20"  # LTS version
ANGULAR_CLI_VERSION="latest"

main() {
    show_banner "Angular Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "node" "--version"
    show_tool_status "npm" "--version"
    show_tool_status "ng" "version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "build-essential" "Build Essential"
    
    # Install Node.js via NodeSource
    log_step "Installing Node.js"
    if command_exists node; then
        current_node=$(node --version | sed 's/v//')
        log_info "Current Node.js version: $current_node"
        
        if [[ "${current_node%%.*}" -lt "$NODE_VERSION" ]]; then
            log_warning "Node.js version is older than recommended v$NODE_VERSION"
            if confirm "Update Node.js to v$NODE_VERSION?"; then
                curl -fsSL https://deb.nodesource.com/setup_$NODE_VERSION.x | sudo -E bash -
                install_apt_package "nodejs" "Node.js"
            fi
        else
            log_success "Node.js version is sufficient"
        fi
    else
        if confirm "Install Node.js v$NODE_VERSION from NodeSource?"; then
            curl -fsSL https://deb.nodesource.com/setup_$NODE_VERSION.x | sudo -E bash -
            install_apt_package "nodejs" "Node.js"
            log_success "Node.js installed"
        fi
    fi
    
    # Install global npm packages
    if command_exists npm; then
        log_step "Installing Angular CLI and development tools"
        
        # Install Angular CLI
        if ! command_exists ng; then
            if confirm "Install Angular CLI globally?"; then
                sudo npm install -g @angular/cli@$ANGULAR_CLI_VERSION
                log_success "Angular CLI installed"
            fi
        else
            log_info "Angular CLI is already installed"
            if confirm "Update Angular CLI to latest version?"; then
                sudo npm update -g @angular/cli
                log_success "Angular CLI updated"
            fi
        fi
        
        # Install additional useful tools
        log_step "Installing additional development tools"
        
        # Install TypeScript globally
        if ! command_exists tsc; then
            if confirm "Install TypeScript globally?"; then
                sudo npm install -g typescript
                log_success "TypeScript installed"
            fi
        else
            log_info "TypeScript is already installed"
        fi
        
        # Install http-server for quick static serving
        if ! command_exists http-server; then
            if confirm "Install http-server (for quick static file serving)?"; then
                sudo npm install -g http-server
                log_success "http-server installed"
            fi
        else
            log_info "http-server is already installed"
        fi
        
        # Install npm-check-updates
        if ! command_exists ncu; then
            if confirm "Install npm-check-updates (for dependency management)?"; then
                sudo npm install -g npm-check-updates
                log_success "npm-check-updates installed"
            fi
        else
            log_info "npm-check-updates is already installed"
        fi
    fi
    
    # Configure npm
    log_step "Configuring npm"
    if confirm "Configure npm with better defaults?"; then
        # Set npm init defaults
        npm config set init-author-name "$(git config --global user.name || echo 'Your Name')"
        npm config set init-author-email "$(git config --global user.email || echo 'your.email@example.com')"
        npm config set init-license "MIT"
        
        # Improve npm performance
        npm config set progress false
        npm config set save-exact true
        
        log_success "npm configured with better defaults"
    fi
    
    # Setup Chrome for testing
    log_step "Chrome/Chromium setup for testing"
    if ! command_exists google-chrome && ! command_exists chromium-browser; then
        if confirm "Install Chromium for Angular testing?"; then
            install_apt_package "chromium-browser" "Chromium Browser"
        fi
    else
        log_info "Chrome/Chromium is already installed"
    fi
    
    # VS Code extensions
    if command_exists code; then
        log_step "VS Code Angular extensions"
        if confirm "Install Angular VS Code extensions?"; then
            code --install-extension Angular.ng-template
            code --install-extension johnpapa.Angular2
            code --install-extension nrwl.angular-console
            log_success "VS Code extensions installed"
        fi
    fi
    
    # Create Angular workspace directory
    log_step "Setting up workspace"
    ANGULAR_WORKSPACE="$HOME/angular-projects"
    if confirm "Create Angular workspace directory at $ANGULAR_WORKSPACE?"; then
        create_directory "$ANGULAR_WORKSPACE" "Angular workspace"
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "node" "--version"
    show_tool_status "npm" "--version"
    show_tool_status "ng" "version"
    show_tool_status "tsc" "--version"
    
    echo
    log_success "Angular development environment is ready!"
    log_info "To create a new Angular project, run:"
    echo -e "  ${CYAN}ng new my-app${RESET}"
    echo -e "  ${CYAN}cd my-app${RESET}"
    echo -e "  ${CYAN}ng serve${RESET}"
    
    if command_exists ng; then
        echo
        log_info "Run 'ng version' to see detailed Angular CLI information."
    fi
}

# Run main function
main "$@" 