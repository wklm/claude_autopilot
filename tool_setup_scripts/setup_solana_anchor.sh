#!/usr/bin/env bash

# Setup script for Solana blockchain development with Anchor framework
# Installs: Rust, Solana CLI, Anchor, and related tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions and requirements
SOLANA_VERSION="stable"
ANCHOR_VERSION="0.30.1"
NODE_VERSION="20"

main() {
    show_banner "Solana & Anchor Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "rustc" "--version"
    show_tool_status "solana" "--version"
    show_tool_status "anchor" "--version"
    show_tool_status "node" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "pkg-config" "pkg-config"
    install_apt_package "libssl-dev" "SSL Development Libraries"
    install_apt_package "libudev-dev" "udev Development Libraries"
    
    # Install Rust
    log_step "Installing Rust"
    if command_exists rustc; then
        log_info "Rust is already installed"
        current_rust=$(rustc --version | awk '{print $2}')
        log_info "Current Rust version: $current_rust"
        
        if confirm "Update Rust to latest stable?"; then
            rustup update stable
            log_success "Rust updated"
        fi
    else
        if confirm "Install Rust via rustup?"; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            
            # Source cargo env
            source "$HOME/.cargo/env"
            
            log_success "Rust installed"
        fi
    fi
    
    # Install Solana CLI
    log_step "Installing Solana CLI"
    if command_exists solana; then
        log_info "Solana CLI is already installed"
        if confirm "Update Solana CLI to latest $SOLANA_VERSION?"; then
            solana-install update
            log_success "Solana CLI updated"
        fi
    else
        if confirm "Install Solana CLI?"; then
            sh -c "$(curl -sSfL https://release.solana.com/v1.18.17/install)"
            
            # Add Solana to PATH
            solana_path_line='export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"'
            
            if [[ -f "$HOME/.bashrc" ]]; then
                add_to_file_if_missing "$HOME/.bashrc" "$solana_path_line" "Solana PATH"
            fi
            
            if [[ -f "$HOME/.zshrc" ]]; then
                add_to_file_if_missing "$HOME/.zshrc" "$solana_path_line" "Solana PATH"
            fi
            
            # Source for current session
            export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"
            
            log_success "Solana CLI installed"
            
            # Configure Solana
            log_info "Configuring Solana CLI..."
            solana config set --url localhost
            solana config set --keypair ~/.config/solana/id.json
            
            # Generate a new keypair if it doesn't exist
            if [[ ! -f "$HOME/.config/solana/id.json" ]]; then
                if confirm "Generate a new Solana keypair?"; then
                    solana-keygen new --no-passphrase
                    log_success "Solana keypair generated"
                fi
            fi
        fi
    fi
    
    # Install Node.js for Anchor
    log_step "Installing Node.js (required for Anchor)"
    if command_exists node; then
        current_node=$(node --version | sed 's/v//')
        log_info "Current Node.js version: $current_node"
        
        if [[ "${current_node%%.*}" -lt "$NODE_VERSION" ]]; then
            log_warning "Node.js version is older than recommended v$NODE_VERSION"
            if confirm "Update Node.js to v$NODE_VERSION?"; then
                curl -fsSL https://deb.nodesource.com/setup_$NODE_VERSION.x | sudo -E bash -
                install_apt_package "nodejs" "Node.js"
            fi
        fi
    else
        if confirm "Install Node.js v$NODE_VERSION?"; then
            curl -fsSL https://deb.nodesource.com/setup_$NODE_VERSION.x | sudo -E bash -
            install_apt_package "nodejs" "Node.js"
            log_success "Node.js installed"
        fi
    fi
    
    # Install Yarn
    if ! command_exists yarn; then
        if confirm "Install Yarn package manager?"; then
            sudo npm install -g yarn
            log_success "Yarn installed"
        fi
    else
        log_info "Yarn is already installed"
    fi
    
    # Install Anchor
    log_step "Installing Anchor Framework"
    if command_exists anchor; then
        log_info "Anchor is already installed"
        current_anchor=$(anchor --version | awk '{print $2}')
        log_info "Current Anchor version: $current_anchor"
        
        if confirm "Update Anchor?"; then
            cargo install --git https://github.com/coral-xyz/anchor --tag v$ANCHOR_VERSION anchor-cli --locked
            log_success "Anchor updated"
        fi
    else
        if confirm "Install Anchor framework v$ANCHOR_VERSION?"; then
            # Install Anchor dependencies
            cargo install --git https://github.com/coral-xyz/anchor --tag v$ANCHOR_VERSION anchor-cli --locked
            
            log_success "Anchor installed"
        fi
    fi
    
    # Install additional Solana tools
    if command_exists cargo; then
        log_step "Installing additional Solana development tools"
        
        # Install SPL Token CLI
        if ! command_exists spl-token; then
            if confirm "Install SPL Token CLI?"; then
                cargo install spl-token-cli
                log_success "SPL Token CLI installed"
            fi
        else
            log_info "SPL Token CLI is already installed"
        fi
        
        # Install Solana Program Library tools
        if confirm "Install additional Solana development tools?"; then
            # Install solana-test-validator if not present
            if ! command_exists solana-test-validator; then
                log_info "solana-test-validator is included with Solana CLI"
            fi
        fi
    fi
    
    # Create Solana workspace
    log_step "Setting up Solana workspace"
    SOLANA_WORKSPACE="$HOME/solana-projects"
    if confirm "Create Solana workspace directory at $SOLANA_WORKSPACE?"; then
        create_directory "$SOLANA_WORKSPACE" "Solana workspace"
        
        # Create a sample Anchor project structure
        if command_exists anchor && confirm "Create a sample Anchor project?"; then
            cd "$SOLANA_WORKSPACE"
            log_info "Creating sample Anchor project..."
            anchor init hello-anchor --javascript
            cd hello-anchor
            
            log_success "Sample Anchor project created at $SOLANA_WORKSPACE/hello-anchor"
        fi
    fi
    
    # VS Code extensions
    if command_exists code; then
        log_step "VS Code Solana/Rust extensions"
        if confirm "Install Solana and Rust VS Code extensions?"; then
            code --install-extension rust-lang.rust-analyzer
            code --install-extension matklad.rust-analyzer
            code --install-extension Solana.solana
            log_success "VS Code extensions installed"
        fi
    fi
    
    # Setup local validator
    log_step "Local Solana validator setup"
    if command_exists solana-test-validator; then
        if confirm "Create systemd service for local Solana validator?"; then
            # Create systemd service file
            cat << EOF | sudo tee /etc/systemd/system/solana-test-validator.service
[Unit]
Description=Solana Test Validator
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME
ExecStart=$HOME/.local/share/solana/install/active_release/bin/solana-test-validator
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
            
            sudo systemctl daemon-reload
            log_success "Solana test validator service created"
            log_info "To start the validator: sudo systemctl start solana-test-validator"
            log_info "To enable on boot: sudo systemctl enable solana-test-validator"
        fi
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "rustc" "--version"
    show_tool_status "cargo" "--version"
    show_tool_status "solana" "--version"
    show_tool_status "anchor" "--version"
    show_tool_status "node" "--version"
    show_tool_status "yarn" "--version"
    
    echo
    log_success "Solana & Anchor development environment is ready!"
    log_info "To create a new Anchor project:"
    echo -e "  ${CYAN}cd $SOLANA_WORKSPACE${RESET}"
    echo -e "  ${CYAN}anchor init my-project${RESET}"
    echo -e "  ${CYAN}cd my-project${RESET}"
    echo -e "  ${CYAN}anchor build${RESET}"
    echo -e "  ${CYAN}anchor test${RESET}"
    
    echo
    log_info "To start local validator: ${CYAN}solana-test-validator${RESET}"
    log_info "To check config: ${CYAN}solana config get${RESET}"
}

# Run main function
main "$@" 