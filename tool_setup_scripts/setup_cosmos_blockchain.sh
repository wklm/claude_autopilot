#!/usr/bin/env bash

# Setup script for Cosmos Blockchain development environment
# Installs: Go, Cosmos SDK, Ignite CLI, IBC, development tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
GO_VERSION="1.21.5"
IGNITE_VERSION="v28.2.0"

main() {
    show_banner "Cosmos Blockchain Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "go" "version"
    show_tool_status "ignite" "version"
    show_tool_status "cosmovisor" "version"
    show_tool_status "gaiad" "version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "jq" "jq (JSON processor)"
    install_apt_package "make" "Make"
    install_apt_package "gcc" "GCC"
    install_apt_package "g++" "G++"
    install_apt_package "wget" "Wget"
    
    # Install Go
    log_step "Installing Go"
    if ! command_exists go; then
        if confirm "Install Go $GO_VERSION?"; then
            install_go
        else
            log_error "Go is required for Cosmos development"
            exit 1
        fi
    else
        log_info "Go is already installed"
        current_go=$(go version | awk '{print $3}' | sed 's/go//')
        log_info "Current version: $current_go"
        if confirm "Update Go to $GO_VERSION?"; then
            install_go
        fi
    fi
    
    # Install Ignite CLI (formerly Starport)
    log_step "Installing Ignite CLI"
    if ! command_exists ignite; then
        if confirm "Install Ignite CLI (blockchain scaffolding tool)?"; then
            install_ignite
        fi
    else
        log_info "Ignite CLI is already installed"
        if confirm "Update Ignite CLI to latest version?"; then
            install_ignite
        fi
    fi
    
    # Install Cosmos SDK
    log_step "Installing Cosmos SDK"
    if confirm "Clone and set up Cosmos SDK?"; then
        setup_cosmos_sdk
    fi
    
    # Install CosmWasm
    log_step "Installing CosmWasm"
    if confirm "Install CosmWasm (smart contracts for Cosmos)?"; then
        install_cosmwasm
    fi
    
    # Install development tools
    log_step "Installing Cosmos development tools"
    
    # Cosmovisor
    if confirm "Install Cosmovisor (upgrade manager)?"; then
        go install cosmossdk.io/tools/cosmovisor/cmd/cosmovisor@latest
        log_success "Cosmovisor installed"
    fi
    
    # IBC-go
    if confirm "Install IBC-go (Inter-Blockchain Communication)?"; then
        go install github.com/cosmos/ibc-go/v7@latest
        log_success "IBC-go installed"
    fi
    
    # Gaia (Cosmos Hub)
    if confirm "Install Gaia (Cosmos Hub implementation)?"; then
        install_gaia
    fi
    
    # Install testing and debugging tools
    log_step "Installing testing tools"
    if confirm "Install blockchain testing tools?"; then
        # Lens (Cosmos SDK module explorer)
        go install github.com/cosmos/lens@latest
        
        # Cosmjs testing tools
        npm install -g @cosmjs/cli || log_warning "npm not found, skipping @cosmjs/cli"
        
        log_success "Testing tools installed"
    fi
    
    # Setup development environment
    log_step "Setting up development environment"
    if confirm "Create Cosmos development workspace?"; then
        create_cosmos_workspace
    fi
    
    # Install Rust (for CosmWasm development)
    log_step "Installing Rust (for CosmWasm)"
    if ! command_exists rustc; then
        if confirm "Install Rust (required for CosmWasm smart contracts)?"; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source "$HOME/.cargo/env"
            
            # Add wasm target
            rustup target add wasm32-unknown-unknown
            
            log_success "Rust installed with WASM target"
        fi
    else
        log_info "Rust is already installed"
        if ! rustup target list | grep -q "wasm32-unknown-unknown (installed)"; then
            rustup target add wasm32-unknown-unknown
            log_success "WASM target added"
        fi
    fi
    
    # Setup shell aliases
    log_step "Setting up shell aliases"
    if confirm "Add Cosmos development aliases to shell?"; then
        setup_cosmos_aliases
    fi
    
    # Install VS Code extensions
    log_step "VS Code Extensions"
    if command_exists code; then
        if confirm "Install recommended VS Code extensions for Cosmos?"; then
            code --install-extension golang.go
            code --install-extension rust-lang.rust-analyzer
            code --install-extension tamasfe.even-better-toml
            log_success "VS Code extensions installed"
        fi
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "go" "version"
    show_tool_status "ignite" "version"
    show_tool_status "cosmovisor" "version"
    show_tool_status "rustc" "--version"
    
    echo
    log_success "Cosmos Blockchain development environment is ready!"
    log_info "Useful commands:"
    echo -e "  ${CYAN}ignite scaffold chain mychain${RESET} - Create new blockchain"
    echo -e "  ${CYAN}ignite chain serve${RESET} - Run blockchain locally"
    echo -e "  ${CYAN}cosmos-dev-new <name>${RESET} - Create new Cosmos project"
    echo -e "  ${CYAN}cosmwasm-new <name>${RESET} - Create new CosmWasm contract"
    echo
    log_info "Documentation:"
    echo -e "  Cosmos SDK: ${CYAN}https://docs.cosmos.network${RESET}"
    echo -e "  Ignite CLI: ${CYAN}https://docs.ignite.com${RESET}"
}

install_go() {
    log_info "Installing Go $GO_VERSION..."
    
    # Remove old installation if exists
    sudo rm -rf /usr/local/go
    
    # Download and install
    wget -q https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz
    sudo tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz
    rm go${GO_VERSION}.linux-amd64.tar.gz
    
    # Add Go to PATH
    go_path_config='
# Go configuration
export PATH=$PATH:/usr/local/go/bin
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin
export GO111MODULE=on'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$go_path_config" "Go configuration"
    fi
    
    if [[ -f "$HOME/.zshrc" ]]; then
        add_to_file_if_missing "$HOME/.zshrc" "$go_path_config" "Go configuration"
    fi
    
    # Source for current session
    export PATH=$PATH:/usr/local/go/bin
    export GOPATH=$HOME/go
    export PATH=$PATH:$GOPATH/bin
    export GO111MODULE=on
    
    log_success "Go $GO_VERSION installed"
}

install_ignite() {
    log_info "Installing Ignite CLI..."
    
    # Install using curl
    curl -L https://get.ignite.com/cli@${IGNITE_VERSION}! | bash
    
    # Move to PATH
    sudo mv ignite /usr/local/bin/
    
    log_success "Ignite CLI installed"
}

setup_cosmos_sdk() {
    log_info "Setting up Cosmos SDK..."
    
    # Create workspace
    mkdir -p "$HOME/cosmos-workspace"
    cd "$HOME/cosmos-workspace"
    
    # Clone Cosmos SDK
    if [[ ! -d "cosmos-sdk" ]]; then
        git clone https://github.com/cosmos/cosmos-sdk.git
        cd cosmos-sdk
        git checkout main
    else
        cd cosmos-sdk
        git pull
    fi
    
    # Build Cosmos SDK tools
    log_info "Building Cosmos SDK tools..."
    make tools
    make build
    
    cd "$SCRIPT_DIR"
    log_success "Cosmos SDK setup complete"
}

install_cosmwasm() {
    log_info "Installing CosmWasm..."
    
    # Install wasmd
    go install github.com/CosmWasm/wasmd/cmd/wasmd@latest
    
    # Install cosmwasm-check
    cargo install cosmwasm-check
    
    # Install cosmwasm-cli tools
    cargo install cargo-generate --features vendored-openssl
    cargo install cargo-run-script
    
    log_success "CosmWasm installed"
}

install_gaia() {
    log_info "Installing Gaia (Cosmos Hub)..."
    
    # Clone and build Gaia
    cd "$HOME/cosmos-workspace"
    if [[ ! -d "gaia" ]]; then
        git clone https://github.com/cosmos/gaia.git
        cd gaia
    else
        cd gaia
        git pull
    fi
    
    # Build
    make install
    
    cd "$SCRIPT_DIR"
    log_success "Gaia installed"
}

create_cosmos_workspace() {
    log_info "Creating Cosmos development workspace..."
    
    WORKSPACE="$HOME/cosmos-workspace"
    mkdir -p "$WORKSPACE"/{chains,contracts,scripts,configs}
    
    # Create example chain scaffold script
    cat > "$WORKSPACE/scripts/new-chain.sh" << 'EOF'
#!/usr/bin/env bash
# Create a new Cosmos blockchain

if [[ -z "$1" ]]; then
    echo "Usage: $0 <chain-name>"
    exit 1
fi

CHAIN_NAME=$1
cd "$HOME/cosmos-workspace/chains"

echo "Creating new blockchain: $CHAIN_NAME"
ignite scaffold chain $CHAIN_NAME --no-module

cd $CHAIN_NAME
echo "Blockchain scaffolded at: $(pwd)"
echo ""
echo "Next steps:"
echo "1. cd $CHAIN_NAME"
echo "2. ignite chain serve"
EOF
    chmod +x "$WORKSPACE/scripts/new-chain.sh"
    
    # Create CosmWasm contract template
    cat > "$WORKSPACE/scripts/new-contract.sh" << 'EOF'
#!/usr/bin/env bash
# Create a new CosmWasm contract

if [[ -z "$1" ]]; then
    echo "Usage: $0 <contract-name>"
    exit 1
fi

CONTRACT_NAME=$1
cd "$HOME/cosmos-workspace/contracts"

echo "Creating new CosmWasm contract: $CONTRACT_NAME"
cargo generate --git https://github.com/CosmWasm/cw-template.git --name $CONTRACT_NAME

cd $CONTRACT_NAME
echo "Contract scaffolded at: $(pwd)"
echo ""
echo "Next steps:"
echo "1. cd $CONTRACT_NAME"
echo "2. cargo build"
echo "3. cargo test"
EOF
    chmod +x "$WORKSPACE/scripts/new-contract.sh"
    
    # Create local testnet config
    cat > "$WORKSPACE/configs/local-testnet.sh" << 'EOF'
#!/usr/bin/env bash
# Configure local testnet

# Chain configuration
CHAIN_ID="localnet-1"
MONIKER="local-validator"
DENOM="stake"

# Initialize chain
gaiad init $MONIKER --chain-id $CHAIN_ID

# Configure genesis
jq '.app_state.staking.params.bond_denom = "'$DENOM'"' ~/.gaia/config/genesis.json > temp.json && mv temp.json ~/.gaia/config/genesis.json

# Create validator account
gaiad keys add validator --keyring-backend test

# Add genesis account
gaiad add-genesis-account validator 1000000000$DENOM --keyring-backend test

# Create genesis transaction
gaiad gentx validator 1000000$DENOM --chain-id $CHAIN_ID --keyring-backend test

# Collect genesis transactions
gaiad collect-gentxs

echo "Local testnet configured!"
echo "Start with: gaiad start"
EOF
    chmod +x "$WORKSPACE/configs/local-testnet.sh"
    
    log_success "Cosmos workspace created at $WORKSPACE"
}

setup_cosmos_aliases() {
    log_info "Setting up Cosmos development aliases..."
    
    cosmos_aliases='
# Cosmos development aliases
export COSMOS_WORKSPACE="$HOME/cosmos-workspace"

# Ignite shortcuts
alias ignite-serve="ignite chain serve"
alias ignite-build="ignite chain build"
alias ignite-reset="ignite chain serve --reset-once"

# Cosmos SDK shortcuts
alias cosmos-init="gaiad init"
alias cosmos-start="gaiad start"
alias cosmos-keys="gaiad keys"
alias cosmos-query="gaiad query"
alias cosmos-tx="gaiad tx"

# Development helpers
cosmos-dev-new() {
    if [[ -z "$1" ]]; then
        echo "Usage: cosmos-dev-new <chain-name>"
        return 1
    fi
    cd "$COSMOS_WORKSPACE/chains"
    ignite scaffold chain "$1" --no-module
    cd "$1"
    echo "New chain created at: $(pwd)"
}

cosmwasm-new() {
    if [[ -z "$1" ]]; then
        echo "Usage: cosmwasm-new <contract-name>"
        return 1
    fi
    cd "$COSMOS_WORKSPACE/contracts"
    cargo generate --git https://github.com/CosmWasm/cw-template.git --name "$1"
    cd "$1"
    echo "New contract created at: $(pwd)"
}

# Testing helpers
cosmos-test-local() {
    echo "Starting local testnet..."
    gaiad start --home ~/.gaia-test
}

cosmos-benchmark() {
    if [[ -z "$1" ]]; then
        echo "Usage: cosmos-benchmark <module>"
        return 1
    fi
    go test -bench=. -benchmem "./x/$1/..."
}

# IBC helpers
ibc-relayer-setup() {
    echo "Setting up IBC relayer..."
    go install github.com/cosmos/relayer/v2@latest
    rly config init
}

# Module scaffolding
cosmos-module-new() {
    if [[ -z "$1" ]]; then
        echo "Usage: cosmos-module-new <module-name>"
        return 1
    fi
    ignite scaffold module "$1" --ibc
}

# Workspace navigation
alias cdcosmos="cd $COSMOS_WORKSPACE"
alias cdchains="cd $COSMOS_WORKSPACE/chains"
alias cdcontracts="cd $COSMOS_WORKSPACE/contracts"'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$cosmos_aliases" "Cosmos aliases"
    fi
    
    if [[ -f "$HOME/.zshrc" ]]; then
        add_to_file_if_missing "$HOME/.zshrc" "$cosmos_aliases" "Cosmos aliases"
    fi
    
    log_success "Cosmos development aliases added to shell"
}

# Run main function
main "$@" 