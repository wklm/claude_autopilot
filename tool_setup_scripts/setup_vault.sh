#!/usr/bin/env bash

# Setup script for HashiCorp Vault development environment
# Installs: Vault, Consul, Terraform, policy tools, development utilities

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
VAULT_VERSION="1.15.4"
CONSUL_VERSION="1.17.1"
TERRAFORM_VERSION="1.6.6"

main() {
    show_banner "HashiCorp Vault Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "vault" "version"
    show_tool_status "consul" "version"
    show_tool_status "terraform" "version"
    show_tool_status "sentinel" "version"
    show_tool_status "go" "version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "curl" "cURL"
    install_apt_package "wget" "Wget"
    install_apt_package "unzip" "Unzip"
    install_apt_package "git" "Git"
    install_apt_package "jq" "jq (JSON processor)"
    install_apt_package "gnupg" "GnuPG"
    install_apt_package "software-properties-common" "Software Properties Common"
    
    # Add HashiCorp GPG key and repository
    log_step "Adding HashiCorp repository"
    if ! [[ -f /usr/share/keyrings/hashicorp-archive-keyring.gpg ]]; then
        wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
        echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
        update_apt
    fi
    
    # Install Vault
    log_step "Installing HashiCorp Vault"
    if ! command_exists vault; then
        if confirm "Install Vault?"; then
            install_apt_package "vault" "HashiCorp Vault"
            log_success "Vault installed"
        fi
    else
        log_info "Vault is already installed"
        current_version=$(vault version | grep -oP 'Vault v\K[0-9.]+' | head -1)
        log_info "Current version: $current_version"
        if confirm "Update Vault to latest version?"; then
            sudo apt-get update && sudo apt-get install vault
            log_success "Vault updated"
        fi
    fi
    
    # Install Consul (often used with Vault)
    log_step "Installing HashiCorp Consul"
    if ! command_exists consul; then
        if confirm "Install Consul (storage backend for Vault)?"; then
            install_apt_package "consul" "HashiCorp Consul"
            log_success "Consul installed"
        fi
    else
        log_info "Consul is already installed"
    fi
    
    # Install Terraform (for Vault provider)
    log_step "Installing HashiCorp Terraform"
    if ! command_exists terraform; then
        if confirm "Install Terraform (for infrastructure as code)?"; then
            install_apt_package "terraform" "HashiCorp Terraform"
            log_success "Terraform installed"
        fi
    else
        log_info "Terraform is already installed"
    fi
    
    # Install Go (for Vault plugin development)
    log_step "Installing Go"
    if ! command_exists go; then
        if confirm "Install Go (for Vault plugin development)?"; then
            install_go
        fi
    else
        log_info "Go is already installed"
        go_version=$(go version | awk '{print $3}')
        log_info "Current version: $go_version"
    fi
    
    # Install Python tools for Vault
    log_step "Installing Python tools for Vault"
    if confirm "Install Python Vault libraries (hvac, vault-cli)?"; then
        pip3 install --user hvac vault-cli pyvault
        log_success "Python Vault libraries installed"
    fi
    
    # Install development tools
    log_step "Installing Vault development tools"
    
    # Vault UI
    if confirm "Download Vault UI for local development?"; then
        setup_vault_ui
    fi
    
    # Policy tools
    if confirm "Install Vault policy development tools?"; then
        install_policy_tools
    fi
    
    # Create Vault development environment
    log_step "Creating Vault development environment"
    if confirm "Create local Vault development setup?"; then
        create_vault_dev_env
    fi
    
    # Setup shell aliases
    log_step "Setting up shell aliases"
    if confirm "Add Vault development aliases to shell?"; then
        setup_vault_aliases
    fi
    
    # Install VS Code extensions
    log_step "VS Code Extensions"
    if command_exists code; then
        if confirm "Install recommended VS Code extensions for Vault?"; then
            code --install-extension HashiCorp.terraform
            code --install-extension HashiCorp.HCL
            code --install-extension 4ops.terraform
            log_success "VS Code extensions installed"
        fi
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "vault" "version"
    show_tool_status "consul" "version"
    show_tool_status "terraform" "version"
    show_tool_status "go" "version"
    
    echo
    log_success "HashiCorp Vault development environment is ready!"
    log_info "Useful commands:"
    echo -e "  ${CYAN}vault server -dev${RESET} - Start Vault in dev mode"
    echo -e "  ${CYAN}vault-dev-start${RESET} - Start configured dev environment"
    echo -e "  ${CYAN}vault status${RESET} - Check Vault status"
    echo -e "  ${CYAN}vault kv put secret/myapp key=value${RESET} - Store a secret"
    echo -e "  ${CYAN}vault kv get secret/myapp${RESET} - Retrieve a secret"
    echo -e "  ${CYAN}vault policy write my-policy policy.hcl${RESET} - Create policy"
}

install_go() {
    log_info "Installing Go..."
    
    GO_VERSION="1.21.5"
    wget -q https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz
    sudo tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz
    rm go${GO_VERSION}.linux-amd64.tar.gz
    
    # Add Go to PATH
    go_path_config='
# Go configuration
export PATH=$PATH:/usr/local/go/bin
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin'
    
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
    
    log_success "Go installed"
}

setup_vault_ui() {
    log_info "Setting up Vault UI..."
    
    # Vault UI is included in Vault binary since v0.10.0
    log_info "Vault UI is built into Vault and accessible at http://localhost:8200/ui when running in dev mode"
    log_info "For production, ensure 'ui = true' in your Vault configuration"
    
    log_success "Vault UI information provided"
}

install_policy_tools() {
    log_info "Installing Vault policy development tools..."
    
    # Create policy templates directory
    mkdir -p "$HOME/.vault/policies"
    
    # Create example policies
    cat > "$HOME/.vault/policies/dev-policy.hcl" << 'EOF'
# Development team policy
path "secret/data/dev/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/metadata/dev/*" {
  capabilities = ["list"]
}

path "auth/token/create" {
  capabilities = ["create", "update"]
}
EOF
    
    cat > "$HOME/.vault/policies/ops-policy.hcl" << 'EOF'
# Operations team policy
path "secret/data/ops/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "sys/health" {
  capabilities = ["read"]
}

path "sys/policies/acl/*" {
  capabilities = ["read", "list"]
}

path "sys/mounts" {
  capabilities = ["read", "list"]
}
EOF
    
    # Create policy testing script
    cat > "$HOME/.vault/test-policies.sh" << 'EOF'
#!/usr/bin/env bash
# Test Vault policies

VAULT_ADDR=${VAULT_ADDR:-http://127.0.0.1:8200}

echo "Testing Vault policies..."

# Create policies
vault policy write dev-policy "$HOME/.vault/policies/dev-policy.hcl"
vault policy write ops-policy "$HOME/.vault/policies/ops-policy.hcl"

# Create tokens with policies
DEV_TOKEN=$(vault token create -policy=dev-policy -format=json | jq -r '.auth.client_token')
OPS_TOKEN=$(vault token create -policy=ops-policy -format=json | jq -r '.auth.client_token')

echo "Dev token: $DEV_TOKEN"
echo "Ops token: $OPS_TOKEN"

# Test dev policy
echo -e "\nTesting dev policy..."
VAULT_TOKEN=$DEV_TOKEN vault kv put secret/dev/app1 password=secret123
VAULT_TOKEN=$DEV_TOKEN vault kv get secret/dev/app1

# Test ops policy
echo -e "\nTesting ops policy..."
VAULT_TOKEN=$OPS_TOKEN vault kv put secret/ops/server1 ssh_key=key123
VAULT_TOKEN=$OPS_TOKEN vault kv get secret/ops/server1

echo -e "\nPolicy testing complete!"
EOF
    chmod +x "$HOME/.vault/test-policies.sh"
    
    log_success "Policy tools installed"
}

create_vault_dev_env() {
    log_info "Creating Vault development environment..."
    
    # Create Vault directory structure
    mkdir -p "$HOME/vault-dev"/{config,data,logs,scripts,policies}
    
    # Create development Vault configuration
    cat > "$HOME/vault-dev/config/vault-dev.hcl" << 'EOF'
ui = true
disable_mlock = true

storage "file" {
  path = "./data"
}

listener "tcp" {
  address     = "127.0.0.1:8200"
  tls_disable = "true"
}

api_addr = "http://127.0.0.1:8200"
cluster_addr = "https://127.0.0.1:8201"

log_level = "info"
EOF
    
    # Create start script
    cat > "$HOME/vault-dev/start-vault.sh" << 'EOF'
#!/usr/bin/env bash
# Start Vault in development mode

cd "$(dirname "$0")"

# Kill any existing Vault process
pkill -f "vault server" || true

# Start Vault
echo "Starting Vault server..."
vault server -config=config/vault-dev.hcl > logs/vault.log 2>&1 &

# Wait for Vault to start
sleep 2

# Initialize Vault if needed
if ! vault status > /dev/null 2>&1; then
    echo "Initializing Vault..."
    vault operator init -key-shares=1 -key-threshold=1 > init-keys.txt
    
    # Extract keys
    UNSEAL_KEY=$(grep 'Unseal Key' init-keys.txt | awk '{print $4}')
    ROOT_TOKEN=$(grep 'Initial Root Token' init-keys.txt | awk '{print $4}')
    
    # Unseal Vault
    vault operator unseal $UNSEAL_KEY
    
    # Export root token
    export VAULT_TOKEN=$ROOT_TOKEN
    
    echo "Vault initialized!"
    echo "Root token: $ROOT_TOKEN"
    echo "Unseal key: $UNSEAL_KEY"
    echo ""
    echo "IMPORTANT: Save these keys securely!"
else
    echo "Vault is already initialized"
fi

echo ""
echo "Vault UI: http://127.0.0.1:8200/ui"
echo "Set token: export VAULT_TOKEN=<your-token>"
EOF
    chmod +x "$HOME/vault-dev/start-vault.sh"
    
    # Create stop script
    cat > "$HOME/vault-dev/stop-vault.sh" << 'EOF'
#!/usr/bin/env bash
# Stop Vault server

echo "Stopping Vault server..."
pkill -f "vault server"
echo "Vault server stopped"
EOF
    chmod +x "$HOME/vault-dev/stop-vault.sh"
    
    # Create example secrets script
    cat > "$HOME/vault-dev/scripts/setup-secrets.sh" << 'EOF'
#!/usr/bin/env bash
# Setup example secrets in Vault

# Enable KV v2 secrets engine
vault secrets enable -version=2 -path=secret kv

# Create some example secrets
vault kv put secret/database/mysql username=dbuser password=dbpass123 host=localhost port=3306
vault kv put secret/api/external api_key=sk-1234567890 endpoint=https://api.example.com
vault kv put secret/app/config debug=true max_connections=100 timeout=30

echo "Example secrets created!"
echo ""
echo "List secrets: vault kv list secret/"
echo "Read secret: vault kv get secret/database/mysql"
EOF
    chmod +x "$HOME/vault-dev/scripts/setup-secrets.sh"
    
    log_success "Vault development environment created at ~/vault-dev"
}

setup_vault_aliases() {
    log_info "Setting up Vault development aliases..."
    
    vault_aliases='
# Vault development aliases
export VAULT_ADDR="http://127.0.0.1:8200"
alias vault-dev="vault server -dev -dev-root-token-id=root"
alias vault-dev-start="cd ~/vault-dev && ./start-vault.sh"
alias vault-dev-stop="cd ~/vault-dev && ./stop-vault.sh"
alias vault-status="vault status"
alias vault-seal="vault operator seal"
alias vault-unseal="vault operator unseal"

# Secret operations
alias vault-list="vault kv list"
alias vault-get="vault kv get"
alias vault-put="vault kv put"
alias vault-delete="vault kv delete"

# Policy operations
alias vault-policies="vault policy list"
alias vault-policy-read="vault policy read"
alias vault-policy-write="vault policy write"

# Token operations
alias vault-token-create="vault token create"
alias vault-token-lookup="vault token lookup"
alias vault-token-revoke="vault token revoke"

# Audit operations
alias vault-audit-list="vault audit list"
alias vault-audit-enable="vault audit enable file file_path=/var/log/vault_audit.log"

# Helper functions
vault-init() {
    echo "Initializing Vault..."
    vault operator init -key-shares=5 -key-threshold=3
}

vault-login-root() {
    if [[ -z "$1" ]]; then
        echo "Usage: vault-login-root <root-token>"
        return 1
    fi
    export VAULT_TOKEN="$1"
    echo "Logged in with root token"
}

vault-create-token() {
    local policy="${1:-default}"
    vault token create -policy="$policy" -format=json | jq -r ".auth.client_token"
}

vault-enable-approle() {
    vault auth enable approle
    vault write auth/approle/role/my-role \
        secret_id_ttl=10m \
        token_num_uses=10 \
        token_ttl=20m \
        token_max_ttl=30m \
        secret_id_num_uses=40
}

vault-backup() {
    local backup_dir="${1:-vault-backup-$(date +%Y%m%d-%H%M%S)}"
    mkdir -p "$backup_dir"
    
    echo "Backing up Vault to $backup_dir..."
    
    # Export policies
    for policy in $(vault policy list | grep -v default | grep -v root); do
        vault policy read "$policy" > "$backup_dir/policy-$policy.hcl"
    done
    
    # Export auth methods
    vault auth list -format=json > "$backup_dir/auth-methods.json"
    
    # Export secrets engines
    vault secrets list -format=json > "$backup_dir/secrets-engines.json"
    
    echo "Backup complete!"
}'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$vault_aliases" "Vault aliases"
    fi
    
    if [[ -f "$HOME/.zshrc" ]]; then
        add_to_file_if_missing "$HOME/.zshrc" "$vault_aliases" "Vault aliases"
    fi
    
    log_success "Vault development aliases added to shell"
}

# Run main function
main "$@" 