#!/usr/bin/env bash

# Setup script for Terraform with Azure development environment
# Installs: Terraform, Azure CLI, tflint, tfsec, terragrunt, and related tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
TERRAFORM_VERSION="1.10.2"
TFLINT_VERSION="0.55.0"
TFSEC_VERSION="1.29.0"
TERRAGRUNT_VERSION="0.67.0"
INFRACOST_VERSION="latest"

main() {
    show_banner "Terraform with Azure Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "terraform" "version"
    show_tool_status "az" "version --output table"
    show_tool_status "tflint" "--version"
    show_tool_status "tfsec" "--version"
    show_tool_status "terragrunt" "--version"
    show_tool_status "infracost" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "gnupg" "GnuPG"
    install_apt_package "software-properties-common" "Software Properties Common"
    install_apt_package "lsb-release" "LSB Release"
    install_apt_package "jq" "jq (JSON processor)"
    install_apt_package "unzip" "unzip"
    
    # Install Terraform
    log_step "Installing Terraform"
    if command_exists terraform; then
        current_tf=$(terraform version -json 2>/dev/null | jq -r '.terraform_version' || echo "unknown")
        log_info "Current Terraform version: $current_tf"
        
        if [[ "$current_tf" != "$TERRAFORM_VERSION" ]]; then
            if confirm "Update Terraform to version $TERRAFORM_VERSION?"; then
                install_terraform
            fi
        fi
    else
        if confirm "Install Terraform $TERRAFORM_VERSION?"; then
            install_terraform
        fi
    fi
    
    # Install Azure CLI
    log_step "Installing Azure CLI"
    if command_exists az; then
        log_info "Azure CLI is already installed"
        current_az=$(az version --output tsv --query '"azure-cli"' 2>/dev/null || echo "unknown")
        log_info "Current Azure CLI version: $current_az"
        
        if confirm "Update Azure CLI to latest version?"; then
            install_azure_cli
        fi
    else
        if confirm "Install Azure CLI?"; then
            install_azure_cli
        fi
    fi
    
    # Install Azure CLI extensions
    if command_exists az; then
        log_step "Installing Azure CLI extensions"
        
        # Essential extensions for infrastructure work
        extensions=("azure-devops" "containerapp" "ml" "account" "resource-graph")
        
        for ext in "${extensions[@]}"; do
            if ! az extension list --output tsv --query "[?name=='$ext'].name" | grep -q "$ext"; then
                if confirm "Install Azure CLI extension: $ext?"; then
                    az extension add --name "$ext" --yes
                    log_success "Extension $ext installed"
                fi
            else
                log_info "Extension $ext is already installed"
            fi
        done
    fi
    
    # Install tflint
    log_step "Installing tflint (Terraform linter)"
    if command_exists tflint; then
        current_tflint=$(tflint --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "unknown")
        log_info "Current tflint version: $current_tflint"
        
        if [[ "$current_tflint" != "$TFLINT_VERSION" ]]; then
            if confirm "Update tflint to version $TFLINT_VERSION?"; then
                install_tflint
            fi
        fi
    else
        if confirm "Install tflint for Terraform linting?"; then
            install_tflint
        fi
    fi
    
    # Install tfsec
    log_step "Installing tfsec (Terraform security scanner)"
    if command_exists tfsec; then
        current_tfsec=$(tfsec --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "unknown")
        log_info "Current tfsec version: $current_tfsec"
        
        if [[ "$current_tfsec" != "$TFSEC_VERSION" ]]; then
            if confirm "Update tfsec to version $TFSEC_VERSION?"; then
                install_tfsec
            fi
        fi
    else
        if confirm "Install tfsec for security scanning?"; then
            install_tfsec
        fi
    fi
    
    # Install terragrunt
    log_step "Installing Terragrunt (DRY Terraform wrapper)"
    if command_exists terragrunt; then
        current_tg=$(terragrunt --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "unknown")
        log_info "Current Terragrunt version: $current_tg"
        
        if [[ "$current_tg" != "$TERRAGRUNT_VERSION" ]]; then
            if confirm "Update Terragrunt to version $TERRAGRUNT_VERSION?"; then
                install_terragrunt
            fi
        fi
    else
        if confirm "Install Terragrunt (optional)?"; then
            install_terragrunt
        fi
    fi
    
    # Install terraform-docs
    log_step "Installing terraform-docs"
    if ! command_exists terraform-docs; then
        if confirm "Install terraform-docs for documentation generation?"; then
            install_terraform_docs
        fi
    else
        log_info "terraform-docs is already installed"
    fi
    
    # Install terrascan
    log_step "Installing terrascan"
    if ! command_exists terrascan; then
        if confirm "Install terrascan for compliance scanning?"; then
            install_terrascan
        fi
    else
        log_info "terrascan is already installed"
    fi
    
    # Install Infracost
    log_step "Installing Infracost (cost estimation)"
    if ! command_exists infracost; then
        if confirm "Install Infracost for cost estimation?"; then
            install_infracost
        fi
    else
        log_info "Infracost is already installed"
        if confirm "Update Infracost to latest version?"; then
            install_infracost
        fi
    fi
    
    # Setup Terraform configuration
    log_step "Setting up Terraform configuration"
    
    # Create Terraform config directory
    tf_config_dir="$HOME/.terraform.d"
    mkdir -p "$tf_config_dir"
    
    # Create plugin cache directory
    plugin_cache_dir="$tf_config_dir/plugin-cache"
    mkdir -p "$plugin_cache_dir"
    
    # Create Terraform CLI configuration
    if confirm "Create Terraform CLI configuration (~/.terraformrc)?"; then
        cat > "$HOME/.terraformrc" << EOF
plugin_cache_dir = "$HOME/.terraform.d/plugin-cache"

provider_installation {
  filesystem_mirror {
    path    = "$HOME/.terraform.d/plugins"
    include = ["registry.terraform.io/*/*"]
  }
  direct {
    exclude = ["registry.terraform.io/*/*"]
  }
}
EOF
        log_success "Created ~/.terraformrc"
    fi
    
    # Create tflint configuration
    if confirm "Create tflint configuration (~/.tflint.hcl)?"; then
        cat > "$HOME/.tflint.hcl" << 'EOF'
plugin "azurerm" {
  enabled = true
  version = "0.27.0"
  source  = "github.com/terraform-linters/tflint-ruleset-azurerm"
}

plugin "terraform" {
  enabled = true
  preset  = "recommended"
}

rule "terraform_naming_convention" {
  enabled = true
  format  = "snake_case"
}

rule "terraform_documented_variables" {
  enabled = true
}

rule "terraform_documented_outputs" {
  enabled = true
}

rule "terraform_required_version" {
  enabled = true
}

rule "terraform_required_providers" {
  enabled = true
}
EOF
        log_success "Created ~/.tflint.hcl"
        
        # Initialize tflint plugins
        if command_exists tflint; then
            log_info "Initializing tflint plugins..."
            tflint --init
        fi
    fi
    
    # Create example Terraform project structure
    log_step "Creating example project structure"
    if confirm "Create example Terraform Azure project template?"; then
        example_dir="$HOME/terraform-azure-example"
        mkdir -p "$example_dir"/{modules,environments/{dev,staging,production}}
        
        # Create main module structure
        cat > "$example_dir/modules/resource-group/main.tf" << 'EOF'
terraform {
  required_version = ">= 1.10"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.10"
    }
  }
}

variable "name" {
  description = "Name of the resource group"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

resource "azurerm_resource_group" "this" {
  name     = var.name
  location = var.location
  tags     = var.tags
}

output "id" {
  description = "Resource group ID"
  value       = azurerm_resource_group.this.id
}

output "name" {
  description = "Resource group name"
  value       = azurerm_resource_group.this.name
}
EOF
        
        # Create dev environment
        cat > "$example_dir/environments/dev/main.tf" << 'EOF'
terraform {
  required_version = ">= 1.10"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.10"
    }
  }
  
  backend "azurerm" {
    # Configure backend in backend.tf or via CLI
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = true
    }
  }
}

locals {
  environment = "dev"
  location    = "eastus2"
  
  tags = {
    Environment      = local.environment
    ManagedBy        = "Terraform"
    LastUpdated      = timestamp()
  }
}

module "resource_group" {
  source = "../../modules/resource-group"
  
  name     = "rg-example-${local.environment}-${local.location}"
  location = local.location
  tags     = local.tags
}
EOF
        
        # Create variables file
        cat > "$example_dir/environments/dev/variables.tf" << 'EOF'
variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
  sensitive   = true
}

variable "tenant_id" {
  description = "Azure tenant ID"
  type        = string
  sensitive   = true
}
EOF
        
        # Create .gitignore
        cat > "$example_dir/.gitignore" << 'EOF'
# Terraform files
*.tfstate
*.tfstate.*
.terraform/
.terraform.lock.hcl
terraform.tfvars
*.auto.tfvars
override.tf
override.tf.json
*_override.tf
*_override.tf.json

# Azure credentials
.env
.env.local
azure.conf

# IDE files
.idea/
.vscode/
*.swp
*.swo
*~

# OS files
.DS_Store
Thumbs.db
EOF
        
        # Create README
        cat > "$example_dir/README.md" << 'EOF'
# Terraform Azure Example

This is an example Terraform project structure for Azure.

## Structure

```
.
├── environments/          # Environment-specific configurations
│   ├── dev/
│   ├── staging/
│   └── production/
├── modules/              # Reusable Terraform modules
│   └── resource-group/
└── README.md
```

## Usage

1. Navigate to an environment directory:
   ```bash
   cd environments/dev
   ```

2. Initialize Terraform:
   ```bash
   terraform init
   ```

3. Plan changes:
   ```bash
   terraform plan
   ```

4. Apply changes:
   ```bash
   terraform apply
   ```

## Best Practices

- Use remote state storage (Azure Storage)
- Enable state locking
- Use workspaces sparingly
- Follow consistent naming conventions
- Tag all resources appropriately
EOF
        
        log_success "Created example project in $example_dir"
    fi
    
    # Setup shell aliases
    log_step "Setting up shell aliases"
    if confirm "Add Terraform aliases to shell configuration?"; then
        aliases=(
            "alias tf='terraform'"
            "alias tfi='terraform init'"
            "alias tfp='terraform plan'"
            "alias tfa='terraform apply'"
            "alias tfd='terraform destroy'"
            "alias tfv='terraform validate'"
            "alias tff='terraform fmt -recursive'"
            "alias tfs='terraform show'"
            "alias tfw='terraform workspace'"
        )
        
        for alias_cmd in "${aliases[@]}"; do
            if [[ -f "$HOME/.bashrc" ]]; then
                add_to_file_if_missing "$HOME/.bashrc" "$alias_cmd" "Terraform alias"
            fi
            
            if [[ -f "$HOME/.zshrc" ]]; then
                add_to_file_if_missing "$HOME/.zshrc" "$alias_cmd" "Terraform alias"
            fi
        done
        
        log_success "Added Terraform aliases"
    fi
    
    # Setup Git hooks
    log_step "Setting up Git hooks for Terraform"
    if confirm "Create pre-commit configuration for Terraform?"; then
        if command_exists pre-commit || confirm "Install pre-commit first?"; then
            if ! command_exists pre-commit; then
                pip3 install --user pre-commit
            fi
            
            # Create pre-commit config in example directory
            if [[ -d "$HOME/terraform-azure-example" ]]; then
                cat > "$HOME/terraform-azure-example/.pre-commit-config.yaml" << 'EOF'
repos:
  - repo: https://github.com/antonbabenko/pre-commit-terraform
    rev: v1.96.2
    hooks:
      - id: terraform_fmt
      - id: terraform_validate
      - id: terraform_docs
      - id: terraform_tflint
      - id: terraform_tfsec
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
EOF
                log_success "Created .pre-commit-config.yaml"
            fi
        fi
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "terraform" "version"
    show_tool_status "az" "version --output table"
    show_tool_status "tflint" "--version"
    show_tool_status "tfsec" "--version"
    show_tool_status "terragrunt" "--version"
    show_tool_status "terraform-docs" "--version"
    show_tool_status "infracost" "--version"
    
    echo
    log_success "Terraform with Azure development environment is ready!"
    log_info "To get started:"
    echo -e "  ${CYAN}cd ~/terraform-azure-example/environments/dev${RESET}"
    echo -e "  ${CYAN}terraform init${RESET}"
    echo -e "  ${CYAN}az login${RESET}"
    echo -e "  ${CYAN}terraform plan${RESET}"
}

# Installation functions
install_terraform() {
    # Add HashiCorp GPG key
    wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
    
    # Add HashiCorp repository
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
    
    # Update and install
    sudo apt-get update
    sudo apt-get install terraform=$TERRAFORM_VERSION-1
    
    log_success "Terraform $TERRAFORM_VERSION installed"
}

install_azure_cli() {
    # Install via Microsoft's script
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
    log_success "Azure CLI installed/updated"
}

install_tflint() {
    local url="https://github.com/terraform-linters/tflint/releases/download/v${TFLINT_VERSION}/tflint_linux_amd64.zip"
    local temp_dir=$(mktemp -d)
    
    cd "$temp_dir"
    wget -q "$url" -O tflint.zip
    unzip -q tflint.zip
    sudo mv tflint /usr/local/bin/
    cd - > /dev/null
    rm -rf "$temp_dir"
    
    log_success "tflint $TFLINT_VERSION installed"
}

install_tfsec() {
    local url="https://github.com/aquasecurity/tfsec/releases/download/v${TFSEC_VERSION}/tfsec-linux-amd64"
    
    sudo wget -qO /usr/local/bin/tfsec "$url"
    sudo chmod +x /usr/local/bin/tfsec
    
    log_success "tfsec $TFSEC_VERSION installed"
}

install_terragrunt() {
    local url="https://github.com/gruntwork-io/terragrunt/releases/download/v${TERRAGRUNT_VERSION}/terragrunt_linux_amd64"
    
    sudo wget -qO /usr/local/bin/terragrunt "$url"
    sudo chmod +x /usr/local/bin/terragrunt
    
    log_success "Terragrunt $TERRAGRUNT_VERSION installed"
}

install_terraform_docs() {
    local url="https://terraform-docs.io/dl/v0.19.0/terraform-docs-v0.19.0-linux-amd64.tar.gz"
    local temp_dir=$(mktemp -d)
    
    cd "$temp_dir"
    wget -q "$url" -O terraform-docs.tar.gz
    tar -xzf terraform-docs.tar.gz
    sudo mv terraform-docs /usr/local/bin/
    cd - > /dev/null
    rm -rf "$temp_dir"
    
    log_success "terraform-docs installed"
}

install_terrascan() {
    local url="https://github.com/tenable/terrascan/releases/latest/download/terrascan_Linux_x86_64.tar.gz"
    local temp_dir=$(mktemp -d)
    
    cd "$temp_dir"
    wget -q "$url" -O terrascan.tar.gz
    tar -xzf terrascan.tar.gz
    sudo mv terrascan /usr/local/bin/
    cd - > /dev/null
    rm -rf "$temp_dir"
    
    log_success "terrascan installed"
}

install_infracost() {
    curl -fsSL https://raw.githubusercontent.com/infracost/infracost/master/scripts/install.sh | sh
    log_success "Infracost installed"
}

# Run main function
main "$@" 