#!/usr/bin/env bash

# Setup script for Ansible DevOps environment
# Installs: Ansible, ansible-lint, molecule, and related tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions and requirements
PYTHON_MIN_VERSION="3.10"
ANSIBLE_VERSION="latest"

main() {
    show_banner "Ansible DevOps Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "python3" "--version"
    show_tool_status "ansible" "--version"
    show_tool_status "ansible-lint" "--version"
    show_tool_status "molecule" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "python3" "Python 3"
    install_apt_package "python3-pip" "pip"
    install_apt_package "python3-venv" "Python venv"
    install_apt_package "git" "Git"
    install_apt_package "sshpass" "sshpass (for password auth)"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "python3-dev" "Python 3 Development Headers"
    install_apt_package "libssl-dev" "SSL Development Libraries"
    install_apt_package "libffi-dev" "FFI Development Libraries"
    
    # Check Python version
    log_step "Checking Python version"
    if command_exists python3; then
        current_python=$(python3 --version | awk '{print $2}')
        log_info "Current Python version: $current_python"
        
        if [[ "$(printf '%s\n' "$PYTHON_MIN_VERSION" "$current_python" | sort -V | head -n1)" != "$PYTHON_MIN_VERSION" ]]; then
            log_warning "Python $current_python is older than recommended $PYTHON_MIN_VERSION"
        else
            log_success "Python version is sufficient"
        fi
    fi
    
    # Install pipx for isolated tool installations
    log_step "Installing pipx"
    if ! command_exists pipx; then
        if confirm "Install pipx (for isolated Python app installations)?"; then
            python3 -m pip install --user pipx
            python3 -m pipx ensurepath
            
            # Add pipx to PATH for current session
            export PATH="$HOME/.local/bin:$PATH"
            
            log_success "pipx installed"
            log_info "You may need to restart your shell or run: source ~/.bashrc"
        fi
    else
        log_info "pipx is already installed"
    fi
    
    # Install Ansible
    log_step "Installing Ansible"
    if command_exists pipx; then
        if ! command_exists ansible; then
            if confirm "Install Ansible via pipx?"; then
                pipx install --include-deps ansible
                log_success "Ansible installed"
            fi
        else
            log_info "Ansible is already installed"
            if confirm "Upgrade Ansible to latest version?"; then
                pipx upgrade --include-injected ansible
                log_success "Ansible upgraded"
            fi
        fi
        
        # Install ansible-lint
        log_step "Installing Ansible development tools"
        if ! command_exists ansible-lint; then
            if confirm "Install ansible-lint (linting tool)?"; then
                pipx install ansible-lint
                log_success "ansible-lint installed"
            fi
        else
            log_info "ansible-lint is already installed"
            if confirm "Upgrade ansible-lint?"; then
                pipx upgrade ansible-lint
                log_success "ansible-lint upgraded"
            fi
        fi
        
        # Install molecule for testing
        if ! command_exists molecule; then
            if confirm "Install molecule (testing framework)?"; then
                pipx install molecule
                pipx inject molecule molecule-plugins[docker]
                log_success "molecule installed"
            fi
        else
            log_info "molecule is already installed"
        fi
        
        # Install ansible-navigator
        if ! command_exists ansible-navigator; then
            if confirm "Install ansible-navigator (TUI for Ansible)?"; then
                pipx install ansible-navigator
                log_success "ansible-navigator installed"
            fi
        else
            log_info "ansible-navigator is already installed"
        fi
    fi
    
    # Install Docker for molecule testing
    if ! command_exists docker; then
        log_step "Docker setup for molecule testing"
        if confirm "Install Docker (required for molecule testing)?"; then
            # Install Docker dependencies
            install_apt_package "ca-certificates" "CA Certificates"
            install_apt_package "curl" "cURL"
            install_apt_package "gnupg" "GnuPG"
            install_apt_package "lsb-release" "LSB Release"
            
            # Add Docker's official GPG key
            sudo mkdir -p /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            
            # Add Docker repository
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            
            # Update and install Docker
            sudo apt-get update
            install_apt_package "docker-ce" "Docker CE"
            install_apt_package "docker-ce-cli" "Docker CLI"
            install_apt_package "containerd.io" "containerd"
            install_apt_package "docker-compose-plugin" "Docker Compose"
            
            # Add user to docker group
            if confirm "Add current user to docker group (requires logout/login)?"; then
                sudo usermod -aG docker $USER
                log_info "You need to log out and back in for docker group membership to take effect"
            fi
            
            log_success "Docker installed"
        fi
    else
        log_info "Docker is already installed"
    fi
    
    # Create Ansible directory structure
    log_step "Setting up Ansible workspace"
    ANSIBLE_HOME="$HOME/ansible"
    if confirm "Create Ansible workspace at $ANSIBLE_HOME?"; then
        create_directory "$ANSIBLE_HOME" "Ansible workspace"
        create_directory "$ANSIBLE_HOME/playbooks" "Playbooks directory"
        create_directory "$ANSIBLE_HOME/roles" "Roles directory"
        create_directory "$ANSIBLE_HOME/inventories" "Inventories directory"
        create_directory "$ANSIBLE_HOME/group_vars" "Group vars directory"
        create_directory "$ANSIBLE_HOME/host_vars" "Host vars directory"
        
        # Create ansible.cfg
        if confirm "Create default ansible.cfg?"; then
            cat << 'EOF' > "$ANSIBLE_HOME/ansible.cfg"
[defaults]
inventory = ./inventories/hosts
roles_path = ./roles
host_key_checking = False
retry_files_enabled = False
stdout_callback = yaml
callback_whitelist = timer,profile_tasks
gathering = smart
fact_caching = jsonfile
fact_caching_connection = /tmp/ansible_facts_cache
fact_caching_timeout = 86400

[ssh_connection]
pipelining = True
control_path = /tmp/ansible-%%h-%%p-%%r
EOF
            log_success "ansible.cfg created"
        fi
        
        # Create .ansible-lint config
        if confirm "Create .ansible-lint configuration?"; then
            cat << 'EOF' > "$ANSIBLE_HOME/.ansible-lint"
---
exclude_paths:
  - .cache/
  - .github/
skip_list:
  - yaml[line-length]
warn_list:
  - experimental
EOF
            log_success ".ansible-lint created"
        fi
    fi
    
    # VS Code extensions
    if command_exists code; then
        log_step "VS Code Ansible extensions"
        if confirm "Install Ansible VS Code extensions?"; then
            code --install-extension redhat.ansible
            code --install-extension samuelcolvin.jinjahtml
            log_success "VS Code extensions installed"
        fi
    fi
    
    # Setup Git configuration for Ansible
    log_step "Configuring Git for Ansible"
    if confirm "Add Ansible patterns to global .gitignore?"; then
        gitignore_file="$HOME/.gitignore_global"
        touch "$gitignore_file"
        
        # Ansible patterns
        ansible_patterns=(
            "*.retry"
            "*.pyc"
            "__pycache__/"
            ".vault_pass"
            "vault.yml"
            ".ansible/"
            "*.log"
        )
        
        for pattern in "${ansible_patterns[@]}"; do
            if ! grep -Fxq "$pattern" "$gitignore_file"; then
                echo "$pattern" >> "$gitignore_file"
            fi
        done
        
        git config --global core.excludesfile "$gitignore_file"
        log_success "Global .gitignore configured for Ansible"
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "python3" "--version"
    show_tool_status "ansible" "--version"
    show_tool_status "ansible-lint" "--version"
    show_tool_status "molecule" "--version"
    show_tool_status "docker" "--version"
    
    echo
    log_success "Ansible DevOps environment is ready!"
    log_info "To start using Ansible:"
    echo -e "  ${CYAN}cd $ANSIBLE_HOME${RESET}"
    echo -e "  ${CYAN}ansible-galaxy init roles/my-role${RESET}"
    echo -e "  ${CYAN}ansible-playbook -i inventories/hosts playbooks/site.yml${RESET}"
    
    if [[ -d "$ANSIBLE_HOME" ]]; then
        echo
        log_info "Your Ansible workspace is at: $ANSIBLE_HOME"
    fi
}

# Run main function
main "$@" 