#!/usr/bin/env bash

# Setup script for Bash/Zsh scripting development environment
# Installs: ShellCheck, shfmt, bash-completion, zsh plugins, and shell development tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
SHELLCHECK_VERSION="0.10.0"
SHFMT_VERSION="3.8.0"

main() {
    show_banner "Bash/Zsh Scripting Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "bash" "--version | head -1"
    show_tool_status "zsh" "--version"
    show_tool_status "shellcheck" "--version"
    show_tool_status "shfmt" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "build-essential" "Build Essential"
    
    # Install/update Bash
    log_step "Checking Bash installation"
    if ! command_exists bash; then
        install_apt_package "bash" "Bash"
    else
        current_bash=$(bash --version | head -1 | awk '{print $4}' | cut -d'(' -f1)
        log_info "Current Bash version: $current_bash"
        
        # Check if version is 5.2 or higher
        if [[ "$(printf '%s\n' "5.2" "$current_bash" | sort -V | head -n1)" != "5.2" ]]; then
            log_warning "Bash $current_bash is older than recommended 5.2+"
            if confirm "Update Bash to latest version?"; then
                install_apt_package "bash" "Bash (latest)"
            fi
        fi
    fi
    
    # Install bash-completion
    if ! dpkg -l | grep -q "^ii.*bash-completion"; then
        if confirm "Install bash-completion for better tab completion?"; then
            install_apt_package "bash-completion" "Bash Completion"
        fi
    else
        log_info "bash-completion is already installed"
    fi
    
    # Install/update Zsh
    log_step "Checking Zsh installation"
    if ! command_exists zsh; then
        if confirm "Install Zsh?"; then
            install_apt_package "zsh" "Zsh"
            install_apt_package "zsh-syntax-highlighting" "Zsh Syntax Highlighting"
            install_apt_package "zsh-autosuggestions" "Zsh Autosuggestions"
        fi
    else
        current_zsh=$(zsh --version | awk '{print $2}')
        log_info "Current Zsh version: $current_zsh"
    fi
    
    # Install ShellCheck
    log_step "Installing ShellCheck (shell script linter)"
    if command_exists shellcheck; then
        current_sc=$(shellcheck --version | grep "version:" | awk '{print $2}')
        log_info "Current ShellCheck version: $current_sc"
        
        if [[ "$current_sc" != "$SHELLCHECK_VERSION" ]]; then
            if confirm "Update ShellCheck to version $SHELLCHECK_VERSION?"; then
                install_shellcheck
            fi
        fi
    else
        if confirm "Install ShellCheck for shell script linting?"; then
            install_shellcheck
        fi
    fi
    
    # Install shfmt
    log_step "Installing shfmt (shell script formatter)"
    if command_exists shfmt; then
        current_shfmt=$(shfmt --version || echo "unknown")
        log_info "Current shfmt version: $current_shfmt"
        
        if [[ "$current_shfmt" != "v$SHFMT_VERSION" ]]; then
            if confirm "Update shfmt to version $SHFMT_VERSION?"; then
                install_shfmt
            fi
        fi
    else
        if confirm "Install shfmt for shell script formatting?"; then
            install_shfmt
        fi
    fi
    
    # Install additional shell development tools
    log_step "Installing additional shell development tools"
    
    # bats-core for testing
    if ! command_exists bats; then
        if confirm "Install bats-core (Bash Automated Testing System)?"; then
            install_apt_package "bats" "BATS"
        fi
    else
        log_info "bats is already installed"
    fi
    
    # jq for JSON processing
    if ! command_exists jq; then
        if confirm "Install jq (JSON processor)?"; then
            install_apt_package "jq" "jq"
        fi
    else
        log_info "jq is already installed"
    fi
    
    # yq for YAML processing
    if ! command_exists yq; then
        if confirm "Install yq (YAML processor)?"; then
            sudo wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
            sudo chmod +x /usr/local/bin/yq
            log_success "yq installed"
        fi
    else
        log_info "yq is already installed"
    fi
    
    # fzf for fuzzy finding
    if ! command_exists fzf; then
        if confirm "Install fzf (fuzzy finder)?"; then
            install_apt_package "fzf" "fzf"
        fi
    else
        log_info "fzf is already installed"
    fi
    
    # Setup shell configurations
    log_step "Configuring shell environments"
    
    # Bash configuration
    if confirm "Add modern Bash configuration to ~/.bashrc?"; then
        bash_config="$HOME/.bashrc"
        
        # Backup existing config
        cp "$bash_config" "$bash_config.backup.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
        
        # Add modern bash settings
        cat >> "$bash_config" << 'EOF'

# Modern Bash Configuration (added by setup script)
# Enable extended globbing and modern bash features
shopt -s extglob globstar nullglob dotglob 2>/dev/null || true
shopt -s inherit_errexit errtrace 2>/dev/null || true

# History configuration
export HISTSIZE=10000
export HISTFILESIZE=20000
export HISTCONTROL=ignoreboth:erasedups
shopt -s histappend

# Better tab completion
bind "set completion-ignore-case on"
bind "set show-all-if-ambiguous on"
bind "set mark-symlinked-directories on"

# Safety aliases
alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'

# Useful aliases for script development
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias sc='shellcheck'
alias shf='shfmt -w'

# Function to create a new script with best practices
newscript() {
    local name="${1:-script.sh}"
    cat > "$name" << 'SCRIPT_EOF'
#!/usr/bin/env -S bash -euo pipefail

# Script: ${name}
# Description: TODO
# Author: $(whoami)
# Date: $(date +%Y-%m-%d)

set -euo pipefail

# Global variables
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="${0##*/}"

# Error handler
error_handler() {
    local line_no=$1
    echo "Error on line $line_no" >&2
    exit 1
}

trap 'error_handler ${LINENO}' ERR

# Main function
main() {
    echo "Hello from $SCRIPT_NAME"
}

# Run main function
main "$@"
SCRIPT_EOF
    chmod +x "$name"
    echo "Created $name with best practices template"
}

# Load bash-completion if available
[[ -r /usr/share/bash-completion/bash_completion ]] && . /usr/share/bash-completion/bash_completion
EOF
        log_success "Bash configuration updated"
    fi
    
    # Zsh configuration (if installed)
    if command_exists zsh && confirm "Add modern Zsh configuration to ~/.zshrc?"; then
        zsh_config="$HOME/.zshrc"
        
        # Backup existing config
        cp "$zsh_config" "$zsh_config.backup.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
        
        # Add modern zsh settings
        cat >> "$zsh_config" << 'EOF'

# Modern Zsh Configuration (added by setup script)
# Enable recommended options
setopt EXTENDED_GLOB NULL_GLOB DOT_GLOB
setopt PIPE_FAIL ERR_RETURN ERR_EXIT
setopt NO_UNSET WARN_CREATE_GLOBAL
setopt HIST_IGNORE_DUPS HIST_IGNORE_SPACE
setopt SHARE_HISTORY HIST_REDUCE_BLANKS

# Load plugins
[[ -r /usr/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh ]] && source /usr/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
[[ -r /usr/share/zsh-autosuggestions/zsh-autosuggestions.zsh ]] && source /usr/share/zsh-autosuggestions/zsh-autosuggestions.zsh

# Aliases (same as bash)
alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias sc='shellcheck'
alias shf='shfmt -w'
EOF
        log_success "Zsh configuration updated"
    fi
    
    # Setup global gitignore for shell scripts
    log_step "Configuring Git for shell script development"
    if confirm "Add shell script patterns to global .gitignore?"; then
        gitignore_file="$HOME/.gitignore_global"
        touch "$gitignore_file"
        
        # Shell script patterns
        shell_patterns=(
            "*.swp"
            "*.swo"
            "*~"
            ".#*"
            "*.bak"
            "*.tmp"
            "*.log"
            ".shellcheckrc"
            ".shfmt"
        )
        
        for pattern in "${shell_patterns[@]}"; do
            if ! grep -Fxq "$pattern" "$gitignore_file"; then
                echo "$pattern" >> "$gitignore_file"
            fi
        done
        
        git config --global core.excludesfile "$gitignore_file"
        log_success "Global .gitignore configured for shell scripts"
    fi
    
    # Create ShellCheck configuration
    if confirm "Create default .shellcheckrc configuration?"; then
        cat > "$HOME/.shellcheckrc" << 'EOF'
# ShellCheck configuration
# https://www.shellcheck.net/wiki/Ignore

# Enable all optional checks
enable=all

# Specific checks to enable
enable=quote-safe-variables
enable=require-variable-braces

# Exclude specific checks if needed
# disable=SC2034  # Unused variables
EOF
        log_success "Created ~/.shellcheckrc"
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "bash" "--version | head -1"
    show_tool_status "zsh" "--version"
    show_tool_status "shellcheck" "--version"
    show_tool_status "shfmt" "--version"
    show_tool_status "bats" "--version"
    show_tool_status "jq" "--version"
    
    echo
    log_success "Bash/Zsh scripting development environment is ready!"
    log_info "To test your setup, try:"
    echo -e "  ${CYAN}shellcheck your-script.sh${RESET}"
    echo -e "  ${CYAN}shfmt -w your-script.sh${RESET}"
    echo -e "  ${CYAN}newscript my-script.sh${RESET} (if using bash)"
}

# Function to install ShellCheck
install_shellcheck() {
    local sc_url="https://github.com/koalaman/shellcheck/releases/download/v${SHELLCHECK_VERSION}/shellcheck-v${SHELLCHECK_VERSION}.linux.x86_64.tar.xz"
    local temp_dir=$(mktemp -d)
    
    cd "$temp_dir"
    wget -q "$sc_url" -O shellcheck.tar.xz
    tar -xf shellcheck.tar.xz
    sudo mv "shellcheck-v${SHELLCHECK_VERSION}/shellcheck" /usr/local/bin/
    cd - > /dev/null
    rm -rf "$temp_dir"
    
    log_success "ShellCheck $SHELLCHECK_VERSION installed"
}

# Function to install shfmt
install_shfmt() {
    local shfmt_url="https://github.com/mvdan/sh/releases/download/v${SHFMT_VERSION}/shfmt_v${SHFMT_VERSION}_linux_amd64"
    
    sudo wget -qO /usr/local/bin/shfmt "$shfmt_url"
    sudo chmod +x /usr/local/bin/shfmt
    
    log_success "shfmt $SHFMT_VERSION installed"
}

# Run main function
main "$@" 