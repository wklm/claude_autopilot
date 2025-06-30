#!/usr/bin/env bash

# Setup script for Go web application development environment
# Installs: Go 1.23+, golangci-lint, air, migrate, and other Go tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
GO_VERSION="1.23.5"
GOLANGCI_LINT_VERSION="latest"

main() {
    show_banner "Go Web Application Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "go" "version"
    show_tool_status "golangci-lint" "--version"
    show_tool_status "air" "-v"
    show_tool_status "migrate" "-version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "wget" "wget"
    
    # Install Go
    log_step "Installing Go"
    if command_exists go; then
        current_go=$(go version | awk '{print $3}' | sed 's/go//')
        log_info "Current Go version: $current_go"
        
        if [[ "$current_go" != "$GO_VERSION" ]]; then
            log_warning "Go $current_go is installed, but $GO_VERSION is recommended"
            if confirm "Install Go $GO_VERSION?"; then
                install_go
            fi
        else
            log_success "Go $GO_VERSION is already installed"
        fi
    else
        if confirm "Install Go $GO_VERSION?"; then
            install_go
        else
            log_error "Go is required to continue"
            exit 1
        fi
    fi
    
    # Setup Go environment
    setup_go_environment
    
    # Install golangci-lint
    log_step "Installing golangci-lint"
    if command_exists golangci-lint; then
        log_info "golangci-lint is already installed"
        if confirm "Update golangci-lint to latest version?"; then
            curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin
            log_success "golangci-lint updated"
        fi
    else
        if confirm "Install golangci-lint (comprehensive Go linter)?"; then
            curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin
            log_success "golangci-lint installed"
        fi
    fi
    
    # Install air for hot reload
    log_step "Installing development tools"
    if ! command_exists air; then
        if confirm "Install air (Live reload for Go apps)?"; then
            go install github.com/air-verse/air@latest
            log_success "air installed"
        fi
    else
        log_info "air is already installed"
    fi
    
    # Install golang-migrate
    if ! command_exists migrate; then
        if confirm "Install golang-migrate (Database migration tool)?"; then
            go install -tags 'postgres mysql sqlite3' github.com/golang-migrate/migrate/v4/cmd/migrate@latest
            log_success "migrate installed"
        fi
    else
        log_info "migrate is already installed"
    fi
    
    # Install mockery for test mocks
    if ! command_exists mockery; then
        if confirm "Install mockery (Mock generator for Go)?"; then
            go install github.com/vektra/mockery/v2@latest
            log_success "mockery installed"
        fi
    else
        log_info "mockery is already installed"
    fi
    
    # Install go-task as alternative to make
    if ! command_exists task; then
        if confirm "Install Task (Modern task runner, alternative to Make)?"; then
            go install github.com/go-task/task/v3/cmd/task@latest
            log_success "task installed"
        fi
    else
        log_info "task is already installed"
    fi
    
    # Install swag for Swagger docs
    if ! command_exists swag; then
        if confirm "Install swag (Swagger documentation generator)?"; then
            go install github.com/swaggo/swag/cmd/swag@latest
            log_success "swag installed"
        fi
    else
        log_info "swag is already installed"
    fi
    
    # Install govulncheck for security scanning
    if ! command_exists govulncheck; then
        if confirm "Install govulncheck (Go vulnerability checker)?"; then
            go install golang.org/x/vuln/cmd/govulncheck@latest
            log_success "govulncheck installed"
        fi
    else
        log_info "govulncheck is already installed"
    fi
    
    # Setup global gitignore for Go
    log_step "Configuring Git for Go development"
    if confirm "Add Go patterns to global .gitignore?"; then
        gitignore_file="$HOME/.gitignore_global"
        touch "$gitignore_file"
        
        # Go patterns
        go_patterns=(
            "# Binaries for programs and plugins"
            "*.exe"
            "*.exe~"
            "*.dll"
            "*.so"
            "*.dylib"
            "# Test binary, built with 'go test -c'"
            "*.test"
            "# Output of the go coverage tool"
            "*.out"
            "# Go workspace file"
            "go.work"
            "go.work.sum"
            "# Dependency directories"
            "vendor/"
            "# Air tmp directory"
            "tmp/"
            "# Environment variables"
            ".env"
            ".env.local"
            "# IDE specific files"
            ".idea/"
            ".vscode/"
            "*.swp"
            "*.swo"
            "*~"
        )
        
        for pattern in "${go_patterns[@]}"; do
            if ! grep -Fxq "$pattern" "$gitignore_file" 2>/dev/null; then
                echo "$pattern" >> "$gitignore_file"
            fi
        done
        
        git config --global core.excludesfile "$gitignore_file"
        log_success "Global .gitignore configured for Go"
    fi
    
    # Create a sample .golangci.yml
    log_step "Creating sample golangci-lint configuration"
    if confirm "Create a sample .golangci.yml in current directory?"; then
        create_golangci_config
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "go" "version"
    show_tool_status "golangci-lint" "--version"
    show_tool_status "air" "-v"
    show_tool_status "migrate" "-version"
    show_tool_status "mockery" "--version"
    show_tool_status "task" "--version"
    show_tool_status "swag" "--version"
    show_tool_status "govulncheck" "-version"
    
    echo
    log_success "Go web application development environment is ready!"
    log_info "To create a new Go project, run:"
    echo -e "  ${CYAN}mkdir my-project && cd my-project${RESET}"
    echo -e "  ${CYAN}go mod init github.com/username/my-project${RESET}"
    echo -e "  ${CYAN}go get github.com/gofiber/fiber/v3${RESET}"
}

install_go() {
    local go_tar="go${GO_VERSION}.linux-amd64.tar.gz"
    local go_url="https://go.dev/dl/${go_tar}"
    
    log_info "Downloading Go ${GO_VERSION}..."
    wget -q --show-progress "$go_url" -O "/tmp/${go_tar}"
    
    log_info "Installing Go ${GO_VERSION}..."
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf "/tmp/${go_tar}"
    rm "/tmp/${go_tar}"
    
    log_success "Go ${GO_VERSION} installed"
}

setup_go_environment() {
    log_step "Setting up Go environment"
    
    # Go paths
    go_paths=(
        'export PATH=$PATH:/usr/local/go/bin'
        'export PATH=$PATH:$HOME/go/bin'
        'export GOPATH=$HOME/go'
    )
    
    # Add to .bashrc
    if [[ -f "$HOME/.bashrc" ]]; then
        for line in "${go_paths[@]}"; do
            add_to_file_if_missing "$HOME/.bashrc" "$line" "Go path"
        done
    fi
    
    # Add to .zshrc if exists
    if [[ -f "$HOME/.zshrc" ]]; then
        for line in "${go_paths[@]}"; do
            add_to_file_if_missing "$HOME/.zshrc" "$line" "Go path"
        done
    fi
    
    # Export for current session
    export PATH=$PATH:/usr/local/go/bin:$HOME/go/bin
    export GOPATH=$HOME/go
    
    # Create Go workspace directories
    create_directory "$HOME/go/src" "Go source directory"
    create_directory "$HOME/go/bin" "Go binary directory"
    create_directory "$HOME/go/pkg" "Go package directory"
}

create_golangci_config() {
    cat > .golangci.yml << 'EOF'
# golangci-lint configuration for Go web applications
run:
  timeout: 5m
  issues-exit-code: 1
  tests: true
  skip-dirs:
    - vendor
    - tmp

output:
  formats:
    - format: colored-line-number
  print-issued-lines: true
  print-linter-name: true

linters:
  enable-all: true
  disable:
    - exhaustruct     # Too noisy for partial struct initialization
    - depguard        # Overly restrictive
    - gochecknoglobals # Some globals are fine
    - wsl             # Too opinionated on whitespace
    - varnamelen      # Short names are idiomatic in Go

linters-settings:
  revive:
    enable-all-rules: true
    rules:
      - name: cognitive-complexity
        arguments: [15]
  govet:
    enable-all: true
  errcheck:
    check-type-assertions: true
    check-blank: true
  gocyclo:
    min-complexity: 15
  
issues:
  exclude-rules:
    - path: _test\.go
      linters:
        - dupl
        - gosec
EOF
    
    log_success "Created .golangci.yml configuration"
}

# Run main function
main "$@" 