#!/usr/bin/env bash

# Main setup script - Interactive menu for tool installation
# Compatible with both bash and zsh on Ubuntu

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

main() {
    show_banner "Development Environment Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    while true; do
        echo
        log_step "Available Setup Scripts"
        echo
        echo "  ${BOLD}1)${RESET} Python FastAPI       - Python 3.12+, uv, ruff, mypy"
        echo "  ${BOLD}2)${RESET} Go Web Apps         - Go 1.23+, golangci-lint, air, migrate"
        echo "  ${BOLD}3)${RESET} Next.js             - Node.js 22+, Bun, TypeScript, ESLint"
        echo "  ${BOLD}4)${RESET} SvelteKit/Remix/Astro - Vite, Playwright, framework tools"
        echo "  ${BOLD}5)${RESET} Rust Development    - Rust toolchain, cargo tools, web & system"
        echo "  ${BOLD}6)${RESET} Java Enterprise     - Java 21+, Gradle, Maven, Spring Boot"
        echo
        echo "  ${BOLD}0)${RESET} Exit"
        echo
        
        read -p "Select an option (0-6): " choice
        
        case $choice in
            1)
                run_setup "Python FastAPI" "setup_python_fastapi.sh"
                ;;
            2)
                run_setup "Go Web Applications" "setup_go_webapps.sh"
                ;;
            3)
                run_setup "Next.js" "setup_nextjs.sh"
                ;;
            4)
                run_setup "SvelteKit/Remix/Astro" "setup_sveltekit_remix_astro.sh"
                ;;
            5)
                run_setup "Rust Development" "setup_rust.sh"
                ;;
            6)
                run_setup "Java Enterprise" "setup_java_enterprise.sh"
                ;;
            0)
                log_info "Exiting setup..."
                exit 0
                ;;
            *)
                log_error "Invalid option. Please select a number between 0 and 6."
                ;;
        esac
        
        echo
        if ! confirm "Continue with another setup?"; then
            break
        fi
    done
    
    echo
    log_success "Setup complete!"
    log_info "Remember to restart your shell or source your shell config to apply PATH changes:"
    echo -e "  ${CYAN}source ~/.bashrc${RESET}  # for bash"
    echo -e "  ${CYAN}source ~/.zshrc${RESET}   # for zsh"
}

run_setup() {
    local name="$1"
    local script="$2"
    local script_path="$SCRIPT_DIR/$script"
    
    echo
    show_banner "$name Setup"
    
    if [[ -f "$script_path" ]]; then
        log_info "Running $name setup script..."
        bash "$script_path"
    else
        log_error "Setup script not found: $script_path"
        return 1
    fi
}

# Show system information
show_system_info() {
    log_step "System Information"
    echo -e "  ${BOLD}OS:${RESET} $(lsb_release -ds 2>/dev/null || echo "Unknown")"
    echo -e "  ${BOLD}Kernel:${RESET} $(uname -r)"
    echo -e "  ${BOLD}Architecture:${RESET} $(uname -m)"
    echo -e "  ${BOLD}CPU:${RESET} $(nproc) cores"
    echo -e "  ${BOLD}Memory:${RESET} $(free -h | awk '/^Mem:/ {print $2}') total"
    echo -e "  ${BOLD}Disk:${RESET} $(df -h / | awk 'NR==2 {print $4}') available on /"
    echo
}

# Check for required commands
check_prerequisites() {
    local missing=()
    
    for cmd in curl wget git sudo; do
        if ! command_exists "$cmd"; then
            missing+=("$cmd")
        fi
    done
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required commands: ${missing[*]}"
        log_info "Please install them first:"
        echo -e "  ${CYAN}sudo apt-get update && sudo apt-get install -y ${missing[*]}${RESET}"
        return 1
    fi
    
    return 0
}

# Entry point
show_system_info
check_prerequisites || exit 1
main "$@" 