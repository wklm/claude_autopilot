#!/usr/bin/env bash

# Common utilities for setup scripts
# Compatible with both bash and zsh

# Color definitions that work in both bash and zsh
if [[ -t 1 ]]; then
    # Terminal supports colors
    export RED='\033[0;31m'
    export GREEN='\033[0;32m'
    export YELLOW='\033[0;33m'
    export BLUE='\033[0;34m'
    export MAGENTA='\033[0;35m'
    export CYAN='\033[0;36m'
    export WHITE='\033[0;37m'
    export BOLD='\033[1m'
    export RESET='\033[0m'
else
    # No color support
    export RED=''
    export GREEN=''
    export YELLOW=''
    export BLUE=''
    export MAGENTA=''
    export CYAN=''
    export WHITE=''
    export BOLD=''
    export RESET=''
fi

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${RESET} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${RESET} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${RESET} $1"
}

log_step() {
    echo -e "\n${MAGENTA}==>${RESET} ${BOLD}$1${RESET}"
}

# Ask for user confirmation
confirm() {
    local prompt="${1:-Do you want to continue?}"
    local response
    
    while true; do
        echo -en "${CYAN}$prompt${RESET} [${GREEN}y${RESET}/${RED}n${RESET}]: "
        read -r response
        case "$response" in
            [yY][eE][sS]|[yY]) return 0 ;;
            [nN][oO]|[nN]) return 1 ;;
            *) log_warning "Please answer yes or no." ;;
        esac
    done
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get the version of a command if it exists
get_version() {
    local cmd="$1"
    local version_flag="${2:---version}"
    
    if command_exists "$cmd"; then
        "$cmd" "$version_flag" 2>/dev/null | head -n1
    else
        echo "Not installed"
    fi
}

# Check Ubuntu version
check_ubuntu_version() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        if [[ "$ID" != "ubuntu" ]]; then
            log_warning "This script is designed for Ubuntu, but you're running $PRETTY_NAME"
            if ! confirm "Do you want to continue anyway?"; then
                return 1
            fi
        else
            log_info "Detected Ubuntu version: $VERSION"
        fi
    else
        log_warning "Cannot detect OS version. This script is designed for Ubuntu."
        if ! confirm "Do you want to continue anyway?"; then
            return 1
        fi
    fi
    return 0
}

# Update package lists if needed (with caching)
update_apt_if_needed() {
    local apt_update_file="/tmp/apt_updated_$(date +%Y%m%d)"
    
    if [[ ! -f "$apt_update_file" ]]; then
        log_step "Updating package lists"
        if confirm "Update apt package lists? (recommended if you haven't done it today)"; then
            if sudo apt-get update; then
                touch "$apt_update_file"
                log_success "Package lists updated"
            else
                log_error "Failed to update package lists"
                return 1
            fi
        else
            log_info "Skipping apt update"
        fi
    else
        log_info "Package lists already updated today"
    fi
    return 0
}

# Install a package via apt if not already installed
install_apt_package() {
    local package="$1"
    local package_name="${2:-$package}"  # Display name (optional)
    
    if dpkg -l | grep -q "^ii  $package "; then
        log_info "$package_name is already installed"
        return 0
    fi
    
    log_step "Installing $package_name"
    if confirm "Install $package_name via apt?"; then
        if sudo apt-get install -y "$package"; then
            log_success "$package_name installed successfully"
            return 0
        else
            log_error "Failed to install $package_name"
            return 1
        fi
    else
        log_warning "Skipping $package_name installation"
        return 1
    fi
}

# Create a directory with proper permissions
create_directory() {
    local dir="$1"
    local description="${2:-directory}"
    
    if [[ -d "$dir" ]]; then
        log_info "$description already exists: $dir"
        return 0
    fi
    
    log_step "Creating $description"
    if confirm "Create $description at $dir?"; then
        if mkdir -p "$dir"; then
            log_success "$description created: $dir"
            return 0
        else
            log_error "Failed to create $description: $dir"
            return 1
        fi
    else
        log_warning "Skipping $description creation"
        return 1
    fi
}

# Add a line to a file if it doesn't exist
add_to_file_if_missing() {
    local file="$1"
    local line="$2"
    local description="${3:-configuration}"
    
    if [[ -f "$file" ]] && grep -Fxq "$line" "$file"; then
        log_info "$description already configured in $file"
        return 0
    fi
    
    log_step "Adding $description to $file"
    echo -e "${CYAN}Will add:${RESET} $line"
    
    if confirm "Add this $description?"; then
        echo "$line" >> "$file"
        log_success "$description added to $file"
        return 0
    else
        log_warning "Skipping $description"
        return 1
    fi
}

# Display a banner
show_banner() {
    local title="$1"
    local width=60
    local padding=$(( (width - ${#title}) / 2 ))
    
    echo
    echo -e "${BOLD}${CYAN}$(printf '=%.0s' $(seq 1 $width))${RESET}"
    echo -e "${BOLD}${CYAN}$(printf ' %.0s' $(seq 1 $padding))$title${RESET}"
    echo -e "${BOLD}${CYAN}$(printf '=%.0s' $(seq 1 $width))${RESET}"
    echo
}

# Display current tool versions
show_tool_status() {
    local tool="$1"
    local version_cmd="${2:---version}"
    local current_version
    
    current_version=$(get_version "$tool" "$version_cmd")
    if [[ "$current_version" == "Not installed" ]]; then
        echo -e "  ${RED}✗${RESET} $tool: ${RED}Not installed${RESET}"
    else
        echo -e "  ${GREEN}✓${RESET} $tool: ${GREEN}$current_version${RESET}"
    fi
}

# Export functions for use in other scripts
export -f log_info log_success log_warning log_error log_step
export -f confirm command_exists get_version
export -f check_ubuntu_version update_apt_if_needed
export -f install_apt_package create_directory
export -f add_to_file_if_missing show_banner show_tool_status 