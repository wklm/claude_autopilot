#!/usr/bin/env bash

# Setup script for React Native development environment
# Installs: Node.js 22+, React Native CLI, Android SDK, iOS dependencies

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
NODE_VERSION="22"
JAVA_VERSION="17"

main() {
    show_banner "React Native Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "node" "--version"
    show_tool_status "npm" "--version"
    show_tool_status "npx" "--version"
    show_tool_status "java" "-version 2>&1 | head -n 1"
    show_tool_status "react-native" "--version"
    show_tool_status "watchman" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "unzip" "Unzip"
    install_apt_package "wget" "Wget"
    
    # Install Node.js
    log_step "Installing Node.js"
    if command_exists node; then
        current_node=$(node --version | sed 's/v//' | cut -d. -f1)
        log_info "Current Node.js version: v$(node --version | sed 's/v//')"
        
        if [[ "$current_node" -lt "$NODE_VERSION" ]]; then
            log_warning "Node.js v$current_node is older than recommended v$NODE_VERSION"
            if confirm "Install Node.js v$NODE_VERSION via NodeSource?"; then
                install_nodejs
            fi
        else
            log_success "Node.js version is sufficient"
        fi
    else
        if confirm "Install Node.js v$NODE_VERSION?"; then
            install_nodejs
        else
            log_error "Node.js is required to continue"
            exit 1
        fi
    fi
    
    # Install React Native CLI
    log_step "Installing React Native CLI"
    if ! command_exists react-native; then
        if confirm "Install React Native CLI globally?"; then
            npm install -g react-native-cli
            log_success "React Native CLI installed"
        fi
    else
        log_info "React Native CLI is already installed"
        if confirm "Update React Native CLI to latest version?"; then
            npm update -g react-native-cli
            log_success "React Native CLI updated"
        fi
    fi
    
    # Install Expo CLI
    log_step "Installing Expo CLI"
    if ! command_exists expo; then
        if confirm "Install Expo CLI (alternative to React Native CLI)?"; then
            npm install -g expo-cli
            log_success "Expo CLI installed"
        fi
    else
        log_info "Expo CLI is already installed"
        if confirm "Update Expo CLI to latest version?"; then
            npm update -g expo-cli
            log_success "Expo CLI updated"
        fi
    fi
    
    # Install Watchman (for file watching)
    log_step "Installing Watchman"
    if ! command_exists watchman; then
        if confirm "Install Watchman (recommended for React Native)?"; then
            install_watchman
        fi
    else
        log_info "Watchman is already installed"
    fi
    
    # Install Java for Android development
    log_step "Installing Java for Android development"
    if ! command_exists java; then
        if confirm "Install OpenJDK $JAVA_VERSION for Android development?"; then
            install_apt_package "openjdk-${JAVA_VERSION}-jdk" "OpenJDK $JAVA_VERSION"
        fi
    else
        log_info "Java is already installed"
    fi
    
    # Android SDK setup
    log_step "Android SDK setup"
    if confirm "Set up Android SDK (required for Android development)?"; then
        setup_android_sdk
    fi
    
    # Install development tools
    log_step "Installing development tools"
    
    # TypeScript
    if ! command_exists tsc; then
        if confirm "Install TypeScript globally?"; then
            npm install -g typescript
            log_success "TypeScript installed"
        fi
    else
        log_info "TypeScript is already installed"
    fi
    
    # ESLint
    if ! command_exists eslint; then
        if confirm "Install ESLint globally?"; then
            npm install -g eslint
            log_success "ESLint installed"
        fi
    else
        log_info "ESLint is already installed"
    fi
    
    # Prettier
    if ! command_exists prettier; then
        if confirm "Install Prettier globally?"; then
            npm install -g prettier
            log_success "Prettier installed"
        fi
    else
        log_info "Prettier is already installed"
    fi
    
    # Flipper (debugging tool)
    log_step "Flipper setup"
    if confirm "Download Flipper (React Native debugging tool)?"; then
        log_info "Please download Flipper from: https://fbflipper.com/"
        log_info "Install it manually after the script completes"
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "node" "--version"
    show_tool_status "npm" "--version"
    show_tool_status "react-native" "--version"
    show_tool_status "expo" "--version"
    show_tool_status "watchman" "--version"
    show_tool_status "java" "-version 2>&1 | head -n 1"
    show_tool_status "tsc" "--version"
    show_tool_status "eslint" "--version"
    show_tool_status "prettier" "--version"
    
    echo
    log_success "React Native development environment is ready!"
    log_info "To create a new React Native project:"
    echo -e "  ${CYAN}npx react-native init MyApp --template react-native-template-typescript${RESET}"
    echo -e "  ${CYAN}cd MyApp${RESET}"
    echo -e "  ${CYAN}npx react-native run-android${RESET} or ${CYAN}npx react-native run-ios${RESET}"
    echo
    log_info "Or with Expo:"
    echo -e "  ${CYAN}npx create-expo-app MyApp --template${RESET}"
    echo -e "  ${CYAN}cd MyApp${RESET}"
    echo -e "  ${CYAN}npx expo start${RESET}"
}

install_nodejs() {
    log_info "Installing Node.js v$NODE_VERSION via NodeSource repository..."
    
    # Download and execute NodeSource setup script
    curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | sudo -E bash -
    
    # Install Node.js
    install_apt_package "nodejs" "Node.js"
    
    # Verify installation
    if command_exists node; then
        log_success "Node.js $(node --version) installed successfully"
    else
        log_error "Failed to install Node.js"
        return 1
    fi
}

install_watchman() {
    log_info "Installing Watchman from source..."
    
    # Install dependencies
    install_apt_package "libssl-dev" "OpenSSL Dev"
    install_apt_package "autoconf" "Autoconf"
    install_apt_package "automake" "Automake"
    install_apt_package "libtool" "Libtool"
    install_apt_package "pkg-config" "pkg-config"
    install_apt_package "libpython3-dev" "Python 3 Dev"
    
    # Clone and build Watchman
    temp_dir=$(mktemp -d)
    cd "$temp_dir"
    
    git clone https://github.com/facebook/watchman.git
    cd watchman
    git checkout v2023.01.30.00  # Latest stable version
    ./autogen.sh
    ./configure
    make
    sudo make install
    
    cd "$SCRIPT_DIR"
    rm -rf "$temp_dir"
    
    if command_exists watchman; then
        log_success "Watchman installed successfully"
    else
        log_error "Failed to install Watchman"
        return 1
    fi
}

setup_android_sdk() {
    log_info "Setting up Android SDK..."
    
    # Create Android SDK directory
    ANDROID_HOME="$HOME/Android/Sdk"
    mkdir -p "$ANDROID_HOME"
    
    # Download command line tools
    log_info "Downloading Android command line tools..."
    cd "$ANDROID_HOME"
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip -O cmdline-tools.zip
    
    # Extract tools
    unzip -q cmdline-tools.zip
    rm cmdline-tools.zip
    mkdir -p cmdline-tools/latest
    mv cmdline-tools/* cmdline-tools/latest/ 2>/dev/null || true
    
    # Set up environment variables
    android_env="
# Android SDK
export ANDROID_HOME=$ANDROID_HOME
export PATH=\$PATH:\$ANDROID_HOME/cmdline-tools/latest/bin
export PATH=\$PATH:\$ANDROID_HOME/platform-tools
export PATH=\$PATH:\$ANDROID_HOME/emulator"
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$android_env" "Android SDK"
    fi
    
    if [[ -f "$HOME/.zshrc" ]]; then
        add_to_file_if_missing "$HOME/.zshrc" "$android_env" "Android SDK"
    fi
    
    # Source for current session
    export ANDROID_HOME="$ANDROID_HOME"
    export PATH="$PATH:$ANDROID_HOME/cmdline-tools/latest/bin"
    export PATH="$PATH:$ANDROID_HOME/platform-tools"
    export PATH="$PATH:$ANDROID_HOME/emulator"
    
    # Accept licenses
    log_info "Accepting Android SDK licenses..."
    yes | "$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" --licenses >/dev/null 2>&1 || true
    
    # Install essential SDK components
    log_info "Installing Android SDK components..."
    "$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" \
        "platform-tools" \
        "platforms;android-33" \
        "build-tools;33.0.0" \
        "system-images;android-33;google_apis;x86_64" \
        "emulator"
    
    log_success "Android SDK setup complete"
    log_info "You may need to restart your shell or run: source ~/.bashrc"
    
    cd "$SCRIPT_DIR"
}

# Run main function
main "$@" 