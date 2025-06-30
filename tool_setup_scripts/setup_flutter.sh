#!/usr/bin/env bash

# Setup script for Flutter development environment
# Installs: Flutter SDK, Dart, Android SDK, and related tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions and requirements
FLUTTER_CHANNEL="stable"
ANDROID_CMDLINE_VERSION="latest"

main() {
    show_banner "Flutter Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "flutter" "--version"
    show_tool_status "dart" "--version"
    show_tool_status "adb" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "unzip" "Unzip"
    install_apt_package "xz-utils" "XZ Utils"
    install_apt_package "zip" "Zip"
    install_apt_package "libglu1-mesa" "OpenGL Libraries"
    install_apt_package "clang" "Clang"
    install_apt_package "cmake" "CMake"
    install_apt_package "ninja-build" "Ninja Build"
    install_apt_package "pkg-config" "pkg-config"
    install_apt_package "libgtk-3-dev" "GTK3 Development"
    
    # Install Flutter
    log_step "Installing Flutter SDK"
    FLUTTER_DIR="$HOME/development/flutter"
    
    if [[ -d "$FLUTTER_DIR" ]]; then
        log_info "Flutter SDK already exists at $FLUTTER_DIR"
        if confirm "Update Flutter to latest version?"; then
            cd "$FLUTTER_DIR"
            git pull
            flutter upgrade
            log_success "Flutter updated"
        fi
    else
        if confirm "Install Flutter SDK to $FLUTTER_DIR?"; then
            mkdir -p "$HOME/development"
            cd "$HOME/development"
            git clone https://github.com/flutter/flutter.git -b $FLUTTER_CHANNEL
            
            # Add Flutter to PATH
            flutter_path_line='export PATH="$HOME/development/flutter/bin:$PATH"'
            
            if [[ -f "$HOME/.bashrc" ]]; then
                add_to_file_if_missing "$HOME/.bashrc" "$flutter_path_line" "Flutter PATH"
            fi
            
            if [[ -f "$HOME/.zshrc" ]]; then
                add_to_file_if_missing "$HOME/.zshrc" "$flutter_path_line" "Flutter PATH"
            fi
            
            # Source for current session
            export PATH="$HOME/development/flutter/bin:$PATH"
            
            log_success "Flutter SDK installed"
            log_info "Running flutter doctor..."
            flutter doctor
        fi
    fi
    
    # Install Android SDK (if needed for mobile development)
    if confirm "Setup Android development (for mobile apps)?"; then
        log_step "Installing Android SDK"
        
        # Install Java
        install_apt_package "openjdk-17-jdk" "OpenJDK 17"
        
        # Android SDK via command line tools
        ANDROID_HOME="$HOME/Android/Sdk"
        if [[ ! -d "$ANDROID_HOME" ]]; then
            mkdir -p "$ANDROID_HOME"
            cd "$ANDROID_HOME"
            
            # Download command line tools
            log_info "Downloading Android command line tools..."
            curl -o cmdline-tools.zip https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip
            unzip -q cmdline-tools.zip
            rm cmdline-tools.zip
            
            # Move to proper location
            mkdir -p cmdline-tools/latest
            mv cmdline-tools/* cmdline-tools/latest/ 2>/dev/null || true
            
            # Set up environment variables
            android_env_lines=(
                'export ANDROID_HOME="$HOME/Android/Sdk"'
                'export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"'
                'export PATH="$ANDROID_HOME/platform-tools:$PATH"'
            )
            
            for line in "${android_env_lines[@]}"; do
                if [[ -f "$HOME/.bashrc" ]]; then
                    add_to_file_if_missing "$HOME/.bashrc" "$line" "Android environment"
                fi
                if [[ -f "$HOME/.zshrc" ]]; then
                    add_to_file_if_missing "$HOME/.zshrc" "$line" "Android environment"
                fi
            done
            
            # Source for current session
            export ANDROID_HOME="$HOME/Android/Sdk"
            export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"
            export PATH="$ANDROID_HOME/platform-tools:$PATH"
            
            # Accept licenses and install basic components
            log_info "Installing Android SDK components..."
            yes | sdkmanager --licenses >/dev/null 2>&1 || true
            sdkmanager "platform-tools" "platforms;android-33" "build-tools;33.0.0"
            
            log_success "Android SDK installed"
        else
            log_info "Android SDK already exists at $ANDROID_HOME"
        fi
    fi
    
    # Configure Flutter
    if command_exists flutter; then
        log_step "Configuring Flutter"
        
        # Enable web support
        if confirm "Enable Flutter web support?"; then
            flutter config --enable-web
            log_success "Web support enabled"
        fi
        
        # Run flutter doctor
        log_info "Running flutter doctor to check setup..."
        flutter doctor -v
    fi
    
    # Install VS Code extensions
    if command_exists code; then
        log_step "VS Code Flutter extensions"
        if confirm "Install Flutter and Dart VS Code extensions?"; then
            code --install-extension Dart-Code.dart-code
            code --install-extension Dart-Code.flutter
            log_success "VS Code extensions installed"
        fi
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "flutter" "--version"
    show_tool_status "dart" "--version"
    if [[ -n "${ANDROID_HOME:-}" ]]; then
        show_tool_status "adb" "--version"
        show_tool_status "sdkmanager" "--version"
    fi
    
    echo
    log_success "Flutter development environment is ready!"
    log_info "To create a new Flutter project, run:"
    echo -e "  ${CYAN}flutter create my_app${RESET}"
    echo -e "  ${CYAN}cd my_app${RESET}"
    echo -e "  ${CYAN}flutter run${RESET}"
    
    if command_exists flutter; then
        echo
        log_info "Run 'flutter doctor' to see if any additional setup is needed."
    fi
}

# Run main function
main "$@" 