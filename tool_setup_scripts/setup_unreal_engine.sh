#!/usr/bin/env bash

# Setup script for Unreal Engine 5 development environment
# Installs: Build dependencies, development tools, C++ environment

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Unreal Engine requirements
MIN_RAM_GB=16
MIN_DISK_GB=200

main() {
    show_banner "Unreal Engine 5 Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Check system requirements
    log_step "Checking system requirements"
    check_system_requirements
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "gcc" "--version | head -n 1"
    show_tool_status "g++" "--version | head -n 1"
    show_tool_status "clang" "--version | head -n 1"
    show_tool_status "cmake" "--version | head -n 1"
    show_tool_status "git" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install build dependencies
    log_step "Installing build dependencies"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "git" "Git"
    install_apt_package "git-lfs" "Git LFS"
    install_apt_package "cmake" "CMake"
    install_apt_package "python3" "Python 3"
    install_apt_package "python3-pip" "Python 3 pip"
    install_apt_package "mono-complete" "Mono"
    install_apt_package "mono-devel" "Mono Development"
    
    # Install Clang (required by UE5)
    log_step "Installing Clang"
    if ! command_exists clang; then
        if confirm "Install Clang 13 (required by Unreal Engine)?"; then
            install_apt_package "clang-13" "Clang 13"
            install_apt_package "lld-13" "LLD 13"
            install_apt_package "libc++-13-dev" "libc++ 13"
            install_apt_package "libc++abi-13-dev" "libc++abi 13"
            
            # Set as default
            sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-13 100
            sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-13 100
        fi
    else
        log_info "Clang is already installed"
    fi
    
    # Install additional development libraries
    log_step "Installing development libraries"
    install_apt_package "libssl-dev" "OpenSSL Dev"
    install_apt_package "libx11-dev" "X11 Dev"
    install_apt_package "libxcursor-dev" "Xcursor Dev"
    install_apt_package "libxrandr-dev" "Xrandr Dev"
    install_apt_package "libxinerama-dev" "Xinerama Dev"
    install_apt_package "libxi-dev" "Xi Dev"
    install_apt_package "libgl1-mesa-dev" "Mesa GL Dev"
    install_apt_package "libglu1-mesa-dev" "Mesa GLU Dev"
    install_apt_package "libasound2-dev" "ALSA Dev"
    install_apt_package "libpulse-dev" "PulseAudio Dev"
    install_apt_package "libfreetype6-dev" "FreeType Dev"
    install_apt_package "libfontconfig1-dev" "FontConfig Dev"
    install_apt_package "libgtk-3-dev" "GTK3 Dev"
    
    # Install Vulkan SDK
    log_step "Installing Vulkan SDK"
    if ! command_exists vulkaninfo; then
        if confirm "Install Vulkan SDK (recommended for modern graphics)?"; then
            install_vulkan_sdk
        fi
    else
        log_info "Vulkan is already installed"
    fi
    
    # Install development tools
    log_step "Installing development tools"
    
    # Code editors and IDEs
    if command_exists code; then
        if confirm "Install VS Code extensions for Unreal Engine?"; then
            code --install-extension ms-vscode.cpptools
            code --install-extension ms-vscode.cmake-tools
            code --install-extension twxs.cmake
            code --install-extension llvm-vs-code-extensions.vscode-clangd
            log_success "VS Code extensions installed"
        fi
    fi
    
    # Install Rider (optional)
    if confirm "Show instructions for JetBrains Rider installation?"; then
        show_rider_instructions
    fi
    
    # Setup Git LFS
    log_step "Configuring Git LFS"
    git lfs install
    log_success "Git LFS configured"
    
    # Create Unreal Engine workspace
    log_step "Creating Unreal Engine workspace"
    if confirm "Create Unreal Engine workspace directory?"; then
        create_ue_workspace
    fi
    
    # Download Unreal Engine
    log_step "Unreal Engine Download"
    if confirm "Show instructions for downloading Unreal Engine?"; then
        show_ue_download_instructions
    fi
    
    # Setup build scripts
    log_step "Creating build helper scripts"
    if confirm "Create Unreal Engine build helper scripts?"; then
        create_build_scripts
    fi
    
    # Setup shell aliases
    log_step "Setting up shell aliases"
    if confirm "Add Unreal Engine aliases to shell?"; then
        setup_ue_aliases
    fi
    
    # Performance tweaks
    log_step "System optimization"
    if confirm "Apply system optimizations for Unreal Engine?"; then
        apply_system_optimizations
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "gcc" "--version | head -n 1"
    show_tool_status "clang" "--version | head -n 1"
    show_tool_status "cmake" "--version | head -n 1"
    show_tool_status "vulkaninfo" "--summary 2>/dev/null | grep 'Vulkan Instance Version'"
    
    echo
    log_success "Unreal Engine 5 development environment is ready!"
    log_info "Next steps:"
    echo -e "1. Join Epic Games and link GitHub account: ${CYAN}https://www.unrealengine.com${RESET}"
    echo -e "2. Clone UE5 source: ${CYAN}git clone https://github.com/EpicGames/UnrealEngine.git${RESET}"
    echo -e "3. Run setup: ${CYAN}cd UnrealEngine && ./Setup.sh${RESET}"
    echo -e "4. Generate project files: ${CYAN}./GenerateProjectFiles.sh${RESET}"
    echo -e "5. Build: ${CYAN}make${RESET}"
}

check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check RAM
    total_ram_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    total_ram_gb=$((total_ram_kb / 1024 / 1024))
    
    if [[ $total_ram_gb -lt $MIN_RAM_GB ]]; then
        log_warning "System has ${total_ram_gb}GB RAM. Unreal Engine recommends at least ${MIN_RAM_GB}GB"
        if ! confirm "Continue anyway?"; then
            exit 1
        fi
    else
        log_success "RAM: ${total_ram_gb}GB ✓"
    fi
    
    # Check disk space
    available_space_gb=$(df -BG "$HOME" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [[ $available_space_gb -lt $MIN_DISK_GB ]]; then
        log_warning "Available disk space: ${available_space_gb}GB. Unreal Engine requires at least ${MIN_DISK_GB}GB"
        if ! confirm "Continue anyway?"; then
            exit 1
        fi
    else
        log_success "Disk space: ${available_space_gb}GB available ✓"
    fi
    
    # Check CPU
    cpu_cores=$(nproc)
    log_info "CPU cores: $cpu_cores"
    if [[ $cpu_cores -lt 4 ]]; then
        log_warning "System has $cpu_cores CPU cores. 8+ cores recommended for good build performance"
    fi
}

install_vulkan_sdk() {
    log_info "Installing Vulkan SDK..."
    
    # Add Vulkan repository
    wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
    sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-focal.list https://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list
    
    update_apt
    
    # Install Vulkan packages
    install_apt_package "vulkan-sdk" "Vulkan SDK"
    install_apt_package "vulkan-tools" "Vulkan Tools"
    install_apt_package "libvulkan-dev" "Vulkan Dev"
    install_apt_package "vulkan-validationlayers-dev" "Vulkan Validation Layers"
    
    log_success "Vulkan SDK installed"
}

create_ue_workspace() {
    log_info "Creating Unreal Engine workspace..."
    
    UE_WORKSPACE="$HOME/UnrealEngine"
    mkdir -p "$UE_WORKSPACE"/{Projects,Marketplace,Builds}
    
    # Create .gitignore for projects
    cat > "$UE_WORKSPACE/Projects/.gitignore" << 'EOF'
# Unreal Engine
Binaries/
Build/
DerivedDataCache/
Intermediate/
Saved/
.vscode/
.vs/
*.VC.db
*.opensdf
*.opendb
*.sdf
*.sln
*.suo
*.xcodeproj
*.xcworkspace

# Compiled source files
*.dll
*.exe
*.pdb
*.lib
*.a
*.so
*.dylib

# Content
Content/Collections/
Content/Developers/
Content/ExternalActors/
Content/ExternalObjects/
EOF
    
    log_success "Unreal Engine workspace created at $UE_WORKSPACE"
}

show_ue_download_instructions() {
    log_info "Unreal Engine Download Instructions:"
    echo
    echo "1. Create Epic Games account:"
    echo -e "   ${CYAN}https://www.unrealengine.com${RESET}"
    echo
    echo "2. Link your GitHub account:"
    echo -e "   ${CYAN}https://www.unrealengine.com/en-US/ue-on-github${RESET}"
    echo
    echo "3. Clone the repository (after GitHub access is granted):"
    echo -e "   ${CYAN}cd ~/UnrealEngine${RESET}"
    echo -e "   ${CYAN}git clone https://github.com/EpicGames/UnrealEngine.git Source${RESET}"
    echo
    echo "4. Checkout desired version:"
    echo -e "   ${CYAN}cd Source${RESET}"
    echo -e "   ${CYAN}git checkout 5.3${RESET}  # or desired version"
    echo
    echo "Note: The repository is large (~10GB) and may take time to clone"
}

show_rider_instructions() {
    log_info "JetBrains Rider Installation:"
    echo
    echo "Rider is a recommended IDE for Unreal Engine C++ development"
    echo
    echo "1. Download from:"
    echo -e "   ${CYAN}https://www.jetbrains.com/rider/${RESET}"
    echo
    echo "2. Install Unreal Engine support:"
    echo "   - Open Rider"
    echo "   - Go to Settings → Plugins"
    echo "   - Search for 'Unreal Engine' and install"
    echo
    echo "3. Install RiderLink plugin in Unreal:"
    echo "   - Open Unreal Engine"
    echo "   - Go to Edit → Plugins"
    echo "   - Search for 'RiderLink' and enable"
}

create_build_scripts() {
    log_info "Creating build helper scripts..."
    
    # Create build script
    cat > "$HOME/UnrealEngine/build-ue5.sh" << 'EOF'
#!/usr/bin/env bash
# Build Unreal Engine 5

set -e

if [[ ! -f "Setup.sh" ]]; then
    echo "Error: Must be run from Unreal Engine source directory"
    exit 1
fi

echo "=== Unreal Engine 5 Build Script ==="

# Run setup
echo "Running Setup.sh..."
./Setup.sh

# Generate project files
echo "Generating project files..."
./GenerateProjectFiles.sh

# Build
echo "Building Unreal Engine..."
make

echo "Build complete!"
echo "Run the editor with: ./Engine/Binaries/Linux/UnrealEditor"
EOF
    chmod +x "$HOME/UnrealEngine/build-ue5.sh"
    
    # Create project generator
    cat > "$HOME/UnrealEngine/new-project.sh" << 'EOF'
#!/usr/bin/env bash
# Create new Unreal Engine project

if [[ -z "$1" ]]; then
    echo "Usage: $0 <ProjectName>"
    exit 1
fi

PROJECT_NAME="$1"
PROJECT_DIR="$HOME/UnrealEngine/Projects/$PROJECT_NAME"

if [[ -d "$PROJECT_DIR" ]]; then
    echo "Project $PROJECT_NAME already exists!"
    exit 1
fi

echo "Creating project: $PROJECT_NAME"
mkdir -p "$PROJECT_DIR"

# Create project file
cat > "$PROJECT_DIR/$PROJECT_NAME.uproject" << PROJECT
{
    "FileVersion": 3,
    "EngineAssociation": "5.3",
    "Category": "",
    "Description": "",
    "Modules": [
        {
            "Name": "$PROJECT_NAME",
            "Type": "Runtime",
            "LoadingPhase": "Default"
        }
    ]
}
PROJECT

echo "Project created at: $PROJECT_DIR"
echo "Open with Unreal Editor to complete setup"
EOF
    chmod +x "$HOME/UnrealEngine/new-project.sh"
    
    log_success "Build scripts created"
}

apply_system_optimizations() {
    log_info "Applying system optimizations..."
    
    # Increase file watchers
    if confirm "Increase system file watcher limit?"; then
        echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.conf
        sudo sysctl -p
        log_success "File watcher limit increased"
    fi
    
    # Disable CPU frequency scaling
    if confirm "Disable CPU frequency scaling for consistent performance?"; then
        sudo apt-get install -y cpufrequtils
        echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
        sudo systemctl restart cpufrequtils
        log_success "CPU governor set to performance"
    fi
    
    # Setup swap if needed
    swap_size=$(free -g | grep Swap | awk '{print $2}')
    if [[ $swap_size -lt 8 ]]; then
        if confirm "Current swap is ${swap_size}GB. Create 16GB swapfile?"; then
            sudo swapoff -a
            sudo dd if=/dev/zero of=/swapfile bs=1G count=16
            sudo chmod 600 /swapfile
            sudo mkswap /swapfile
            sudo swapon /swapfile
            echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
            log_success "16GB swap created"
        fi
    fi
}

setup_ue_aliases() {
    log_info "Setting up Unreal Engine aliases..."
    
    ue_aliases='
# Unreal Engine aliases
export UE_ROOT="$HOME/UnrealEngine/Source"
export UE_PROJECTS="$HOME/UnrealEngine/Projects"

alias ue-editor="$UE_ROOT/Engine/Binaries/Linux/UnrealEditor"
alias ue-build="cd $UE_ROOT && ./build-ue5.sh"
alias ue-generate="cd $UE_ROOT && ./GenerateProjectFiles.sh"
alias ue-clean="cd $UE_ROOT && make clean"

# Navigation
alias cdue="cd $UE_ROOT"
alias cdprojects="cd $UE_PROJECTS"

# Project management
ue-new-project() {
    if [[ -z "$1" ]]; then
        echo "Usage: ue-new-project <ProjectName>"
        return 1
    fi
    $HOME/UnrealEngine/new-project.sh "$1"
}

ue-open() {
    if [[ -z "$1" ]]; then
        echo "Usage: ue-open <ProjectName>"
        return 1
    fi
    $UE_ROOT/Engine/Binaries/Linux/UnrealEditor "$UE_PROJECTS/$1/$1.uproject"
}

# Build helpers
ue-build-debug() {
    cd $UE_ROOT
    make UnrealEditor-Linux-Debug
}

ue-build-shipping() {
    cd $UE_ROOT
    make UnrealEditor-Linux-Shipping
}

# Shader compilation
ue-compile-shaders() {
    $UE_ROOT/Engine/Binaries/Linux/UnrealEditor -run=DerivedDataCache -fill
}'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$ue_aliases" "Unreal Engine aliases"
    fi
    
    if [[ -f "$HOME/.zshrc" ]]; then
        add_to_file_if_missing "$HOME/.zshrc" "$ue_aliases" "Unreal Engine aliases"
    fi
    
    log_success "Unreal Engine aliases added to shell"
}

# Run main function
main "$@" 