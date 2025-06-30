#!/usr/bin/env bash

# Setup script for C++ systems programming environment
# Installs: GCC, Clang, CMake, debugging tools, and development libraries

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions and requirements
GCC_VERSION="13"
CLANG_VERSION="17"
CMAKE_MIN_VERSION="3.25"

main() {
    show_banner "C++ Systems Programming Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "gcc" "--version"
    show_tool_status "g++" "--version"
    show_tool_status "clang" "--version"
    show_tool_status "cmake" "--version"
    show_tool_status "gdb" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install essential build tools
    log_step "Installing essential build tools"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "git" "Git"
    install_apt_package "wget" "wget"
    install_apt_package "curl" "cURL"
    
    # Install GCC and G++
    log_step "Installing GCC $GCC_VERSION"
    if ! dpkg -l | grep -q "gcc-$GCC_VERSION"; then
        if confirm "Install GCC $GCC_VERSION?"; then
            # Add Ubuntu toolchain PPA for newer GCC versions
            sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
            sudo apt-get update
            install_apt_package "gcc-$GCC_VERSION" "GCC $GCC_VERSION"
            install_apt_package "g++-$GCC_VERSION" "G++ $GCC_VERSION"
            
            # Set as default
            if confirm "Set GCC $GCC_VERSION as default?"; then
                sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$GCC_VERSION 100
                sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$GCC_VERSION 100
                sudo update-alternatives --config gcc
                sudo update-alternatives --config g++
            fi
        fi
    else
        log_info "GCC $GCC_VERSION is already installed"
    fi
    
    # Install Clang
    log_step "Installing Clang $CLANG_VERSION"
    if ! command_exists clang-$CLANG_VERSION; then
        if confirm "Install Clang $CLANG_VERSION?"; then
            # Add LLVM repository
            wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
            sudo add-apt-repository -y "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-$CLANG_VERSION main"
            sudo apt-get update
            
            install_apt_package "clang-$CLANG_VERSION" "Clang $CLANG_VERSION"
            install_apt_package "clang-tools-$CLANG_VERSION" "Clang Tools"
            install_apt_package "clang-format-$CLANG_VERSION" "Clang Format"
            install_apt_package "clang-tidy-$CLANG_VERSION" "Clang Tidy"
            install_apt_package "lld-$CLANG_VERSION" "LLD Linker"
            install_apt_package "lldb-$CLANG_VERSION" "LLDB Debugger"
            
            # Create symlinks for easier access
            if confirm "Create symlinks for Clang tools?"; then
                sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-$CLANG_VERSION 100
                sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-$CLANG_VERSION 100
                sudo update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-$CLANG_VERSION 100
                sudo update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-$CLANG_VERSION 100
            fi
        fi
    else
        log_info "Clang $CLANG_VERSION is already installed"
    fi
    
    # Install CMake
    log_step "Installing CMake"
    if command_exists cmake; then
        current_cmake=$(cmake --version | head -n1 | awk '{print $3}')
        log_info "Current CMake version: $current_cmake"
        
        if [[ "$(printf '%s\n' "$CMAKE_MIN_VERSION" "$current_cmake" | sort -V | head -n1)" != "$CMAKE_MIN_VERSION" ]]; then
            log_warning "CMake version is older than recommended $CMAKE_MIN_VERSION"
            if confirm "Install latest CMake from Kitware APT repository?"; then
                # Add Kitware APT repository
                wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
                echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
                sudo apt-get update
                
                # Remove old CMake if present
                sudo apt-get remove --purge cmake -y
                install_apt_package "cmake" "CMake"
            fi
        else
            log_success "CMake version is sufficient"
        fi
    else
        install_apt_package "cmake" "CMake"
    fi
    
    # Install debugging and profiling tools
    log_step "Installing debugging and profiling tools"
    install_apt_package "gdb" "GDB Debugger"
    install_apt_package "valgrind" "Valgrind"
    install_apt_package "strace" "strace"
    install_apt_package "ltrace" "ltrace"
    install_apt_package "perf-tools-unstable" "Perf Tools" || install_apt_package "linux-tools-generic" "Perf Tools"
    
    # Install development libraries
    log_step "Installing common development libraries"
    
    # System libraries
    install_apt_package "libc6-dev" "C Library Development"
    install_apt_package "libstdc++-$GCC_VERSION-dev" "C++ Standard Library"
    
    # Threading and concurrency
    install_apt_package "libtbb-dev" "Intel TBB"
    install_apt_package "libboost-all-dev" "Boost Libraries"
    
    # Networking
    install_apt_package "libssl-dev" "OpenSSL Development"
    install_apt_package "libcurl4-openssl-dev" "cURL Development"
    
    # Testing frameworks
    install_apt_package "libgtest-dev" "Google Test"
    install_apt_package "libbenchmark-dev" "Google Benchmark"
    
    # Documentation
    install_apt_package "doxygen" "Doxygen"
    install_apt_package "graphviz" "Graphviz"
    
    # Install build systems
    log_step "Installing additional build systems"
    install_apt_package "ninja-build" "Ninja Build"
    install_apt_package "meson" "Meson Build"
    
    # Install vcpkg package manager
    log_step "Installing vcpkg (C++ package manager)"
    VCPKG_ROOT="$HOME/vcpkg"
    if [[ ! -d "$VCPKG_ROOT" ]]; then
        if confirm "Install vcpkg package manager?"; then
            git clone https://github.com/Microsoft/vcpkg.git "$VCPKG_ROOT"
            cd "$VCPKG_ROOT"
            ./bootstrap-vcpkg.sh
            
            # Add vcpkg to PATH
            vcpkg_path_line="export PATH=\"$VCPKG_ROOT:\$PATH\""
            vcpkg_root_line="export VCPKG_ROOT=\"$VCPKG_ROOT\""
            
            if [[ -f "$HOME/.bashrc" ]]; then
                add_to_file_if_missing "$HOME/.bashrc" "$vcpkg_path_line" "vcpkg PATH"
                add_to_file_if_missing "$HOME/.bashrc" "$vcpkg_root_line" "VCPKG_ROOT"
            fi
            
            if [[ -f "$HOME/.zshrc" ]]; then
                add_to_file_if_missing "$HOME/.zshrc" "$vcpkg_path_line" "vcpkg PATH"
                add_to_file_if_missing "$HOME/.zshrc" "$vcpkg_root_line" "VCPKG_ROOT"
            fi
            
            export PATH="$VCPKG_ROOT:$PATH"
            export VCPKG_ROOT="$VCPKG_ROOT"
            
            log_success "vcpkg installed"
        fi
    else
        log_info "vcpkg is already installed at $VCPKG_ROOT"
    fi
    
    # Create C++ project template
    log_step "Setting up C++ workspace"
    CPP_WORKSPACE="$HOME/cpp-projects"
    if confirm "Create C++ workspace directory at $CPP_WORKSPACE?"; then
        create_directory "$CPP_WORKSPACE" "C++ workspace"
        
        # Create a sample CMake project
        if confirm "Create a sample CMake project template?"; then
            SAMPLE_PROJECT="$CPP_WORKSPACE/cpp-template"
            create_directory "$SAMPLE_PROJECT" "Sample project"
            
            # Create CMakeLists.txt
            cat << 'EOF' > "$SAMPLE_PROJECT/CMakeLists.txt"
cmake_minimum_required(VERSION 3.25)
project(cpp_template VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add executable
add_executable(main src/main.cpp)

# Set compile options
target_compile_options(main PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Debug>:-g -O0 -fsanitize=address,undefined>
    $<$<CONFIG:Release>:-O3 -march=native>
)

# Link options for sanitizers in debug mode
target_link_options(main PRIVATE
    $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)

# Find and link threads
find_package(Threads REQUIRED)
target_link_libraries(main PRIVATE Threads::Threads)
EOF
            
            # Create source directory and main.cpp
            create_directory "$SAMPLE_PROJECT/src" "Source directory"
            cat << 'EOF' > "$SAMPLE_PROJECT/src/main.cpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>

int main() {
    std::cout << "Hello, Modern C++!" << std::endl;
    
    // Example of modern C++ features
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    
    // Range-based for loop
    std::cout << "Numbers: ";
    for (const auto& n : numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    // Lambda expression
    auto sum = std::accumulate(numbers.begin(), numbers.end(), 0,
        [](int a, int b) { return a + b; });
    std::cout << "Sum: " << sum << std::endl;
    
    // Threading example
    std::thread t([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "Hello from thread!" << std::endl;
    });
    
    t.join();
    
    return 0;
}
EOF
            
            # Create .clang-format
            cat << 'EOF' > "$SAMPLE_PROJECT/.clang-format"
BasedOnStyle: LLVM
IndentWidth: 4
ColumnLimit: 100
AllowShortFunctionsOnASingleLine: None
AlwaysBreakTemplateDeclarations: Yes
EOF
            
            log_success "Sample C++ project created at $SAMPLE_PROJECT"
        fi
    fi
    
    # VS Code extensions
    if command_exists code; then
        log_step "VS Code C++ extensions"
        if confirm "Install C++ VS Code extensions?"; then
            code --install-extension ms-vscode.cpptools
            code --install-extension ms-vscode.cmake-tools
            code --install-extension llvm-vs-code-extensions.vscode-clangd
            log_success "VS Code extensions installed"
        fi
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "gcc" "--version"
    show_tool_status "g++" "--version"
    show_tool_status "clang" "--version"
    show_tool_status "cmake" "--version"
    show_tool_status "gdb" "--version"
    show_tool_status "valgrind" "--version"
    if [[ -n "${VCPKG_ROOT:-}" ]]; then
        show_tool_status "vcpkg" "version"
    fi
    
    echo
    log_success "C++ systems programming environment is ready!"
    log_info "To build the sample project:"
    echo -e "  ${CYAN}cd $CPP_WORKSPACE/cpp-template${RESET}"
    echo -e "  ${CYAN}cmake -B build -DCMAKE_BUILD_TYPE=Debug${RESET}"
    echo -e "  ${CYAN}cmake --build build${RESET}"
    echo -e "  ${CYAN}./build/main${RESET}"
    
    echo
    log_info "For Release builds: ${CYAN}cmake -B build -DCMAKE_BUILD_TYPE=Release${RESET}"
}

# Run main function
main "$@" 