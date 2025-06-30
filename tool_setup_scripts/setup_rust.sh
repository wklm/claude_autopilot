#!/usr/bin/env bash

# Setup script for Rust development environment
# Installs: Rust toolchain, cargo tools, and development utilities

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

main() {
    show_banner "Rust Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "rustc" "--version"
    show_tool_status "cargo" "--version"
    show_tool_status "rustup" "--version"
    show_tool_status "cargo-watch" "--version"
    show_tool_status "cargo-edit" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "pkg-config" "pkg-config"
    install_apt_package "libssl-dev" "OpenSSL Development Libraries"
    install_apt_package "libsqlite3-dev" "SQLite Development Libraries"
    install_apt_package "libpq-dev" "PostgreSQL Development Libraries"
    
    # Install Rust
    log_step "Installing Rust"
    if command_exists rustc; then
        log_info "Rust is already installed"
        current_rust=$(rustc --version | awk '{print $2}')
        log_info "Current Rust version: $current_rust"
        
        if confirm "Update Rust to latest stable version?"; then
            rustup update stable
            log_success "Rust updated to $(rustc --version | awk '{print $2}')"
        fi
    else
        if confirm "Install Rust via rustup?"; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            
            # Source cargo env
            source "$HOME/.cargo/env"
            
            log_success "Rust installed successfully"
            log_info "Cargo environment loaded for current session"
        else
            log_error "Rust is required to continue"
            exit 1
        fi
    fi
    
    # Ensure cargo is in PATH
    if [[ -f "$HOME/.cargo/env" ]]; then
        source "$HOME/.cargo/env"
    fi
    
    # Install additional Rust targets
    log_step "Installing additional Rust targets"
    if confirm "Install wasm32 target for WebAssembly development?"; then
        rustup target add wasm32-unknown-unknown
        log_success "wasm32-unknown-unknown target installed"
    fi
    
    if confirm "Install musl target for static Linux binaries?"; then
        rustup target add x86_64-unknown-linux-musl
        install_apt_package "musl-tools" "musl tools"
        log_success "x86_64-unknown-linux-musl target installed"
    fi
    
    # Install essential cargo tools
    log_step "Installing Cargo development tools"
    
    # cargo-watch for auto-recompilation
    install_cargo_tool "cargo-watch" "Auto-recompilation on file changes"
    
    # cargo-edit for managing dependencies
    install_cargo_tool "cargo-edit" "Add/remove/upgrade dependencies from CLI"
    
    # cargo-audit for security audits
    install_cargo_tool "cargo-audit" "Security audit for dependencies" "--features=fix"
    
    # cargo-outdated for checking outdated dependencies
    install_cargo_tool "cargo-outdated" "Check for outdated dependencies"
    
    # cargo-expand for macro expansion
    install_cargo_tool "cargo-expand" "Expand macros for debugging"
    
    # cargo-deny for supply chain security
    install_cargo_tool "cargo-deny" "Lint dependencies for security/license issues"
    
    # cargo-nextest for better test runner
    install_cargo_tool "cargo-nextest" "Next-generation test runner" "--locked"
    
    # cargo-llvm-cov for code coverage
    install_cargo_tool "cargo-llvm-cov" "Code coverage with LLVM"
    
    # Web development specific tools
    log_step "Installing web development tools"
    if confirm "Install tools for Rust web development?"; then
        # sqlx-cli for database migrations
        install_cargo_tool "sqlx-cli" "SQLx database migration tool" "--no-default-features --features rustls,postgres,mysql,sqlite"
        
        # sea-orm-cli for SeaORM
        install_cargo_tool "sea-orm-cli" "SeaORM database ORM CLI" "--features runtime-tokio-rustls,codegen"
        
        # cargo-leptos for Leptos framework
        install_cargo_tool "cargo-leptos" "Leptos web framework CLI"
        
        # trunk for WASM frontend development
        install_cargo_tool "trunk" "WASM web application bundler"
        
        # wasm-bindgen-cli
        install_cargo_tool "wasm-bindgen-cli" "WASM bindgen CLI"
    fi
    
    # System programming specific tools
    log_step "Installing system programming tools"
    if confirm "Install tools for Rust system programming?"; then
        # cargo-binutils for low-level utilities
        rustup component add llvm-tools-preview
        install_cargo_tool "cargo-binutils" "Cargo binutils for low-level work"
        
        # cargo-asm for assembly output
        install_cargo_tool "cargo-asm" "Show assembly output of Rust code"
        
        # flamegraph for performance profiling
        install_apt_package "linux-tools-common" "Linux perf tools"
        install_apt_package "linux-tools-generic" "Linux perf tools (generic)"
        install_cargo_tool "flamegraph" "Performance profiling with flamegraphs"
        
        # cargo-miri for undefined behavior detection
        if confirm "Install Miri (undefined behavior detector - requires nightly)?"; then
            rustup +nightly component add miri
            log_success "Miri installed for nightly toolchain"
        fi
    fi
    
    # Install rust-analyzer
    log_step "Installing rust-analyzer (LSP server)"
    if ! command_exists rust-analyzer; then
        if confirm "Install rust-analyzer for IDE support?"; then
            curl -L https://github.com/rust-lang/rust-analyzer/releases/latest/download/rust-analyzer-x86_64-unknown-linux-gnu.gz | gunzip -c - > ~/.local/bin/rust-analyzer
            chmod +x ~/.local/bin/rust-analyzer
            log_success "rust-analyzer installed"
        fi
    else
        log_info "rust-analyzer is already installed"
    fi
    
    # Install sccache for compilation caching
    if ! command_exists sccache; then
        if confirm "Install sccache (shared compilation cache)?"; then
            cargo install sccache --locked
            
            # Configure cargo to use sccache
            mkdir -p ~/.cargo
            echo '[build]' >> ~/.cargo/config.toml
            echo 'rustc-wrapper = "sccache"' >> ~/.cargo/config.toml
            
            log_success "sccache installed and configured"
        fi
    fi
    
    # Setup global gitignore for Rust
    log_step "Configuring Git for Rust development"
    if confirm "Add Rust patterns to global .gitignore?"; then
        gitignore_file="$HOME/.gitignore_global"
        touch "$gitignore_file"
        
        # Rust patterns
        rust_patterns=(
            "# Rust"
            "target/"
            "Cargo.lock"
            "**/*.rs.bk"
            "*.pdb"
            ""
            "# Cargo"
            ".cargo/registry"
            ".cargo/git"
            ""
            "# IDEs"
            ".idea/"
            ".vscode/"
            "*.swp"
            "*.swo"
            "*~"
            ""
            "# OS"
            ".DS_Store"
            "Thumbs.db"
            ""
            "# Test data"
            "tarpaulin-report.html"
            "cobertura.xml"
            ""
            "# Environment"
            ".env"
            ".env.local"
        )
        
        for pattern in "${rust_patterns[@]}"; do
            if [[ -n "$pattern" ]] && ! grep -Fxq "$pattern" "$gitignore_file" 2>/dev/null; then
                echo "$pattern" >> "$gitignore_file"
            elif [[ -z "$pattern" ]]; then
                echo "" >> "$gitignore_file"
            fi
        done
        
        git config --global core.excludesfile "$gitignore_file"
        log_success "Global .gitignore configured for Rust"
    fi
    
    # Create sample configs
    log_step "Creating sample configurations"
    if confirm "Create sample .cargo/config.toml in current directory?"; then
        create_cargo_config
    fi
    
    if confirm "Create sample rustfmt.toml in current directory?"; then
        create_rustfmt_config
    fi
    
    if confirm "Create sample .clippy.toml in current directory?"; then
        create_clippy_config
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "rustc" "--version"
    show_tool_status "cargo" "--version"
    show_tool_status "rustup" "--version"
    show_tool_status "cargo-watch" "--version"
    show_tool_status "cargo-edit" "--version"
    show_tool_status "cargo-audit" "--version"
    show_tool_status "cargo-nextest" "--version"
    show_tool_status "rust-analyzer" "--version"
    
    echo
    log_success "Rust development environment is ready!"
    log_info "To create a new Rust project:"
    echo -e "  ${CYAN}cargo new my-project${RESET}             # Binary project"
    echo -e "  ${CYAN}cargo new my-lib --lib${RESET}           # Library project"
    echo -e "  ${CYAN}cargo new my-workspace --bin${RESET}     # Workspace project"
    echo
    log_info "For web development with Axum:"
    echo -e "  ${CYAN}cargo add axum tokio sqlx${RESET}"
    echo
    log_info "Run 'cargo --list' to see all available commands"
}

install_cargo_tool() {
    local tool="$1"
    local description="$2"
    local extra_args="${3:-}"
    
    if command_exists "$tool"; then
        log_info "$tool is already installed"
        if confirm "Update $tool to latest version?"; then
            cargo install "$tool" --locked --force $extra_args
            log_success "$tool updated"
        fi
    else
        if confirm "Install $tool ($description)?"; then
            cargo install "$tool" --locked $extra_args
            log_success "$tool installed"
        fi
    fi
}

create_cargo_config() {
    mkdir -p .cargo
    cat > .cargo/config.toml << 'EOF'
# Cargo configuration for Rust projects

[build]
# Number of parallel jobs, defaults to # of CPUs
jobs = 8

# Incremental compilation
incremental = true

[target.x86_64-unknown-linux-gnu]
# Use lld linker for faster linking (install: sudo apt install lld)
# linker = "clang"
# rustflags = ["-C", "link-arg=-fuse-ld=lld"]

[profile.dev]
# Optimize dependencies even in debug mode
opt-level = 0
debug = true

[profile.dev.package."*"]
opt-level = 3

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
strip = true

[profile.release-with-debug]
inherits = "release"
strip = false
debug = true

# Custom profile for profiling
[profile.profiling]
inherits = "release"
debug = true
strip = false

[net]
retry = 3
EOF
    
    log_success "Created .cargo/config.toml"
}

create_rustfmt_config() {
    cat > rustfmt.toml << 'EOF'
# Rust formatting configuration

edition = "2021"
max_width = 100
hard_tabs = false
tab_spaces = 4
newline_style = "Auto"
use_small_heuristics = "Default"
reorder_imports = true
reorder_modules = true
remove_nested_parens = true
match_arm_leading_pipes = "Never"
fn_params_layout = "Tall"
edition = "2021"
merge_derives = true
use_try_shorthand = true
use_field_init_shorthand = true
force_explicit_abi = true
imports_granularity = "Crate"
group_imports = "StdExternalCrate"
EOF
    
    log_success "Created rustfmt.toml"
}

create_clippy_config() {
    cat > .clippy.toml << 'EOF'
# Clippy configuration

# Maximum cognitive complexity allowed
cognitive-complexity-threshold = 20

# Maximum number of lines in a function
too-many-lines-threshold = 100

# Maximum number of arguments in a function
too-many-arguments-threshold = 7

# Avoid breaking changes
avoid-breaking-exported-api = true

# Disallowed names
disallowed-names = ["foo", "bar", "baz"]
EOF
    
    log_success "Created .clippy.toml"
}

# Run main function
main "$@" 