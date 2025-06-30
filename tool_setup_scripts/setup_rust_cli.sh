#!/usr/bin/env bash

# Setup script for Rust CLI Tools development environment
# Installs: Rust toolchain, cargo extensions, CLI frameworks, testing tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
RUST_VERSION="stable"

main() {
    show_banner "Rust CLI Tools Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "rustc" "--version"
    show_tool_status "cargo" "--version"
    show_tool_status "rustup" "--version"
    show_tool_status "sccache" "--version"
    show_tool_status "cargo-watch" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "pkg-config" "pkg-config"
    install_apt_package "libssl-dev" "OpenSSL Dev"
    install_apt_package "cmake" "CMake"
    
    # Install Rust via rustup
    log_step "Installing Rust"
    if ! command_exists rustup; then
        if confirm "Install Rust via rustup?"; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source "$HOME/.cargo/env"
            log_success "Rust installed"
        else
            log_error "Rust is required to continue"
            exit 1
        fi
    else
        log_info "Rust is already installed"
        if confirm "Update Rust to latest stable?"; then
            rustup update stable
            log_success "Rust updated"
        fi
    fi
    
    # Ensure Rust is in PATH
    source "$HOME/.cargo/env" 2>/dev/null || true
    
    # Install additional Rust toolchains
    log_step "Installing additional Rust components"
    if confirm "Install Rust nightly toolchain (for experimental features)?"; then
        rustup toolchain install nightly
        log_success "Nightly toolchain installed"
    fi
    
    # Install Rust components
    if confirm "Install essential Rust components (rust-src, rustfmt, clippy)?"; then
        rustup component add rust-src rustfmt clippy
        rustup component add rust-analyzer
        log_success "Rust components installed"
    fi
    
    # Install cargo extensions for CLI development
    log_step "Installing Cargo extensions for CLI development"
    
    # Core CLI development tools
    if confirm "Install core CLI development tools?"; then
        cargo install cargo-watch  # Auto-rebuild on file changes
        cargo install cargo-edit   # Add/remove dependencies from CLI
        cargo install cargo-outdated  # Check for outdated dependencies
        cargo install cargo-audit  # Security vulnerability audit
        cargo install cargo-release  # Release automation
        log_success "Core CLI tools installed"
    fi
    
    # CLI frameworks and libraries
    log_step "Installing CLI frameworks"
    if confirm "Install popular CLI framework crates?"; then
        cargo install clap --example  # Command Line Argument Parser examples
        cargo install structopt --example  # Derive macro for clap
        log_info "Note: These are examples. Add clap/structopt to your Cargo.toml for actual use"
    fi
    
    # Build and packaging tools
    if confirm "Install build and packaging tools?"; then
        cargo install cargo-deb  # Create Debian packages
        cargo install cargo-rpm  # Create RPM packages
        cargo install cargo-generate  # Project templates
        cargo install cross  # Cross-compilation
        cargo install cargo-make  # Task runner
        log_success "Build tools installed"
    fi
    
    # Performance and optimization tools
    log_step "Installing performance tools"
    if confirm "Install performance analysis tools?"; then
        cargo install flamegraph  # Performance profiling
        cargo install cargo-profiler  # Code profiling
        cargo install hyperfine  # Command-line benchmarking
        cargo install cargo-bloat  # Binary size analysis
        log_success "Performance tools installed"
    fi
    
    # Testing and quality tools
    if confirm "Install testing and quality tools?"; then
        cargo install cargo-nextest  # Next-generation test runner
        cargo install cargo-tarpaulin  # Code coverage
        cargo install cargo-mutants  # Mutation testing
        log_success "Testing tools installed"
    fi
    
    # Documentation tools
    if confirm "Install documentation tools?"; then
        cargo install mdbook  # Create books from Markdown
        cargo install cargo-readme  # Generate README from doc comments
        log_success "Documentation tools installed"
    fi
    
    # Install sccache for faster builds
    log_step "Installing sccache"
    if ! command_exists sccache; then
        if confirm "Install sccache (compiler cache for faster builds)?"; then
            cargo install sccache
            export RUSTC_WRAPPER=sccache
            log_success "sccache installed"
        fi
    else
        log_info "sccache is already installed"
    fi
    
    # Create CLI project templates
    log_step "Creating CLI project templates"
    if confirm "Create Rust CLI project templates?"; then
        create_cli_templates
    fi
    
    # Setup shell integration
    log_step "Setting up shell integration"
    if confirm "Add Rust CLI development aliases to shell?"; then
        setup_rust_cli_aliases
    fi
    
    # Install VS Code extensions
    log_step "VS Code Extensions"
    if command_exists code; then
        if confirm "Install recommended VS Code extensions for Rust?"; then
            code --install-extension rust-lang.rust-analyzer
            code --install-extension tamasfe.even-better-toml
            code --install-extension serayuzgur.crates
            code --install-extension vadimcn.vscode-lldb
            log_success "VS Code extensions installed"
        fi
    fi
    
    # Create example CLI project
    log_step "Example CLI project"
    if confirm "Create an example Rust CLI project?"; then
        create_example_cli_project
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "rustc" "--version"
    show_tool_status "cargo" "--version"
    show_tool_status "rustup" "--version"
    show_tool_status "cargo-watch" "--version"
    show_tool_status "cargo-edit" "--version"
    show_tool_status "sccache" "--version"
    
    echo
    log_success "Rust CLI Tools development environment is ready!"
    log_info "Useful commands:"
    echo -e "  ${CYAN}cargo new my-cli --bin${RESET} - Create new CLI project"
    echo -e "  ${CYAN}cargo watch -x run${RESET} - Auto-rebuild and run on changes"
    echo -e "  ${CYAN}cargo add clap --features derive${RESET} - Add clap with derive feature"
    echo -e "  ${CYAN}cargo clippy${RESET} - Run linter"
    echo -e "  ${CYAN}cargo fmt${RESET} - Format code"
    echo -e "  ${CYAN}rust-cli-new <name>${RESET} - Create new CLI project from template"
}

create_cli_templates() {
    log_info "Creating Rust CLI project templates..."
    
    # Create templates directory
    mkdir -p "$HOME/.cargo/cli-templates"
    
    # Basic CLI template
    cat > "$HOME/.cargo/cli-templates/basic-cli.rs" << 'EOF'
use clap::Parser;
use anyhow::Result;

/// A simple CLI application
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    name: String,

    /// Number of times to greet
    #[arg(short, long, default_value_t = 1)]
    count: u8,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.verbose {
        println!("Running with verbose output...");
    }

    for _ in 0..args.count {
        println!("Hello, {}!", args.name);
    }

    Ok(())
}
EOF
    
    # Advanced CLI template with subcommands
    cat > "$HOME/.cargo/cli-templates/advanced-cli.rs" << 'EOF'
use clap::{Parser, Subcommand};
use anyhow::{Context, Result};
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Optional config file
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Enable verbose output
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process a file
    Process {
        /// The file to process
        file: PathBuf,
        
        /// Output format
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    /// List available items
    List {
        /// Filter pattern
        #[arg(short, long)]
        filter: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging based on verbosity
    let log_level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level)).init();

    match &cli.command {
        Commands::Process { file, format } => {
            println!("Processing file: {}", file.display());
            println!("Output format: {}", format);
            // Add processing logic here
        }
        Commands::List { filter } => {
            println!("Listing items...");
            if let Some(pattern) = filter {
                println!("Filter: {}", pattern);
            }
            // Add listing logic here
        }
    }

    Ok(())
}
EOF
    
    # Create Cargo.toml template
    cat > "$HOME/.cargo/cli-templates/Cargo.toml" << 'EOF'
[package]
name = "my-cli"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <you@example.com>"]
description = "A brief description of your CLI"
license = "MIT OR Apache-2.0"
repository = "https://github.com/yourusername/my-cli"
keywords = ["cli", "tool"]
categories = ["command-line-utilities"]

[dependencies]
clap = { version = "4", features = ["derive", "env"] }
anyhow = "1"
env_logger = "0.10"
log = "0.4"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.11", features = ["json"] }
indicatif = "0.17"
colored = "2"
directories = "5"
config = "0.13"

[dev-dependencies]
assert_cmd = "2"
predicates = "3"
tempfile = "3"

[profile.release]
strip = true
opt-level = "z"
lto = true
codegen-units = 1
EOF
    
    log_success "CLI templates created in ~/.cargo/cli-templates/"
}

create_example_cli_project() {
    log_info "Creating example Rust CLI project..."
    
    project_name="rust-cli-example"
    project_dir="$HOME/$project_name"
    
    if [[ -d "$project_dir" ]]; then
        log_warning "Example project already exists at $project_dir"
        return
    fi
    
    # Create project
    cd "$HOME"
    cargo new "$project_name" --bin
    cd "$project_name"
    
    # Add dependencies
    cargo add clap --features derive
    cargo add anyhow
    cargo add env_logger
    cargo add colored
    cargo add indicatif
    
    # Add dev dependencies
    cargo add --dev assert_cmd predicates
    
    # Create example CLI code
    cat > src/main.rs << 'EOF'
use clap::Parser;
use anyhow::Result;
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::{thread, time::Duration};

/// A demonstration CLI tool built with Rust
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Text to display
    #[arg(short, long, default_value = "Hello, Rust CLI!")]
    message: String,

    /// Show progress bar demonstration
    #[arg(short, long)]
    progress: bool,

    /// Use colored output
    #[arg(short, long)]
    color: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    if args.color {
        println!("{}", args.message.green().bold());
        println!("{}", "This is red!".red());
        println!("{}", "This is blue!".blue());
        println!("{}", "This is yellow!".yellow());
    } else {
        println!("{}", args.message);
    }

    if args.progress {
        demonstrate_progress()?;
    }

    println!("\n{}", "CLI demonstration complete!".cyan());
    Ok(())
}

fn demonstrate_progress() -> Result<()> {
    println!("\nDemonstrating progress bar:");
    
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?
            .progress_chars("#>-")
    );

    for i in 0..100 {
        pb.set_position(i);
        thread::sleep(Duration::from_millis(20));
    }
    pb.finish_with_message("Done!");
    
    Ok(())
}
EOF
    
    # Create tests
    mkdir -p tests
    cat > tests/cli_test.rs << 'EOF'
use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_default_message() {
    let mut cmd = Command::cargo_bin("rust-cli-example").unwrap();
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Hello, Rust CLI!"));
}

#[test]
fn test_custom_message() {
    let mut cmd = Command::cargo_bin("rust-cli-example").unwrap();
    cmd.arg("--message").arg("Custom message")
        .assert()
        .success()
        .stdout(predicate::str::contains("Custom message"));
}

#[test]
fn test_help() {
    let mut cmd = Command::cargo_bin("rust-cli-example").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("A demonstration CLI tool"));
}
EOF
    
    # Create README
    cat > README.md << 'EOF'
# Rust CLI Example

A demonstration CLI tool built with Rust, showcasing common patterns and best practices.

## Features

- Command-line argument parsing with clap
- Colored output
- Progress bars
- Error handling with anyhow
- Comprehensive tests

## Usage

```bash
# Run with default message
cargo run

# Custom message
cargo run -- --message "Hello, World!"

# With colors
cargo run -- --color

# Show progress bar
cargo run -- --progress

# All options
cargo run -- --message "Building..." --color --progress
```

## Development

```bash
# Watch for changes and auto-rebuild
cargo watch -x run

# Run tests
cargo test

# Run clippy
cargo clippy

# Format code
cargo fmt
```
EOF
    
    log_success "Example CLI project created at $project_dir"
    log_info "To try it out:"
    echo -e "  ${CYAN}cd $project_dir${RESET}"
    echo -e "  ${CYAN}cargo run -- --help${RESET}"
    echo -e "  ${CYAN}cargo run -- --color --progress${RESET}"
    
    cd "$SCRIPT_DIR"
}

setup_rust_cli_aliases() {
    log_info "Setting up Rust CLI development aliases..."
    
    rust_cli_aliases='
# Rust CLI development aliases
alias cargo-w="cargo watch -x run"
alias cargo-wt="cargo watch -x test"
alias cargo-wc="cargo watch -x check"
alias cargo-wr="cargo watch -x "run --release""
alias cargo-fix="cargo fix --allow-dirty --allow-staged"
alias cargo-udeps="cargo +nightly udeps"

# Quick commands
alias clippy-all="cargo clippy -- -W clippy::all"
alias clippy-fix="cargo clippy --fix -- -W clippy::all"
alias fmt-check="cargo fmt -- --check"
alias audit-fix="cargo audit fix"

# Build commands
alias build-release="cargo build --release"
alias build-static="RUSTFLAGS="-C target-feature=+crt-static" cargo build --release --target x86_64-unknown-linux-gnu"

# Testing shortcuts
alias test-all="cargo test --all-features"
alias test-doc="cargo test --doc"
alias bench="cargo bench"
alias nextest="cargo nextest run"

# Project creation function
rust-cli-new() {
    if [[ -z "$1" ]]; then
        echo "Usage: rust-cli-new <project-name>"
        return 1
    fi
    
    cargo new "$1" --bin
    cd "$1"
    
    # Add common CLI dependencies
    cargo add clap --features derive,env
    cargo add anyhow
    cargo add env_logger log
    cargo add serde --features derive
    cargo add serde_json
    
    # Add dev dependencies
    cargo add --dev assert_cmd predicates
    
    # Copy template if available
    if [[ -f "$HOME/.cargo/cli-templates/basic-cli.rs" ]]; then
        cp "$HOME/.cargo/cli-templates/basic-cli.rs" src/main.rs
    fi
    
    echo "Rust CLI project $1 created with common dependencies!"
}

# Cargo extension installer
cargo-install-cli-tools() {
    echo "Installing/updating Rust CLI development tools..."
    cargo install cargo-watch cargo-edit cargo-outdated cargo-audit
    cargo install cargo-release cargo-deb cargo-rpm cargo-generate
    cargo install flamegraph hyperfine cargo-bloat
    cargo install cargo-nextest cargo-tarpaulin
    cargo install mdbook cargo-readme
}

# Cross-compilation helper
rust-cross-build() {
    if [[ -z "$1" ]]; then
        echo "Usage: rust-cross-build <target>"
        echo "Common targets:"
        echo "  x86_64-pc-windows-gnu"
        echo "  x86_64-apple-darwin"
        echo "  aarch64-unknown-linux-gnu"
        echo "  armv7-unknown-linux-gnueabihf"
        return 1
    fi
    
    cross build --release --target "$1"
}'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$rust_cli_aliases" "Rust CLI aliases"
    fi
    
    if [[ -f "$HOME/.zshrc" ]]; then
        add_to_file_if_missing "$HOME/.zshrc" "$rust_cli_aliases" "Rust CLI aliases"
    fi
    
    # Add cargo/bin to PATH if not already there
    if [[ ":$PATH:" != *":$HOME/.cargo/bin:"* ]]; then
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    fi
    
    # Configure sccache
    sccache_config='
# sccache configuration
export RUSTC_WRAPPER=sccache
export SCCACHE_CACHE_SIZE="10G"'
    
    if [[ -f "$HOME/.bashrc" ]]; then
        add_to_file_if_missing "$HOME/.bashrc" "$sccache_config" "sccache configuration"
    fi
    
    log_success "Rust CLI development aliases added to shell"
}

# Run main function
main "$@" 