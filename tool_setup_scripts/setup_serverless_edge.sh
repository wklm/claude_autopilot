#!/usr/bin/env bash

# Setup script for Serverless Edge development environment
# Installs: Node.js, Bun, Deno, Wrangler, Vercel CLI, WASM toolchain

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

# Tool versions
NODE_VERSION="22"
BUN_VERSION="latest"
DENO_VERSION="latest"
RUST_VERSION="stable"

main() {
    show_banner "Serverless Edge Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "node" "--version"
    show_tool_status "bun" "--version"
    show_tool_status "deno" "--version"
    show_tool_status "wrangler" "--version"
    show_tool_status "vercel" "--version"
    show_tool_status "rustc" "--version"
    show_tool_status "wasm-pack" "--version"
    echo
    
    # Update apt if needed
    update_apt_if_needed || exit 1
    
    # Install system dependencies
    log_step "Installing system dependencies"
    install_apt_package "curl" "cURL"
    install_apt_package "git" "Git"
    install_apt_package "build-essential" "Build Essential"
    install_apt_package "pkg-config" "pkg-config"
    install_apt_package "libssl-dev" "OpenSSL Dev"
    
    # Install Node.js
    log_step "Installing Node.js"
    if command_exists node; then
        current_node=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
        log_info "Current Node.js version: v$(node --version | cut -d'v' -f2)"
        
        if [[ "$current_node" -lt "$NODE_VERSION" ]]; then
            log_warning "Node.js version is older than recommended v${NODE_VERSION}"
            if confirm "Update Node.js to v${NODE_VERSION}?"; then
                install_nodejs
            fi
        fi
    else
        if confirm "Install Node.js v${NODE_VERSION}?"; then
            install_nodejs
        fi
    fi
    
    # Install Bun
    log_step "Installing Bun (Fast JavaScript runtime)"
    if command_exists bun; then
        log_info "Bun is already installed: $(bun --version)"
        if confirm "Update Bun to latest version?"; then
            curl -fsSL https://bun.sh/install | bash
            log_success "Bun updated"
        fi
    else
        if confirm "Install Bun (recommended for edge development)?"; then
            curl -fsSL https://bun.sh/install | bash
            
            # Add to PATH
            bun_path_line='export BUN_INSTALL="$HOME/.bun"'
            path_line='export PATH="$BUN_INSTALL/bin:$PATH"'
            
            if [[ -f "$HOME/.bashrc" ]]; then
                add_to_file_if_missing "$HOME/.bashrc" "$bun_path_line" "Bun install"
                add_to_file_if_missing "$HOME/.bashrc" "$path_line" "Bun PATH"
            fi
            
            if [[ -f "$HOME/.zshrc" ]]; then
                add_to_file_if_missing "$HOME/.zshrc" "$bun_path_line" "Bun install"
                add_to_file_if_missing "$HOME/.zshrc" "$path_line" "Bun PATH"
            fi
            
            # Source for current session
            export BUN_INSTALL="$HOME/.bun"
            export PATH="$BUN_INSTALL/bin:$PATH"
            log_success "Bun installed"
        fi
    fi
    
    # Install Deno
    log_step "Installing Deno"
    if command_exists deno; then
        log_info "Deno is already installed: $(deno --version | head -1)"
        if confirm "Update Deno to latest version?"; then
            deno upgrade
            log_success "Deno updated"
        fi
    else
        if confirm "Install Deno (for Deno Deploy)?"; then
            curl -fsSL https://deno.land/install.sh | sh
            
            # Add to PATH
            deno_path_line='export DENO_INSTALL="$HOME/.deno"'
            path_line='export PATH="$DENO_INSTALL/bin:$PATH"'
            
            if [[ -f "$HOME/.bashrc" ]]; then
                add_to_file_if_missing "$HOME/.bashrc" "$deno_path_line" "Deno install"
                add_to_file_if_missing "$HOME/.bashrc" "$path_line" "Deno PATH"
            fi
            
            if [[ -f "$HOME/.zshrc" ]]; then
                add_to_file_if_missing "$HOME/.zshrc" "$deno_path_line" "Deno install"
                add_to_file_if_missing "$HOME/.zshrc" "$path_line" "Deno PATH"
            fi
            
            # Source for current session
            export DENO_INSTALL="$HOME/.deno"
            export PATH="$DENO_INSTALL/bin:$PATH"
            log_success "Deno installed"
        fi
    fi
    
    # Install Cloudflare Wrangler
    log_step "Installing Cloudflare Wrangler"
    if ! command_exists wrangler; then
        if confirm "Install Wrangler (Cloudflare Workers CLI)?"; then
            if command_exists npm; then
                npm install -g wrangler
                log_success "Wrangler installed"
            elif command_exists bun; then
                bun install -g wrangler
                log_success "Wrangler installed via Bun"
            else
                log_error "npm or bun required to install Wrangler"
            fi
        fi
    else
        log_info "Wrangler is already installed"
        if confirm "Update Wrangler to latest version?"; then
            if command_exists npm; then
                npm update -g wrangler
            elif command_exists bun; then
                bun update -g wrangler
            fi
            log_success "Wrangler updated"
        fi
    fi
    
    # Install Vercel CLI
    log_step "Installing Vercel CLI"
    if ! command_exists vercel; then
        if confirm "Install Vercel CLI (for Vercel Edge)?"; then
            if command_exists npm; then
                npm install -g vercel
                log_success "Vercel CLI installed"
            elif command_exists bun; then
                bun install -g vercel
                log_success "Vercel CLI installed via Bun"
            else
                log_error "npm or bun required to install Vercel CLI"
            fi
        fi
    else
        log_info "Vercel CLI is already installed"
        if confirm "Update Vercel CLI to latest version?"; then
            if command_exists npm; then
                npm update -g vercel
            elif command_exists bun; then
                bun update -g vercel
            fi
            log_success "Vercel CLI updated"
        fi
    fi
    
    # Install Rust and WASM toolchain
    log_step "Setting up Rust and WASM toolchain"
    if ! command_exists rustc; then
        if confirm "Install Rust (for WASM development)?"; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source "$HOME/.cargo/env"
            log_success "Rust installed"
        fi
    else
        log_info "Rust is already installed: $(rustc --version)"
    fi
    
    if command_exists rustc; then
        # Add WASM target
        if ! rustup target list --installed | grep -q "wasm32-unknown-unknown"; then
            if confirm "Add wasm32-unknown-unknown target to Rust?"; then
                rustup target add wasm32-unknown-unknown
                log_success "WASM target added"
            fi
        else
            log_info "WASM target already installed"
        fi
        
        # Add WASI target
        if ! rustup target list --installed | grep -q "wasm32-wasi"; then
            if confirm "Add wasm32-wasi target to Rust?"; then
                rustup target add wasm32-wasi
                log_success "WASI target added"
            fi
        else
            log_info "WASI target already installed"
        fi
        
        # Install wasm-pack
        if ! command_exists wasm-pack; then
            if confirm "Install wasm-pack (WASM build tool)?"; then
                curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
                log_success "wasm-pack installed"
            fi
        else
            log_info "wasm-pack is already installed"
        fi
        
        # Install wasm-bindgen-cli
        if ! command_exists wasm-bindgen; then
            if confirm "Install wasm-bindgen-cli?"; then
                cargo install wasm-bindgen-cli
                log_success "wasm-bindgen-cli installed"
            fi
        else
            log_info "wasm-bindgen-cli is already installed"
        fi
    fi
    
    # Install additional edge tools
    log_step "Installing additional edge development tools"
    
    # Install miniflare (local Cloudflare Workers simulator)
    if ! command_exists miniflare; then
        if confirm "Install Miniflare (local Workers runtime)?"; then
            if command_exists npm; then
                npm install -g miniflare
                log_success "Miniflare installed"
            elif command_exists bun; then
                bun install -g miniflare
                log_success "Miniflare installed via Bun"
            fi
        fi
    else
        log_info "Miniflare is already installed"
    fi
    
    # Install edge-runtime (for local Vercel Edge testing)
    if ! command_exists edge-runtime; then
        if confirm "Install Edge Runtime (local Vercel Edge runtime)?"; then
            if command_exists npm; then
                npm install -g edge-runtime
                log_success "Edge Runtime installed"
            elif command_exists bun; then
                bun install -g edge-runtime
                log_success "Edge Runtime installed via Bun"
            fi
        fi
    else
        log_info "Edge Runtime is already installed"
    fi
    
    # Setup TypeScript for edge development
    log_step "Setting up TypeScript configuration"
    if confirm "Install TypeScript and edge type definitions?"; then
        if command_exists npm; then
            npm install -g typescript @types/node @cloudflare/workers-types
            log_success "TypeScript and types installed"
        elif command_exists bun; then
            bun install -g typescript @types/node @cloudflare/workers-types
            log_success "TypeScript and types installed via Bun"
        fi
    fi
    
    # Create example configurations
    log_step "Creating example configurations"
    if confirm "Create example edge project templates?"; then
        examples_dir="$HOME/edge-examples"
        mkdir -p "$examples_dir"
        
        # Cloudflare Worker example
        cat > "$examples_dir/worker-example.ts" << 'EOF'
export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    
    // Router pattern
    switch (url.pathname) {
      case '/':
        return new Response('Hello from Cloudflare Workers!');
      case '/api/time':
        return Response.json({ time: new Date().toISOString() });
      default:
        return new Response('Not Found', { status: 404 });
    }
  },
};
EOF
        
        # Vercel Edge example
        cat > "$examples_dir/vercel-edge-example.ts" << 'EOF'
import { NextRequest, NextResponse } from 'next/server';

export const config = {
  runtime: 'edge',
};

export default async function handler(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const name = searchParams.get('name') || 'World';
  
  return NextResponse.json({
    message: `Hello ${name} from Vercel Edge!`,
    region: process.env.VERCEL_REGION || 'unknown',
  });
}
EOF
        
        # Deno Deploy example
        cat > "$examples_dir/deno-deploy-example.ts" << 'EOF'
import { serve } from "https://deno.land/std@0.208.0/http/server.ts";

const handler = async (request: Request): Promise<Response> => {
  const url = new URL(request.url);
  
  if (url.pathname === "/") {
    return new Response("Hello from Deno Deploy!");
  }
  
  if (url.pathname === "/api/kv") {
    const kv = await Deno.openKv();
    const key = ["visits", url.hostname];
    const result = await kv.atomic()
      .sum(key, 1n)
      .commit();
    
    const visits = await kv.get<bigint>(key);
    return Response.json({ visits: Number(visits.value || 0n) });
  }
  
  return new Response("Not Found", { status: 404 });
};

serve(handler);
EOF
        
        log_success "Created example templates in $examples_dir"
    fi
    
    # Setup global gitignore for edge projects
    log_step "Configuring Git for edge development"
    if confirm "Add edge development patterns to global .gitignore?"; then
        gitignore_file="$HOME/.gitignore_global"
        touch "$gitignore_file"
        
        # Edge development patterns
        edge_patterns=(
            "node_modules/"
            ".wrangler/"
            ".vercel/"
            ".miniflare/"
            "dist/"
            "build/"
            ".env"
            ".env.local"
            ".dev.vars"
            "*.wasm"
            "pkg/"
            "target/"
            ".deno/"
        )
        
        for pattern in "${edge_patterns[@]}"; do
            if ! grep -Fxq "$pattern" "$gitignore_file"; then
                echo "$pattern" >> "$gitignore_file"
            fi
        done
        
        git config --global core.excludesfile "$gitignore_file"
        log_success "Global .gitignore configured for edge development"
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "node" "--version"
    show_tool_status "bun" "--version"
    show_tool_status "deno" "--version"
    show_tool_status "wrangler" "--version"
    show_tool_status "vercel" "--version"
    show_tool_status "rustc" "--version"
    show_tool_status "wasm-pack" "--version"
    show_tool_status "miniflare" "--version"
    
    echo
    log_success "Serverless Edge development environment is ready!"
    log_info "To create a new edge project, try:"
    echo -e "  ${CYAN}# Cloudflare Workers:${RESET}"
    echo -e "  ${CYAN}npm create cloudflare@latest${RESET}"
    echo -e ""
    echo -e "  ${CYAN}# Vercel Edge:${RESET}"
    echo -e "  ${CYAN}npx create-next-app@latest --example edge-functions${RESET}"
    echo -e ""
    echo -e "  ${CYAN}# Deno Deploy:${RESET}"
    echo -e "  ${CYAN}deno init my-deno-project${RESET}"
}

# Function to install Node.js
install_nodejs() {
    # Use NodeSource repository
    curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | sudo -E bash -
    install_apt_package "nodejs" "Node.js"
}

# Run main function
main "$@" 