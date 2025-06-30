#!/usr/bin/env bash

# Setup script for SvelteKit, Remix, and Astro development
# Extends the base Node.js setup with framework-specific tools

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common_utils.sh"

main() {
    show_banner "SvelteKit/Remix/Astro Development Setup"
    
    # Check Ubuntu version
    check_ubuntu_version || exit 1
    
    # First, ensure Node.js environment is set up
    log_step "Checking Node.js environment"
    if ! command_exists node || ! command_exists bun; then
        log_warning "Node.js/Bun environment not fully set up"
        if confirm "Run Node.js/Next.js setup first?"; then
            bash "$SCRIPT_DIR/setup_nextjs.sh"
        else
            log_error "Node.js and Bun are required. Please run setup_nextjs.sh first."
            exit 1
        fi
    fi
    
    # Show current status
    log_step "Current tool status"
    show_tool_status "node" "--version"
    show_tool_status "bun" "--version"
    show_tool_status "vite" "--version"
    show_tool_status "playwright" "--version"
    echo
    
    # Install framework-specific tools
    log_step "Installing framework-specific tools"
    
    # Vite (used by SvelteKit)
    if ! command_exists vite; then
        if confirm "Install Vite globally (build tool for SvelteKit)?"; then
            bun add -g vite
            log_success "Vite installed"
        fi
    else
        log_info "Vite is already installed"
    fi
    
    # Playwright for testing
    if ! command_exists playwright; then
        if confirm "Install Playwright (E2E testing framework)?"; then
            bun add -g @playwright/test
            log_success "Playwright installed"
            
            if confirm "Install Playwright browsers?"; then
                bunx playwright install
                log_success "Playwright browsers installed"
            fi
        fi
    else
        log_info "Playwright is already installed"
    fi
    
    # Install SvelteKit development tools
    log_step "SvelteKit specific tools"
    if confirm "Install SvelteKit development tools?"; then
        # Svelte language tools
        if ! command_exists svelteserver; then
            bun add -g svelte-language-server
            log_success "Svelte language server installed"
        fi
        
        # Create SvelteKit starter
        if confirm "Install create-svelte globally?"; then
            bun add -g create-svelte@latest
            log_success "create-svelte installed"
        fi
        
        # Svelte check for type checking
        if ! command_exists svelte-check; then
            bun add -g svelte-check
            log_success "svelte-check installed"
        fi
    fi
    
    # Install Remix development tools
    log_step "Remix specific tools"
    if confirm "Install Remix development tools?"; then
        # Create Remix app
        if ! command_exists create-remix; then
            bun add -g create-remix@latest
            log_success "create-remix installed"
        fi
        
        # Remix dev tools
        if confirm "Install Remix Dev Tools extension helper?"; then
            bun add -g rmx-cli
            log_success "rmx-cli installed"
        fi
    fi
    
    # Install Astro development tools
    log_step "Astro specific tools"
    if confirm "Install Astro development tools?"; then
        # Create Astro app
        if ! command_exists create-astro; then
            bun add -g create-astro@latest
            log_success "create-astro installed"
        fi
        
        # Astro language tools
        if ! command_exists astro-ls; then
            bun add -g @astrojs/language-server
            log_success "Astro language server installed"
        fi
    fi
    
    # Install additional useful tools
    log_step "Installing additional development tools"
    
    # Vitest for unit testing
    if ! command_exists vitest; then
        if confirm "Install Vitest (unit testing framework)?"; then
            bun add -g vitest
            log_success "Vitest installed"
        fi
    fi
    
    # Biome for fast formatting/linting
    if ! command_exists biome; then
        if confirm "Install Biome (fast formatter/linter alternative to ESLint+Prettier)?"; then
            bun add -g @biomejs/biome
            log_success "Biome installed"
        fi
    fi
    
    # Histoire for component development
    if confirm "Install Histoire (component story development)?"; then
        bun add -g histoire
        log_success "Histoire installed"
    fi
    
    # Create sample configuration files
    log_step "Creating sample configurations"
    
    if confirm "Create sample vite.config.ts?"; then
        create_vite_config
    fi
    
    if confirm "Create sample playwright.config.ts?"; then
        create_playwright_config
    fi
    
    if confirm "Create sample biome.json?"; then
        create_biome_config
    fi
    
    # Performance monitoring tools
    log_step "Performance tools"
    if confirm "Install Lighthouse CI for performance testing?"; then
        bun add -g @lhci/cli
        log_success "Lighthouse CI installed"
    fi
    
    # Final status
    echo
    log_step "Setup complete! Final tool status:"
    show_tool_status "node" "--version"
    show_tool_status "bun" "--version"
    show_tool_status "vite" "--version"
    show_tool_status "playwright" "--version"
    show_tool_status "vitest" "--version"
    show_tool_status "biome" "--version"
    
    echo
    log_success "Framework development environment is ready!"
    echo
    log_info "To create new projects:"
    echo -e "  ${BOLD}SvelteKit:${RESET}"
    echo -e "  ${CYAN}bunx create-svelte@latest my-sveltekit-app${RESET}"
    echo -e "  ${CYAN}cd my-sveltekit-app && bun install && bun dev${RESET}"
    echo
    echo -e "  ${BOLD}Remix:${RESET}"
    echo -e "  ${CYAN}bunx create-remix@latest my-remix-app${RESET}"
    echo -e "  ${CYAN}cd my-remix-app && bun install && bun dev${RESET}"
    echo
    echo -e "  ${BOLD}Astro:${RESET}"
    echo -e "  ${CYAN}bunx create-astro@latest my-astro-site${RESET}"
    echo -e "  ${CYAN}cd my-astro-site && bun install && bun dev${RESET}"
}

create_vite_config() {
    cat > vite.config.ts << 'EOF'
import { defineConfig } from 'vite';
import { sveltekit } from '@sveltejs/kit/vite';
// import react from '@vitejs/plugin-react'; // For React/Remix

export default defineConfig({
    plugins: [
        sveltekit(),
        // react(), // Uncomment for React/Remix
    ],
    
    // Performance optimizations
    build: {
        target: 'esnext',
        minify: 'esbuild',
        rollupOptions: {
            output: {
                manualChunks: {
                    vendor: ['svelte', '@sveltejs/kit'],
                },
            },
        },
    },
    
    // Development server configuration
    server: {
        port: 5173,
        strictPort: false,
        host: true, // Listen on all addresses
    },
    
    // Optimize dependencies
    optimizeDeps: {
        include: ['svelte', '@sveltejs/kit'],
    },
    
    // Enable source maps in production
    build: {
        sourcemap: true,
    },
});
EOF
    
    log_success "Created vite.config.ts"
}

create_playwright_config() {
    cat > playwright.config.ts << 'EOF'
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
    testDir: './tests/e2e',
    fullyParallel: true,
    forbidOnly: !!process.env.CI,
    retries: process.env.CI ? 2 : 0,
    workers: process.env.CI ? 1 : undefined,
    reporter: 'html',
    
    use: {
        baseURL: 'http://localhost:5173',
        trace: 'on-first-retry',
        screenshot: 'only-on-failure',
    },
    
    projects: [
        {
            name: 'chromium',
            use: { ...devices['Desktop Chrome'] },
        },
        {
            name: 'firefox',
            use: { ...devices['Desktop Firefox'] },
        },
        {
            name: 'webkit',
            use: { ...devices['Desktop Safari'] },
        },
        {
            name: 'Mobile Chrome',
            use: { ...devices['Pixel 5'] },
        },
        {
            name: 'Mobile Safari',
            use: { ...devices['iPhone 12'] },
        },
    ],
    
    webServer: {
        command: 'bun run dev',
        port: 5173,
        reuseExistingServer: !process.env.CI,
    },
});
EOF
    
    log_success "Created playwright.config.ts"
}

create_biome_config() {
    cat > biome.json << 'EOF'
{
    "$schema": "https://biomejs.dev/schemas/1.9.4/schema.json",
    "organizeImports": {
        "enabled": true
    },
    "linter": {
        "enabled": true,
        "rules": {
            "recommended": true,
            "complexity": {
                "noBannedTypes": "error",
                "noUselessConstructor": "error",
                "useOptionalChain": "error"
            },
            "correctness": {
                "noUnusedVariables": "error",
                "useExhaustiveDependencies": "warn"
            },
            "style": {
                "noNonNullAssertion": "warn",
                "useConst": "error",
                "useTemplate": "error"
            },
            "suspicious": {
                "noExplicitAny": "warn"
            }
        }
    },
    "formatter": {
        "enabled": true,
        "formatWithErrors": false,
        "indentStyle": "space",
        "indentWidth": 4,
        "lineWidth": 100,
        "lineEnding": "lf"
    },
    "javascript": {
        "formatter": {
            "quoteStyle": "single",
            "jsxQuoteStyle": "double",
            "semicolons": "always",
            "trailingCommas": "all",
            "arrowParentheses": "always"
        }
    },
    "files": {
        "ignore": [
            "node_modules",
            "dist",
            "build",
            ".svelte-kit",
            ".astro",
            ".remix"
        ]
    }
}
EOF
    
    log_success "Created biome.json"
}

# Run main function
main "$@" 