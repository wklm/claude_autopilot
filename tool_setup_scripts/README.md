# Tool Setup Scripts

This directory contains interactive setup scripts for various development environments on Ubuntu. Each script is designed to be safe, interactive, and non-destructive - they will always ask for confirmation before making changes to your system.

## Features

- 🎨 **Colorful Output**: Clear, color-coded logging for easy reading
- 🔒 **Safe Installation**: Always asks for confirmation before installing or modifying anything
- 🔍 **Smart Detection**: Checks for existing installations to avoid conflicts
- 🛡️ **Non-Destructive**: Won't overwrite existing configurations without permission
- 🐚 **Shell Agnostic**: Works with both bash and zsh
- 📊 **Progress Tracking**: Shows what's installed and what's pending

## Quick Start

Run the main setup script to see all available options:

```bash
cd tool_setup_scripts
./setup.sh
```

Or run a specific setup script directly:

```bash
./setup_python_fastapi.sh    # Python/FastAPI environment
./setup_go_webapps.sh        # Go web development
./setup_nextjs.sh            # Next.js/Node.js environment
# ... etc
```

## Available Setup Scripts

### 1. Python FastAPI (`setup_python_fastapi.sh`)
Installs and configures:
- Python 3.12+ (via deadsnakes PPA if needed)
- uv (fast Python package manager)
- ruff (fast Python linter/formatter)
- mypy (type checker)
- pre-commit (git hooks)
- ipython (enhanced REPL)

### 2. Go Web Apps (`setup_go_webapps.sh`)
Installs and configures:
- Go 1.23+
- golangci-lint (comprehensive linter)
- air (hot reload)
- golang-migrate (database migrations)
- mockery (mock generation)
- Task (modern Make alternative)
- swag (Swagger docs)
- govulncheck (security scanner)

### 3. Next.js (`setup_nextjs.sh`)
Installs and configures:
- Node.js 22+ (via NodeSource)
- Bun (fast JavaScript runtime)
- pnpm (efficient package manager)
- TypeScript
- ESLint & Prettier
- Vercel CLI
- create-next-app

### 4. SvelteKit/Remix/Astro (`setup_sveltekit_remix_astro.sh`)
Extends Next.js setup with:
- Vite (build tool)
- Playwright (E2E testing)
- Vitest (unit testing)
- Biome (fast linter/formatter)
- Framework-specific tools for SvelteKit, Remix, and Astro
- Lighthouse CI

### 5. Rust Development (`setup_rust.sh`)
Installs and configures:
- Rust stable toolchain via rustup
- Essential cargo tools (cargo-watch, cargo-edit, cargo-audit)
- Web development tools (sqlx-cli, sea-orm-cli, trunk)
- System programming tools (cargo-binutils, flamegraph, miri)
- rust-analyzer (LSP)
- sccache (compilation cache)

### 6. Java Enterprise (`setup_java_enterprise.sh`)
Installs and configures:
- Java 21 LTS (choice of OpenJDK, Temurin, Corretto, or Liberica)
- SDKMAN (SDK manager)
- Gradle 8.11+
- Maven 3.9+
- JBang (Java scripting)
- Docker (for Testcontainers)
- Optimized configurations

## Script Structure

Each setup script follows this pattern:

1. **System Check**: Verifies Ubuntu compatibility
2. **Current Status**: Shows what's already installed
3. **Dependencies**: Installs required system packages
4. **Main Tools**: Installs the primary development tools
5. **Additional Tools**: Offers optional related tools
6. **Configuration**: Creates sample config files
7. **Git Setup**: Configures global .gitignore patterns
8. **Summary**: Shows final status and next steps

## Common Utilities

The `common_utils.sh` file provides shared functions used by all scripts:

- **Logging**: `log_info`, `log_success`, `log_warning`, `log_error`
- **User Input**: `confirm` - always asks before making changes
- **System Checks**: `command_exists`, `check_ubuntu_version`
- **Installation**: `install_apt_package`, `update_apt_if_needed`
- **Configuration**: `add_to_file_if_missing`, `create_directory`

## Safety Features

1. **No Automatic Overwrites**: Scripts check for existing installations
2. **Confirmation Prompts**: Every significant action requires user approval
3. **Rollback Information**: Scripts explain how to undo changes when relevant
4. **Non-Root Execution**: Scripts run as normal user, only use sudo when necessary
5. **Path Safety**: Adds to PATH without overwriting existing entries

## Usage Tips

1. **Run in Order**: Some scripts depend on others (e.g., SvelteKit setup needs Node.js)
2. **Restart Shell**: After installation, restart your shell or source your rc file
3. **Check Versions**: Scripts show current vs recommended versions
4. **Skip Steps**: You can skip any installation step by answering 'n'
5. **Rerun Safe**: Scripts are idempotent - safe to run multiple times

## Customization

You can customize the scripts by:

1. **Editing Versions**: Change version numbers at the top of each script
2. **Adding Tools**: Add new `install_*` functions following the existing pattern
3. **Changing Defaults**: Modify the confirmation prompts and default choices
4. **Config Templates**: Update the sample configuration file generators

## Troubleshooting

If a script fails:

1. **Check Prerequisites**: Ensure curl, wget, git are installed
2. **Update Package Lists**: Run `sudo apt update`
3. **Check Permissions**: Ensure scripts are executable (`chmod +x`)
4. **Review Logs**: Scripts provide detailed error messages
5. **Manual Installation**: Each script shows manual commands if automation fails

## Shell Integration

Scripts automatically configure both bash and zsh when adding to PATH:
- Checks for ~/.bashrc and ~/.zshrc
- Only adds entries if they don't already exist
- Preserves existing PATH entries

## Contributing

To add a new setup script:

1. Copy an existing script as a template
2. Update the tool versions and installation commands
3. Add framework/language specific configurations
4. Update the main `setup.sh` menu
5. Test on a clean Ubuntu system

## License

These scripts are part of the claude_code_agent_farm project and follow the same license. 