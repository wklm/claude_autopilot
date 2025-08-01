# Claude Flutter Agent - Alias Reference Card 🚀

All commands automatically append " ultrathink" to enable Claude's thinking mode.

## Basic Commands
```bash
cfa "Your prompt here"           # Run with a direct prompt
cfa @requirements.md            # Run with prompt from file
cfa-attach                      # Attach to running session
cfa-stop                        # Stop running session
cfa-config                      # Show configuration
cfa-help                        # Show help
```

## Quick Actions
```bash
cfa-fix                         # Fix Flutter analyzer errors
cfa-test                        # Write comprehensive tests
cfa-lint                        # Run flutter analyze and fix issues
cfa-format                      # Format all Dart files
cfa-deps                        # Update dependencies to latest versions
cfa-clean                       # Clean project and ensure fresh build
```

## Development Workflow
```bash
cfa-feature prompt.md           # Run with a feature specification file
cfa-review                      # Review code for best practices
cfa-pr                          # Prepare pull request description
cfa-todo                        # Find all TODO comments
cfa-refactor                    # Suggest refactoring improvements
cfa-doc                         # Add missing documentation
```

## Carenji-Specific
```bash
cfa-carenji                     # Switch to carenji directory and run
cfa-carenji-firebase            # Set up Firebase emulators
cfa-carenji-test                # Run all tests and fix failures
cfa-carenji-build               # Build app for all platforms
```

## Examples

### Simple task:
```bash
cfa "Add a loading spinner to the login screen"
```

### Complex feature:
```bash
echo "Create a medication reminder feature with:
- Daily notifications
- Snooze functionality
- History tracking
- Family notifications" > feature.md

cfa @feature.md
```

### Quick fixes:
```bash
cfa-fix    # Automatically fixes all analyzer issues
cfa-test   # Writes tests for recent changes
cfa-lint   # Ensures code quality
```

## Tips
- All prompts automatically include "ultrathink" for better reasoning
- Use `@filename` syntax for longer prompts
- The agent restarts automatically after each task
- Usage limits are handled with smart retry logic

## Installation
Run the setup script to install these aliases:
```bash
./setup_aliases.sh
source ~/.bashrc  # or ~/.zshrc
```