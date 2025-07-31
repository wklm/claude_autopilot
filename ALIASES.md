# Claude Flutter Firebase Agent - Bash Aliases Reference

## Quick Start
```bash
source ~/.bashrc  # or ~/.zshrc
cffa-help        # Show quick reference
```

## Common Commands

### Agent Control
- `cffa` - Start the agent with default settings
- `cffa-carenji` - Start agent for Carenji development
- `cffa-stop` - Stop the agent
- `cffa-restart` - Restart the agent
- `cffa-status` - Check agent status
- `cffa-logs` - View agent logs (with follow)

### Tmux Session
- `cffa-attach` - Attach to agent tmux session
- `cffa-capture` - Capture current pane content
- `cffa-send <text>` - Send text to agent session

### Docker
- `cffa-build` - Build Docker image
- `cffa-docker` - Run agent in Docker
- `cffa-docker-carenji` - Run in Docker for Carenji
- `cffa-shell` - Shell into agent container

### Testing
- `cffa-test` - Run all tests
- `cffa-test-unit` - Run unit tests only
- `cffa-test-quick` - Run quick tests (no Docker/slow)
- `cffa-test-cov` - Run tests with coverage report
- `cffa-test-carenji` - Run Carenji-specific tests

### Firebase Emulators
- `fb-start` - Start all Firebase emulators
- `fb-kill` - Kill all emulator processes
- `fb-ui` - Open emulator UI in browser

### Flutter Shortcuts
- `fl` - flutter
- `flr` - flutter run
- `flt` - flutter test
- `fla` - flutter analyze
- `flc` - flutter clean
- `flpg` - flutter pub get

### Utilities
- `cffa-cleanup` - Clean up old sessions and logs
- `cffa-backup-logs` - Backup agent logs
- `cffa-watch` - Watch agent status (live update)
- `cffa-custom "prompt"` - Start with custom prompt

## Functions

### Start with custom prompt
```bash
cffa-custom "Implement user authentication for Carenji"
```

### Run agent for specific duration
```bash
cffa-timed 3600  # Run for 1 hour
```

### Check colored status
```bash
cffa-status-color
```

## Environment Variables
- `CLAUDE_AGENT_CONFIG` - Path to agent configuration
- `CLAUDE_EDITOR` - Editor for config files
- `CLAUDE_PROJECT_PATH` - Default project path

## Tips
1. Use `cffa-test-quick` for rapid development feedback
2. Run `cffa-cleanup` weekly to remove old logs
3. Use `cffa-docker` for isolated testing environments
4. Check `cffa-status-color` in your prompt for agent status
