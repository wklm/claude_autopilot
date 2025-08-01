# Claude Flutter Agent 🤖🦋

> An intelligent monitoring system for Claude CLI that manages Flutter development sessions with automatic restart, usage limit handling, and Firebase integration

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flutter](https://img.shields.io/badge/Flutter-3.24+-02569B.svg)](https://flutter.dev)
[![Firebase](https://img.shields.io/badge/Firebase-Latest-FFA000.svg)](https://firebase.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What is this?

Claude Flutter Agent is an intelligent monitoring system that manages Claude CLI sessions for Flutter development. It provides automatic error recovery, usage limit handling, Firebase integration, and session persistence to maximize development productivity.

### Key Features

- 🔄 **Automatic Recovery**: Restarts Claude on errors, completion, or usage limits
- ⏰ **Smart Usage Limit Handling**: Detects and waits for rate limits with intelligent retry
- 🦋 **Flutter Integration**: Built for Flutter development with MCP support
- 🔥 **Firebase Support**: Integrated Firebase emulator management
- 💾 **Session Persistence**: Saves and restores session state across restarts
- 📊 **Real-time Monitoring**: Live status updates and health checks
- 🎯 **Flexible Configuration**: Environment variables, config files, and CLI options
- 🚀 **Easy Setup**: Simple bash aliases for common tasks

### Advanced Features

- 📈 **Exponential Backoff**: Smart retry strategy for usage limits
- 🏥 **Health Monitoring**: Tracks session health and performance
- 🐕 **Watchdog Timer**: Detects and recovers from hung sessions
- 💾 **State Management**: Comprehensive session and event tracking
- 🔍 **Pattern Detection**: Identifies usage limit messages intelligently
- ⚡ **Fast Response**: Sub-second monitoring intervals available
- 🛠️ **Extensible**: Clean architecture with Pydantic models

## 📋 Prerequisites

- **Python 3.10+** 
- **Claude CLI** installed and configured
- **tmux** for session management
- **Flutter SDK** (for Flutter development)
- **Firebase CLI** (optional, for Firebase features)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/claude_code_agent_farm.git
cd claude_code_agent_farm

# Run the setup script to install aliases
./setup_aliases.sh
source ~/.bashrc  # or ~/.zshrc
```

### 2. Basic Usage

```bash
# Run with a prompt
cfa --prompt-text "Implement user authentication"

# Run with a prompt file  
cfa --prompt-file ./prompts/feature.md

# Quick commands
cfa-fix      # Fix Flutter analyzer errors
cfa-test     # Write tests for recent changes
cfa-review   # Review code for best practices
```

### 3. Advanced Installation

```bash
# For full Python version with all features
cd bin
./claude-flutter-agent-python --help

# The Python launcher will automatically:
# - Create a virtual environment
# - Install all dependencies
# - Run the agent with your configuration
```

## 🏥 Carenji Project Support

### Pre-configured for Carenji Architecture
- Clean architecture with domain/data/presentation layers
- Built-value serialization support
- Injectable dependency injection
- Comprehensive test infrastructure

### Monitored Features
- 💊 Medication management and tracking
- 📊 Vitals monitoring with real-time updates
- 👥 Staff scheduling with OR-Tools optimization
- 👨‍👩‍👧‍👦 Family portal with secure access
- 📷 Barcode/QR code scanning

### Automatic Checks
- Flutter analyzer compliance
- Test coverage (80% minimum)
- Firestore security rules validation
- Built-value model generation
- Import path correctness

## 🔧 Configuration

### Environment Variables

```bash
# Project settings
CLAUDE_PROJECT_PATH=/path/to/carenji
CLAUDE_PROMPT_TEXT="Your development task"

# Firebase settings
CLAUDE_FIREBASE_PROJECT_ID=carenji-24ab8
CLAUDE_FIREBASE_EMULATORS_ENABLED=true

# Flutter MCP settings  
CLAUDE_MCP_ENABLED=true
CLAUDE_MCP_VMSERVICE_PORT=8182
CLAUDE_MCP_DDS_PORT=8181

# Agent behavior
CLAUDE_WAIT_ON_LIMIT=true
CLAUDE_RESTART_ON_COMPLETE=true
CLAUDE_CHECK_INTERVAL=5
```

### Configuration File (.env)

Create a `.env` file in the agent directory:

```env
# Carenji project location
CLAUDE_PROJECT_PATH=/home/user/dev/carenji

# Default prompt
CLAUDE_PROMPT_TEXT=Help develop carenji following CLAUDE.md guidelines

# Enable all carenji features
CLAUDE_CARENJI_FEATURES_ENABLED=medication_management,vitals_monitoring,staff_scheduling,family_portal,barcode_scanning
```

## 🔥 Firebase Emulator Integration

### Automatic detection and startup

The agent automatically detects Firebase configuration and starts emulators if needed.

### Using Project Emulators

```bash
# In the carenji directory
cd ../carenji
docker-compose -f docker-compose.emulators.yml up -d
```

## 🦋 Flutter MCP Integration

When developing with AI assistance, ensure Flutter runs with MCP flags:

```bash
# The agent automatically uses these flags
flutter run --debug \
  --host-vmservice-port=8182 \
  --dds-port=8181 \
  --enable-vm-service \
  --disable-service-auth-codes
```

MCP tools available:
- `get_app_errors`: Retrieve Flutter errors and stack traces
- `view_screenshot`: Capture current app state
- `get_view_details`: Inspect widget tree and state

## 📝 Common Tasks

### Using Aliases
```bash
# Quick fixes
cfa-fix                    # Fix analyzer errors
cfa-test                   # Write tests
cfa-review                 # Code review

# Custom prompts
cfa --prompt-text "Your task here"
cfa --prompt-file feature.md
```

### Fix Analyzer Errors
```bash
claude-flutter-agent run -p "Fix all Flutter analyzer errors following carenji standards"
```

### Implement New Feature
```bash
claude-flutter-agent run -p "Implement medication reminder notifications with tests"
```

### Improve Test Coverage
```bash
claude-flutter-agent run -p "Add tests to achieve 90% coverage for medication management"
```

### Review and Refactor
```bash
claude-flutter-agent run -p "Review vitals monitoring for performance optimization"
```

### Session Management
```bash
# Attach to running session
cfa-attach

# Stop the agent
cfa-stop

# Show configuration
cfa-config
```

## 🛠️ Development

### Running Tests
```bash
# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest

# With coverage
pytest --cov=claude_code_agent_farm
```

### Code Quality
```bash
# Linting
ruff check src/

# Type checking  
mypy src/
```

## 📊 Architecture

### Clean Module Structure
```
src/claude_code_agent_farm/
├── models_new/          # Pydantic models
│   ├── session.py      # Session state management
│   ├── events.py       # Event tracking system
│   └── commands.py     # Command execution models
├── utils/              # Utilities
│   ├── shell.py        # Safe subprocess execution
│   ├── flutter_helpers.py  # Flutter/Firebase helpers
│   └── time_parser.py  # Usage limit parsing
├── flutter_agent_monitor.py  # Main monitoring logic
├── flutter_agent_settings.py # Configuration
└── flutter_agent_cli.py      # CLI interface
```

## 🚨 Troubleshooting

### Common Issues

1. **tmux not found**: Install with `sudo apt install tmux`
2. **Import errors**: Dependencies are auto-installed by the Python launcher
3. **Claude CLI not found**: Ensure Claude CLI is in your PATH
4. **Firebase issues**: Check emulator ports in configuration

### Debug Mode
```bash
CLAUDE_LOG_LEVEL=DEBUG cfa --prompt-text "Debug this"
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Follow Carenji's coding standards
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built specifically for the [Carenji Healthcare App](https://github.com/yourusername/carenji)
- Powered by [Claude](https://claude.ai) AI assistant
- Uses [Flutter](https://flutter.dev) and [Firebase](https://firebase.google.com)

---

**Note**: This agent includes special support for the Carenji healthcare app but works with any Flutter project.