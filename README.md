# Claude Flutter Firebase Agent 🦋🔥

> AI-powered development assistant specifically tailored for the Carenji healthcare Flutter app with Firebase backend

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flutter](https://img.shields.io/badge/Flutter-3.24+-02569B.svg)](https://flutter.dev)
[![Firebase](https://img.shields.io/badge/Firebase-Latest-FFA000.svg)](https://firebase.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What is this?

Claude Flutter Firebase Agent is a specialized AI development assistant that monitors and manages Claude sessions for developing the Carenji healthcare app. It provides intelligent automation, Firebase emulator integration, Flutter MCP support, and automatic usage limit handling to maximize development productivity.

### Key Features

- 🦋 **Flutter-Specific**: Tailored for Flutter development with hot reload detection and MCP integration
- 🔥 **Firebase Integration**: Built-in support for Firebase emulators and Firestore rules
- 🏥 **Carenji-Focused**: Pre-configured for the Carenji healthcare app architecture
- 🔄 **Auto-Resume**: Handles Claude usage limits intelligently with automatic retry
- 📊 **Real-time Monitoring**: Live status dashboard with Flutter and Firebase metrics
- 🧪 **Test Coverage**: Ensures 80% minimum test coverage following Carenji standards
- 📱 **MCP Support**: Flutter Model Context Protocol for enhanced AI assistance
- 🐳 **Docker Ready**: Complete containerized development environment

### Enhanced Monitoring Features (v1.1.0)

- 🎯 **Advanced Usage Limit Detection**: 28 different patterns for detecting rate limits
- 📈 **Exponential Backoff**: Smart retry timing with jitter (60s → 120s → 240s → ...)
- 🏥 **Health Checks**: Pre/post restart validation ensures reliability
- 🐕 **Watchdog Timer**: Automatically detects and recovers from hung agents
- 💾 **Session Checkpoints**: Periodic state saves enable crash recovery
- 🚦 **Restart Limits**: Prevents rapid cycling with cooldown periods
- 🔍 **Stuck Pattern Detection**: Identifies when agent is stuck on repetitive tasks

## 📋 Prerequisites

### On Your Host Machine
- **Docker** and **Docker Compose**
- **Git** (for repository management)
- **Claude CLI** (optional - can be mounted from host)

### Provided in Docker Container
- **Flutter SDK 3.24+**
- **Firebase CLI**
- **Python 3.9+**
- **tmux** (for session management)
- **All Carenji development dependencies**

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the agent repository
git clone https://github.com/yourusername/claude-flutter-firebase-agent.git
cd claude-flutter-firebase-agent

# Set the path to your carenji project
export CARENJI_PATH=/path/to/carenji
```

### 2. Simple Installation (No Dependencies)

```bash
# Add alias to your shell configuration
echo "alias claude-flutter-agent='/home/wojtek/dev/claude_code_agent_farm/bin/claude-flutter-agent-simple'" >> ~/.bashrc
source ~/.bashrc

# Now you can use from anywhere
claude-flutter-agent run -p "Fix all Flutter analyzer errors"
```

### 3. Run with Docker Compose

```bash
# Start the agent
docker-compose up -d

# View agent output
docker-compose logs -f claude-carenji-agent

# Attach to the tmux session
docker exec -it claude-carenji-agent tmux attach-session -t claude-carenji
```

### 4. Run Locally with Full Features

```bash
# Install dependencies
pip install typer rich pydantic pydantic-settings

# Install the agent
pip install -e .

# Run with a specific prompt
claude-flutter-agent run --prompt-text "Fix all Flutter analyzer errors in carenji"

# Or use a prompt file
claude-flutter-agent run --prompt-file prompts/implement_feature.txt
```

## 🏥 Carenji-Specific Features

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

### Using Docker Compose Emulators

```bash
# Start with Firebase emulators
docker-compose --profile with-emulators up -d

# Emulator ports:
# - 4000: Emulator UI
# - 8079: Firestore  
# - 9098: Auth
# - 5001: Functions
# - 9199: Storage
```

### Using Carenji's Emulators

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

### Quick Command Usage
```bash
# Simple one-liner from anywhere
claude-flutter-agent "Fix all Flutter analyzer errors"

# With explicit prompt flag
claude-flutter-agent run -p "Implement medication reminder notifications"

# Using a prompt file
claude-flutter-agent run -f /path/to/prompt.txt
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

### Monitor Agent Status
```bash
# Check if agent is running
claude-flutter-agent status

# Attach to see Claude working
claude-flutter-agent attach

# Stop the agent
claude-flutter-agent stop
```

## 🛠️ Development Scripts

### Start Development Environment
```bash
./scripts/start-carenji-dev.sh
```

### Run All Tests
```bash
./scripts/run-carenji-tests.sh
```

### Check Coverage
```bash
./scripts/check-coverage.sh
```

## 📊 Monitoring

### View Agent Status
```bash
# Attach to tmux session
docker exec -it claude-carenji-agent tmux attach-session -t claude-carenji

# View logs
docker-compose logs -f claude-carenji-agent
```

### Status Dashboard Shows
- Current Claude status (working/idle/usage_limit)
- Total runs and restarts
- Usage limit tracking with retry times
- Firebase emulator health
- Flutter compilation status

## 🚨 Troubleshooting

### Claude Not Found
```bash
# Install Claude CLI on host
npm install -g @anthropic-ai/claude-cli

# Or mount from host in docker-compose.yml
volumes:
  - /usr/local/bin/claude:/usr/local/bin/claude:ro
```

### Firebase Emulators Not Starting
```bash
# Check ports are available
lsof -i :8079,9098,5001,4001

# Use alternative ports in docker-compose.yml
```

### Flutter MCP Not Connecting
```bash
# Ensure Flutter is running with correct flags
# Check VM service is accessible
curl http://localhost:8182
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

**Note**: This agent is specifically designed for Carenji development. For general Flutter projects, you may need to adjust the configuration.