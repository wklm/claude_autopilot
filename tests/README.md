# Claude Flutter Firebase Agent Test Suite

This directory contains comprehensive tests for the Claude Flutter Firebase Agent, ensuring it works correctly for Carenji app development within Docker tmux containers.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_settings.py     # Settings configuration and validation
│   ├── test_utils.py        # Utility functions and helpers
│   ├── test_monitor.py      # Agent monitoring functionality
│   └── test_models.py       # Data models and structures
├── integration/             # Integration tests
│   ├── test_docker_setup.py # Docker container and tools
│   ├── test_tmux_integration.py # Tmux session management
│   ├── test_agent_lifecycle.py # Complete agent lifecycle
│   └── test_carenji_tools.py # Carenji development tools
├── e2e/                     # End-to-end tests
│   ├── test_carenji_queries.py # Real-world Carenji queries
│   ├── test_usage_limits.py # Usage limit handling
│   └── test_firebase_integration.py # Firebase emulator integration
├── fixtures.py              # Shared test fixtures
├── mocks.py                 # Mock implementations
├── conftest.py              # Pytest configuration and fixtures
└── README.md                # This file
```

## Test Coverage

The test suite ensures:

1. **Agent Lifecycle**: The agent starts correctly in tmux, processes prompts, and restarts automatically when tasks complete or errors occur.

2. **Carenji Tools**: All necessary tools for Carenji development are available:
   - Flutter SDK and commands
   - Firebase CLI and emulators
   - Git version control
   - Build tools (make, gcc, cmake)
   - Package managers (npm, yarn)
   - Testing frameworks

3. **Usage Limit Handling**: Proper detection and scheduling of restarts when Claude usage limits are hit.

4. **Firebase Integration**: Complete Firebase emulator suite works correctly for Carenji's backend needs.

## Running Tests

### Run all tests:
```bash
pytest
```

### Run specific test categories:
```bash
# Unit tests only
pytest tests/unit -v

# Integration tests (non-Docker)
pytest tests/integration -m "not docker" -v

# Docker-based tests
pytest tests/integration tests/e2e -m docker -v

# Carenji-specific tests
pytest -m carenji -v

# Firebase tests
pytest -m firebase -v
```

### Run with coverage:
```bash
pytest --cov=claude_code_agent_farm --cov-report=html
```

## Test Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.docker` - Tests requiring Docker
- `@pytest.mark.carenji` - Carenji-specific tests
- `@pytest.mark.firebase` - Firebase-related tests
- `@pytest.mark.slow` - Long-running tests

## Key Test Scenarios

### 1. Agent Monitoring (`test_agent_lifecycle.py`)
Tests the complete lifecycle including:
- Agent startup in tmux
- Prompt sending and processing
- Status monitoring (READY, WORKING, ERROR, USAGE_LIMIT)
- Automatic restart on task completion
- Error recovery with delay
- Usage limit detection and scheduled restart

### 2. Carenji Queries (`test_carenji_queries.py`)
Tests real-world development scenarios:
- "What is this app about?" - Tests understanding of Carenji
- "List main features" - Tests knowledge of app capabilities
- "Run flutter analyze" - Tests code analysis
- "Fix errors" - Tests error correction
- "Run tests with coverage" - Tests quality assurance

### 3. Docker Integration (`test_docker_setup.py`)
Verifies the Docker container includes:
- Flutter SDK (latest stable)
- Firebase CLI tools
- Tmux with proper configuration
- Python and Node.js
- All required build tools
- Correct user permissions

### 4. Firebase Integration (`test_firebase_integration.py`)
Tests Firebase emulator functionality:
- Emulator startup and configuration
- Firestore CRUD operations
- Authentication with role-based access
- Storage operations
- Real-time updates
- Offline support

## Mocking Strategy

The test suite uses comprehensive mocks to enable testing without external dependencies:

- **MockClaudeAPI**: Simulates Claude responses for various queries
- **MockFirebaseEmulatorSuite**: Complete Firebase emulator simulation
- **MockTmuxSession**: Tmux session management without real tmux
- **MockDockerContainer**: Docker operations without containers

## CI/CD Integration

GitHub Actions workflows run tests automatically:
- On every push to main/develop branches
- On pull requests
- Daily scheduled runs
- Multi-Python version testing (3.8-3.11)
- Docker image building and testing
- Security scanning with Trivy
- Coverage reporting with Codecov

## Test Data

The suite includes realistic test data for Carenji:
- Patient records with medical conditions
- Medication schedules and tracking
- Vital signs monitoring data
- Staff scheduling information
- Family portal access patterns

## Performance

Tests are optimized for speed:
- Unit tests run without Docker or external services
- Integration tests use lightweight mocks where possible
- E2E tests run only when necessary
- Parallel test execution supported

## Maintenance

To add new tests:
1. Place unit tests in `tests/unit/`
2. Place integration tests in `tests/integration/`
3. Place end-to-end tests in `tests/e2e/`
4. Add appropriate markers for test categorization
5. Update this README with new test descriptions

## Troubleshooting

Common issues and solutions:

1. **Docker tests failing**: Ensure Docker daemon is running and you have permissions
2. **Tmux tests failing**: Install tmux system package (`apt-get install tmux`)
3. **Firebase tests failing**: Check if ports 8080, 9099, 9199 are available
4. **Coverage too low**: Run all test categories, not just unit tests