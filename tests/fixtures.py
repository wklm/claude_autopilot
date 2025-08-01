"""Shared test fixtures and utilities for Claude Flutter Firebase Agent tests."""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import Mock

import pytest

from claude_code_agent_farm.models_new.events import SystemEvent, EventType
from claude_code_agent_farm.models_new.session import AgentStatus, UsageLimitInfo


class MockClaudeResponses:
    """Collection of mock Claude responses for various scenarios."""

    # Basic responses
    READY = ">> "
    THINKING = "Let me analyze that..."
    WORKING = "Working on implementing the feature..."

    # Carenji-specific responses
    CARENJI_INTRO = """Looking at the Carenji healthcare management system...

This is a comprehensive Flutter application designed for nursing homes and care facilities. 
The app uses Firebase as the backend and follows clean architecture principles with:
- MVVM pattern for presentation layer
- Provider for state management
- GetIt + Injectable for dependency injection
- Comprehensive test coverage (>80%)

Key features include medication management, vital signs monitoring, staff scheduling, 
and a secure family portal for relatives to stay connected with their loved ones.
>> """

    # Error responses
    ERROR_FIREBASE = "Error: Failed to connect to Firebase emulator at localhost:8080"
    ERROR_FLUTTER = "Error: Flutter SDK not found in PATH"
    ERROR_BUILD = "FAILURE: Build failed with an exception."
    ERROR_TEST = "✗ Tests failed: 3 failures out of 45 tests"

    # Usage limit responses
    USAGE_LIMIT_FORMATS = [
        "You've reached your usage limit. Please try again at {time} {tz}",
        "Usage limit reached. Come back at {time} {tz}",
        "Rate limit hit. Try again at {time} {tz}",
        "Daily limit exceeded. Please try again at {time} {tz} tomorrow",
    ]

    @classmethod
    def generate_usage_limit(cls, retry_time: datetime, timezone: str = "PST") -> str:
        """Generate a usage limit message with specific retry time."""
        import random

        template = random.choice(cls.USAGE_LIMIT_FORMATS)
        time_str = retry_time.strftime("%-I:%M %p")
        return template.format(time=time_str, tz=timezone)

    @classmethod
    def generate_progress(cls, steps: List[str], include_completion: bool = True) -> str:
        """Generate a progress response with multiple steps."""
        response = []
        for step in steps:
            response.append(f"✓ {step}")

        if include_completion:
            response.append("\nTask completed successfully!")
            response.append(cls.READY)

        return "\n".join(response)


class MockTmuxSession:
    """Mock tmux session for testing."""

    def __init__(self, session_name: str):
        self.session_name = session_name
        self.pane_content = []
        self.command_history = []
        self.is_alive = True

    def send_keys(self, keys: str):
        """Simulate sending keys to tmux."""
        self.command_history.append(keys)
        if keys.strip():
            self.pane_content.append(keys)

    def capture_pane(self) -> str:
        """Capture current pane content."""
        return "\n".join(self.pane_content)

    def clear_history(self):
        """Clear pane history."""
        self.pane_content = []

    def kill(self):
        """Kill the session."""
        self.is_alive = False


class MockDockerContainer:
    """Mock Docker container for testing."""

    def __init__(self, container_id: str, image: str):
        self.container_id = container_id
        self.image = image
        self.is_running = True
        self.env_vars = {}
        self.installed_tools = {
            "flutter": "3.16.0",
            "dart": "3.2.0",
            "firebase": "12.0.0",
            "git": "2.34.1",
            "tmux": "3.2a",
            "python3": "3.10.6",
            "node": "18.12.0",
            "npm": "9.2.0",
        }

    def exec_run(self, command: List[str]) -> Mock:
        """Simulate executing command in container."""
        result = Mock()

        # Simulate tool version checks
        if len(command) >= 2 and command[1] == "--version":
            tool = command[0]
            if tool in self.installed_tools:
                result.exit_code = 0
                result.output = f"{tool} version {self.installed_tools[tool]}"
            else:
                result.exit_code = 127
                result.output = f"{tool}: command not found"
        else:
            result.exit_code = 0
            result.output = "Command executed successfully"

        return result

    def stop(self):
        """Stop the container."""
        self.is_running = False

    def remove(self):
        """Remove the container."""
        self.is_running = False


@pytest.fixture
def mock_claude_cli():
    """Mock Claude CLI for testing without actual Claude."""

    class MockClaudeCLI:
        def __init__(self):
            self.responses = []
            self.response_index = 0
            self.commands_received = []

        def add_response(self, response: str):
            """Add a response to the queue."""
            self.responses.append(response)

        def get_next_response(self) -> str:
            """Get the next response in queue."""
            if self.response_index < len(self.responses):
                response = self.responses[self.response_index]
                self.response_index += 1
                return response
            return MockClaudeResponses.READY

        def reset(self):
            """Reset the mock."""
            self.responses = []
            self.response_index = 0
            self.commands_received = []

    return MockClaudeCLI()


@pytest.fixture
def mock_firebase_emulators():
    """Mock Firebase emulators for testing."""

    class MockFirebaseEmulators:
        def __init__(self):
            self.is_running = False
            self.ports = {"auth": 9099, "firestore": 8080, "storage": 9199, "functions": 5001, "ui": 4000}
            self.data = {"patients": {}, "medications": {}, "vitals": {}, "staff": {}}

        def start(self) -> bool:
            """Start emulators."""
            self.is_running = True
            return True

        def stop(self):
            """Stop emulators."""
            self.is_running = False

        def add_document(self, collection: str, doc_id: str, data: dict):
            """Add document to mock Firestore."""
            if collection not in self.data:
                self.data[collection] = {}
            self.data[collection][doc_id] = data

        def get_document(self, collection: str, doc_id: str) -> Optional[dict]:
            """Get document from mock Firestore."""
            return self.data.get(collection, {}).get(doc_id)

        def query(self, collection: str, **filters) -> List[dict]:
            """Query documents with filters."""
            results = []
            for doc_id, doc_data in self.data.get(collection, {}).items():
                match = True
                for field, value in filters.items():
                    if doc_data.get(field) != value:
                        match = False
                        break
                if match:
                    results.append({"id": doc_id, **doc_data})
            return results

    return MockFirebaseEmulators()


@pytest.fixture
def mock_carenji_codebase(temp_dir: Path) -> Path:
    """Create a comprehensive mock Carenji codebase."""
    project_root = temp_dir / "carenji_full"
    project_root.mkdir()

    # Create directory structure
    directories = [
        "lib/domain/entities",
        "lib/domain/repositories",
        "lib/domain/services",
        "lib/data/models",
        "lib/data/repositories",
        "lib/data/datasources",
        "lib/presentation/views",
        "lib/presentation/widgets",
        "lib/presentation/providers",
        "lib/core/constants",
        "lib/core/utils",
        "lib/core/errors",
        "test/unit",
        "test/widget",
        "test/integration",
        "android/app",
        "ios/Runner",
        ".vscode",
        "firebase/functions/src",
    ]

    for dir_path in directories:
        (project_root / dir_path).mkdir(parents=True)

    # Create key files
    files = {
        "pubspec.yaml": """
name: carenji
description: Healthcare management system for nursing homes
version: 2.0.0+15

environment:
  sdk: '>=2.19.0 <3.0.0'

dependencies:
  flutter:
    sdk: flutter
  
  # Firebase
  firebase_core: ^2.24.0
  cloud_firestore: ^4.13.0
  firebase_auth: ^4.15.0
  firebase_storage: ^11.5.0
  cloud_functions: ^4.5.0
  
  # State Management & DI
  provider: ^6.1.0
  get_it: ^7.6.0
  injectable: ^2.3.0
  
  # UI/UX
  flutter_barcode_scanner: ^2.0.0
  charts_flutter: ^0.12.0
  intl: ^0.18.1
  
  # Utilities
  uuid: ^4.2.0
  path_provider: ^2.1.0
  connectivity_plus: ^5.0.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  build_runner: ^2.4.0
  injectable_generator: ^2.4.0
  flutter_lints: ^3.0.0
  test: ^1.24.0
  mockito: ^5.4.0
  integration_test:
    sdk: flutter
""",
        "lib/main.dart": """
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:carenji/injection.dart';
import 'package:carenji/presentation/app.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await configureDependencies();
  runApp(const CarenjiApp());
}
""",
        "lib/injection.dart": """
import 'package:get_it/get_it.dart';
import 'package:injectable/injectable.dart';
import 'injection.config.dart';

final getIt = GetIt.instance;

@InjectableInit()
Future<void> configureDependencies() async => getIt.init();
""",
        "lib/domain/entities/patient.dart": """
class Patient {
  final String id;
  final String firstName;
  final String lastName;
  final DateTime dateOfBirth;
  final String roomNumber;
  final List<String> conditions;
  final List<String> medications;
  final String primaryCaregiverId;
  
  const Patient({
    required this.id,
    required this.firstName,
    required this.lastName,
    required this.dateOfBirth,
    required this.roomNumber,
    required this.conditions,
    required this.medications,
    required this.primaryCaregiverId,
  });
}
""",
        "lib/domain/entities/medication.dart": """
class Medication {
  final String id;
  final String patientId;
  final String name;
  final String dosage;
  final String frequency;
  final DateTime startDate;
  final DateTime? endDate;
  final String prescribedBy;
  final List<String> sideEffects;
  final bool requiresRefrigeration;
  
  const Medication({
    required this.id,
    required this.patientId,
    required this.name,
    required this.dosage,
    required this.frequency,
    required this.startDate,
    this.endDate,
    required this.prescribedBy,
    required this.sideEffects,
    required this.requiresRefrigeration,
  });
}
""",
        "test/widget/patient_view_test.dart": """
import 'package:flutter_test/flutter_test.dart';
import 'package:carenji/presentation/views/patient_view.dart';

void main() {
  testWidgets('Patient view displays patient information', (tester) async {
    // Test implementation
  });
}
""",
        "firebase.json": """
{
  "emulators": {
    "auth": {"port": 9099},
    "firestore": {"port": 8080},
    "storage": {"port": 9199},
    "functions": {"port": 5001},
    "ui": {"enabled": true, "port": 4000}
  }
}
""",
        "CLAUDE.md": """
# Carenji Development Guide

## Overview
Carenji is a healthcare management system for nursing homes built with Flutter and Firebase.

## Architecture
- Clean Architecture with MVVM
- Provider for state management
- GetIt + Injectable for dependency injection

## Key Features
1. Patient Management
2. Medication Tracking
3. Vital Signs Monitoring
4. Staff Scheduling
5. Family Portal

## Development Commands
- `flutter analyze` - Run static analysis
- `flutter test` - Run all tests
- `flutter test --coverage` - Generate coverage report
- `firebase emulators:start` - Start Firebase emulators

## Code Standards
- Minimum 80% test coverage
- Follow Flutter style guide
- Use dependency injection for all services
""",
    }

    for file_path, content in files.items():
        (project_root / file_path).write_text(content.strip())

    return project_root


@pytest.fixture
def mock_agent_states():
    """Generate various agent states for testing."""

    states = {
        "fresh": {"status": AgentStatus.READY, "runs": 0, "events": [], "usage_limit_hits": 0},
        "working": {
            "status": AgentStatus.WORKING,
            "runs": 1,
            "events": [
                SystemEvent(event_type=EventType.AGENT_STARTED, message="Agent started", details={}),
                SystemEvent(event_type=EventType.PROMPT_SENT, message="Working on tests", details={}),
            ],
            "usage_limit_hits": 0,
        },
        "completed_once": {
            "status": AgentStatus.READY,
            "runs": 1,
            "events": [
                SystemEvent(event_type=EventType.AGENT_STARTED, message="Agent started", details={}),
                SystemEvent(event_type=EventType.AGENT_WORKING, message="Tests implemented", details={"task_completed": True}),
            ],
            "usage_limit_hits": 0,
        },
        "usage_limited": {
            "status": AgentStatus.USAGE_LIMIT,
            "runs": 2,
            "events": [SystemEvent(event_type=EventType.USAGE_LIMIT_HIT, message="Limit hit at 3:45 PM", details={})],
            "usage_limit_hits": 1,
            "usage_limit_info": UsageLimitInfo(
                message="Try again at 3:45 PM PST", retry_time=datetime.now() + timedelta(hours=2)
            ),
        },
        "error_state": {
            "status": AgentStatus.ERROR,
            "runs": 1,
            "events": [SystemEvent(event_type=EventType.ERROR_OCCURRED, message="Firebase connection failed", details={})],
            "usage_limit_hits": 0,
        },
    }

    return states


@pytest.fixture
def performance_monitor():
    """Monitor for tracking test performance."""

    class PerformanceMonitor:
        def __init__(self):
            self.start_times = {}
            self.durations = {}

        def start(self, operation: str):
            """Start timing an operation."""
            self.start_times[operation] = time.time()

        def stop(self, operation: str) -> float:
            """Stop timing and return duration."""
            if operation in self.start_times:
                duration = time.time() - self.start_times[operation]
                self.durations[operation] = duration
                del self.start_times[operation]
                return duration
            return 0.0

        def get_report(self) -> Dict[str, float]:
            """Get performance report."""
            return self.durations.copy()

    return PerformanceMonitor()


@pytest.fixture
def subprocess_recorder():
    """Record subprocess calls for verification."""

    class SubprocessRecorder:
        def __init__(self):
            self.calls = []

        def record_call(self, args, **kwargs):
            """Record a subprocess call."""
            self.calls.append({"args": args, "kwargs": kwargs, "timestamp": datetime.now()})

        def get_calls_matching(self, pattern: str) -> List[dict]:
            """Get calls matching a pattern."""
            matching = []
            for call in self.calls:
                if pattern in " ".join(call["args"]):
                    matching.append(call)
            return matching

        def clear(self):
            """Clear recorded calls."""
            self.calls = []

    return SubprocessRecorder()


# Utility functions for tests


def create_mock_flutter_output(success: bool = True, test_count: int = 45) -> str:
    """Create mock Flutter test output."""
    if success:
        return f"00:03 +{test_count}: All tests passed!"
    else:
        failures = 3
        return f"00:03 +{test_count - failures} -{failures}: Some tests failed."


def create_mock_firebase_status(running: bool = True) -> str:
    """Create mock Firebase emulator status output."""
    if running:
        return """
┌─────────────────────────────────────────────────────────────┐
│ ✔  All emulators ready! It is now safe to connect your app. │
└─────────────────────────────────────────────────────────────┘
"""
    else:
        return "No emulators running."


def generate_carenji_test_data() -> Dict[str, any]:
    """Generate test data for Carenji app."""
    return {
        "patients": [
            {
                "id": "patient_001",
                "firstName": "John",
                "lastName": "Doe",
                "dateOfBirth": "1950-05-15",
                "roomNumber": "205",
                "conditions": ["Diabetes Type 2", "Hypertension"],
                "medications": ["med_001", "med_002"],
            }
        ],
        "medications": [
            {
                "id": "med_001",
                "name": "Metformin",
                "dosage": "500mg",
                "frequency": "Twice daily",
                "patientId": "patient_001",
            }
        ],
        "vitals": [
            {
                "id": "vital_001",
                "patientId": "patient_001",
                "type": "blood_pressure",
                "value": "120/80",
                "timestamp": datetime.now().isoformat(),
            }
        ],
    }
