"""Constants for Claude Flutter Firebase Agent.

This module contains all constant values used throughout the application,
specifically tailored for Flutter & Firebase development with the carenji app.
"""

# Claude Status Indicators
USAGE_LIMIT_INDICATORS = [
    "usage limit reached",
    "daily limit exceeded",
    "usage quota exceeded",
    "try again later",
    "rate limit exceeded",
    "you've reached your usage limit",
    "you have reached the daily usage limit",
    "please wait before trying again",
    "usage limits have been reached",
    "you've hit your rate limit",
    "rate limit hit",
    "usage limit hit",
    "limit exceeded",
    "quota limit reached",
    "maximum usage reached",
    "too many requests",
    "please try again at",
    "come back at",
    "retry after",
    "available again at",
    "usage will reset at",
    "limit will reset",
    "try again tomorrow",
    "daily quota reached",
    "hourly limit reached",
    "api rate limit",
    "request limit exceeded",
    "you can try again",
]

CLAUDE_ERROR_INDICATORS = [
    "error:",
    "failed to",
    "exception:",
    "traceback",
    "an error occurred",
    "something went wrong",
    "unexpected error",
    "fatal:",
    "critical:",
]

CLAUDE_WORKING_INDICATORS = [
    "thinking...",
    "analyzing...",
    "processing...",
    "searching...",
    "reading file",
    "writing file",
    "running command",
    "executing",
    "building",
    "compiling",
    "testing",
    "generating",
]

CLAUDE_READY_INDICATORS = [
    ">>",  # Claude's prompt indicator
    "ready",
    "what would you like",
    "how can i help",
    "i'm ready",
    "task completed",
    "done",
    "finished",
]

# Flutter-specific indicators
FLUTTER_HOT_RELOAD_INDICATORS = [
    "hot reload",
    "reloaded",
    "syncing files",
    "flutter reload",
    "🔥",
]

FLUTTER_ERROR_INDICATORS = [
    "══╡ EXCEPTION CAUGHT BY",
    "flutter:",
    "dart:",
    "build failed",
    "compilation error",
    "the following assertion was thrown",
]

FLUTTER_BUILD_INDICATORS = [
    "running gradle",
    "building flutter",
    "flutter build",
    "xcode build",
    "gradle build",
]

# Firebase Emulator indicators
FIREBASE_EMULATOR_READY = [
    "all emulators ready",
    "emulator hub running",
    "firestore emulator ready",
    "auth emulator ready",
    "functions emulator ready",
]

FIREBASE_EMULATOR_ERROR = [
    "emulator failed to start",
    "port already in use",
    "firebase error",
    "emulator error",
]

# Carenji-specific patterns
CARENJI_TEST_INDICATORS = [
    "running carenji tests",
    "test/",
    "flutter test",
    "✓",  # Test pass indicator
    "✗",  # Test fail indicator
]

CARENJI_FEATURES = [
    "medication",
    "vitals",
    "scheduling",
    "family portal",
    "barcode",
    "bluetooth",
    "firebase",
    "firestore",
]

# File patterns
CARENJI_IMPORTANT_FILES = [
    "pubspec.yaml",
    "firebase.json",
    "firestore.rules",
    "CLAUDE.md",
    "lib/main.dart",
    "lib/injection.dart",
]

# Default configurations
DEFAULT_TMUX_SESSION = "claude-carenji"
DEFAULT_PROJECT_PATH = "/home/wojtek/dev/carenji"
DEFAULT_CHECK_INTERVAL = 5
DEFAULT_IDLE_TIMEOUT = 300

# Firebase configuration for carenji
CARENJI_FIREBASE_PROJECT = "carenji-24ab8"
CARENJI_FIREBASE_EMULATOR_PORTS = {
    "auth": 9098,
    "firestore": 8079,
    "functions": 5001,
    "ui": 4001,
}

# Flutter MCP configuration
FLUTTER_MCP_PORTS = {
    "vmservice": 8182,
    "dds": 8181,
}

FLUTTER_RUN_FLAGS = [
    "--debug",
    f"--host-vmservice-port={FLUTTER_MCP_PORTS['vmservice']}",
    f"--dds-port={FLUTTER_MCP_PORTS['dds']}",
    "--enable-vm-service",
    "--disable-service-auth-codes",
]

# Prompt templates for carenji development
CARENJI_PROMPT_TEMPLATES = {
    "fix_errors": "Review the current Flutter analyzer errors and fix them following carenji's coding standards in CLAUDE.md",
    "implement_feature": "Implement the feature following carenji's clean architecture with proper models, repositories, services, and ViewModels",
    "write_tests": "Write comprehensive tests for the implementation with minimum 80% coverage",
    "review_pr": "Review the changes for adherence to carenji's architecture and Flutter best practices",
}
