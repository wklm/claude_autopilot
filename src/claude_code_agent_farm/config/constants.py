"""Constants and default values for Claude Code Agent Farm."""

# Default values
DEFAULT_NUM_AGENTS = 20
DEFAULT_SESSION_NAME = "claude_agents"
DEFAULT_STAGGER_TIME = 10.0  # Seconds between agent starts
DEFAULT_WAIT_AFTER_CC = 15.0  # Seconds to wait after Claude Code starts
DEFAULT_CHECK_INTERVAL = 10  # Seconds between status checks
DEFAULT_CONTEXT_THRESHOLD = 20  # Percentage threshold for context warning
DEFAULT_IDLE_TIMEOUT = 60  # Seconds before marking agent as idle
DEFAULT_MAX_ERRORS = 3  # Maximum errors before restart

# tmux related
TMUX_CONTROLLER_WINDOW = "controller"
TMUX_AGENTS_WINDOW = "agents"

# Paths
HEARTBEATS_DIR = ".heartbeats"
PROMPTS_DIR = "prompts"
CONFIGS_DIR = "configs"
BEST_PRACTICES_DIR = "best_practices_guides"

# Claude Code indicators
CLAUDE_READY_INDICATORS = [
    "Welcome to Claude Code!",
    "│ > Try",
    "? for shortcuts",
    "/help for help",
    "cwd:",
    "Bypassing Permissions",
]

CLAUDE_WORKING_INDICATORS = [
    "✻ Pontificating",
    "● Bash(",
    "✻ Running",
    "✻ Thinking",
    "esc to interrupt",
]

CLAUDE_WELCOME_INDICATORS = [
    "Choose the text style",
    "Choose your language",
    "Let's get started",
    "run /theme",
    "Dark mode✔",
    "Light mode",
    "colorblind-friendly",
]

CLAUDE_ERROR_INDICATORS = [
    # Login/auth prompts
    "Select login method:",
    "Claude account with subscription",
    "Sign in to Claude",
    "Log in to Claude",
    "Enter your API key",
    "API key",
    # Configuration errors
    "Configuration error",
    "Settings corrupted",
    "Invalid API key",
    "Authentication failed",
    "Rate limit exceeded",
    "Claude usage limit reached",
    "Unauthorized",
    "Permission denied",
    "Failed to load configuration",
    "Invalid configuration",
    "Error loading settings",
    "Settings file is corrupted",
    "Failed to parse settings",
    "Invalid settings",
    "Corrupted settings",
    "Config corrupted",
    "configuration is corrupted",
    "Unable to load settings",
    "Error reading settings",
    "Settings error",
    "config error",
    # Parse errors
    "TypeError",
    "SyntaxError",
    "JSONDecodeError",
    "ParseError",
    # Other login-related text
    "Choose your login method",
    "Continue with Claude account",
    "I have a Claude account",
    "Create account",
]

USAGE_LIMIT_INDICATORS = [
    "Claude usage limit reached",
    "usage limit reached",
    "daily limit exceeded",
    "usage quota exceeded",
]

# Context detection patterns
CONTEXT_PATTERNS = [
    r"Context left until\s*auto-compact:\s*(\d+)%",
    r"Context remaining:\s*(\d+)%",
    r"(\d+)%\s*context\s*remaining",
    r"Context:\s*(\d+)%",
]

# Agent status values
STATUS_WORKING = "working"
STATUS_READY = "ready"
STATUS_IDLE = "idle"
STATUS_ERROR = "error"
STATUS_USAGE_LIMIT = "usage_limit"
STATUS_STARTING = "starting"
STATUS_UNKNOWN = "unknown"

# Status emojis
STATUS_EMOJIS = {
    STATUS_WORKING: "🔧",
    STATUS_READY: "✅",
    STATUS_IDLE: "💤",
    STATUS_ERROR: "❌",
    STATUS_USAGE_LIMIT: "⏸️",
    STATUS_STARTING: "🚀",
    STATUS_UNKNOWN: "❓",
}

# Retry configuration
MAX_RETRIES = 3
BASE_RETRY_DELAY = 0.5  # Base delay for exponential backoff