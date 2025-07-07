#!/bin/bash
# Wrapper script for claude command that ensures proper environment setup
# This script is placed in /usr/local/bin/cc in the Docker container

# Set up environment variables
export ENABLE_BACKGROUND_TASKS=1
export CLAUDE_DANGEROUSLY_SKIP_PERMISSIONS=1

# Configure MCP if enabled
if [ "${MCP_ENABLED}" = "true" ] && [ -n "${MCP_SERVER_URL}" ]; then
    export CLAUDE_MCP_ENABLED=true
    export CLAUDE_MCP_SERVER_URL="${MCP_SERVER_URL}"
    export CLAUDE_MCP_TRANSPORT=http
    
    # Log MCP configuration for debugging
    echo "MCP enabled: connecting to ${MCP_SERVER_URL}" >&2
fi

# Add Flutter and Android SDK to PATH
export FLUTTER_HOME=/opt/flutter
export ANDROID_HOME=/opt/android-sdk
export PATH="${FLUTTER_HOME}/bin:${ANDROID_HOME}/cmdline-tools/latest/bin:${ANDROID_HOME}/platform-tools:${PATH}"

# Set up Flutter cache directories in user's home
export PUB_CACHE="$HOME/.pub-cache"
export FLUTTER_STORAGE_BASE_URL="https://storage.googleapis.com"
export FLUTTER_CACHE_DIR="$HOME/.flutter/cache"

# Create cache directories if they don't exist
mkdir -p "$PUB_CACHE" 2>/dev/null || true
mkdir -p "$FLUTTER_CACHE_DIR" 2>/dev/null || true

# Check if claude-auto-resume is available
if command -v claude-auto-resume >/dev/null 2>&1; then
    # Use claude-auto-resume for automatic API quota handling
    exec claude-auto-resume /home/claude/.venv/bin/claude --dangerously-skip-permissions "$@"
else
    # Fallback to direct claude execution
    exec /home/claude/.venv/bin/claude --dangerously-skip-permissions "$@"
fi