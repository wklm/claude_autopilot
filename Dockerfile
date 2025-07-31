# Claude Flutter Firebase Agent for Carenji Development
# This Docker image provides a complete environment for AI-assisted 
# Flutter development with Firebase backend for the carenji healthcare app

FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install base dependencies
RUN apt-get update && apt-get install -y \
    # Basic tools
    curl \
    git \
    wget \
    unzip \
    xz-utils \
    zip \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    # Python for scripts
    python3 \
    python3-pip \
    python3-venv \
    # Terminal tools
    tmux \
    htop \
    nano \
    vim \
    # Networking tools
    net-tools \
    iputils-ping \
    # Required for Flutter
    libglu1-mesa \
    libgtk-3-0 \
    libstdc++6 \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (required for Firebase CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Firebase CLI
RUN npm install -g firebase-tools

# Create claude user
RUN useradd -m -s /bin/bash claude && \
    echo "claude ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to claude user
USER claude
WORKDIR /home/claude

# Install Flutter
ENV FLUTTER_VERSION=3.24.3
ENV FLUTTER_HOME=/home/claude/flutter
ENV PATH="$FLUTTER_HOME/bin:$PATH"

RUN git clone https://github.com/flutter/flutter.git -b stable $FLUTTER_HOME && \
    flutter precache && \
    flutter config --no-analytics && \
    flutter doctor -v

# Install Claude CLI (placeholder - replace with actual installation when available)
# For now, we'll create a script that reminds to install Claude
RUN echo '#!/bin/bash\necho "Please install Claude CLI manually or mount from host"\necho "Visit: https://claude.ai/cli for installation instructions"\nexit 1' > /home/claude/.local/bin/claude && \
    chmod +x /home/claude/.local/bin/claude && \
    mkdir -p /home/claude/.local/bin

# Create claude-auto-resume wrapper
RUN echo '#!/bin/bash\n# Claude auto-resume wrapper for usage limit handling\nif command -v claude >/dev/null 2>&1; then\n    claude "$@"\nelse\n    echo "Claude CLI not installed. Please install it first."\n    exit 1\nfi' > /home/claude/.local/bin/claude-auto-resume && \
    chmod +x /home/claude/.local/bin/claude-auto-resume

# Add .local/bin to PATH
ENV PATH="/home/claude/.local/bin:$PATH"

# Install Python dependencies for the agent
COPY --chown=claude:claude requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --user -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy the agent code
COPY --chown=claude:claude . /home/claude/agent

# Install the agent
WORKDIR /home/claude/agent
RUN python3 -m pip install --user -e .

# Create workspace directory for carenji
RUN mkdir -p /workspace

# Set up tmux configuration for better experience
RUN echo 'set -g mouse on\nset -g history-limit 50000\nset -g status-bg colour235\nset -g status-fg colour136' > /home/claude/.tmux.conf

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Check if Claude CLI is available\n\
if ! command -v claude >/dev/null 2>&1; then\n\
    echo "⚠️  Claude CLI not found!"\n\
    echo ""\n\
    echo "Please install Claude CLI by:"\n\
    echo "1. On the host: npm install -g @anthropic-ai/claude-cli"\n\
    echo "2. Or mount it from host using Docker volumes"\n\
    echo ""\n\
    echo "For now, starting a bash shell..."\n\
    exec /bin/bash\n\
fi\n\
\n\
# Check if carenji project is mounted\n\
if [ ! -f "/workspace/pubspec.yaml" ]; then\n\
    echo "⚠️  Carenji project not found in /workspace!"\n\
    echo ""\n\
    echo "Please mount the carenji project:"\n\
    echo "docker run -v /path/to/carenji:/workspace ..."\n\
    echo ""\n\
fi\n\
\n\
# Start the Flutter Firebase Agent\n\
echo "🦋 Starting Claude Flutter Firebase Agent for Carenji Development"\n\
echo ""\n\
\n\
# Set environment variables\n\
export CLAUDE_PROJECT_PATH=/workspace\n\
export CLAUDE_SINGLE_AGENT_DOCKER=1\n\
export CLAUDE_FLUTTER_AGENT_DOCKER=1\n\
\n\
# Run the agent\n\
exec claude-flutter-agent run "$@"\n\
' > /home/claude/entrypoint.sh && \
    chmod +x /home/claude/entrypoint.sh

# Expose ports for Flutter development
# Flutter DevTools
EXPOSE 9100
# Flutter Observatory
EXPOSE 8181
# Flutter VM Service
EXPOSE 8182
# Firebase Emulator Suite
EXPOSE 4000 5000 5001 8080 8085 9000 9099 9199

# Set working directory
WORKDIR /workspace

# Set entrypoint
ENTRYPOINT ["/home/claude/entrypoint.sh"]

# Labels
LABEL maintainer="Claude Flutter Firebase Agent"
LABEL description="AI-powered Flutter development environment for carenji healthcare app"
LABEL version="1.0.0"