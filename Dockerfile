# Claude Code Agent Farm with Flutter Environment
FROM node:20-bookworm

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set up Flutter environment variables
ENV FLUTTER_HOME=/opt/flutter
ENV ANDROID_HOME=/opt/android-sdk
ENV PATH="${FLUTTER_HOME}/bin:${ANDROID_HOME}/cmdline-tools/latest/bin:${ANDROID_HOME}/platform-tools:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic tools
    curl \
    git \
    wget \
    unzip \
    xz-utils \
    zip \
    tmux \
    sudo \
    jq \
    # Flutter dependencies
    libglu1-mesa \
    clang \
    cmake \
    ninja-build \
    pkg-config \
    libgtk-3-dev \
    # Android SDK dependencies
    openjdk-17-jdk \
    # Python 3.13 dependencies
    software-properties-common \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3 (Debian has 3.11 which is sufficient)
RUN apt-get update && \
    apt-get install -y python3 python3-dev python3-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/

# Install Claude Code CLI from npm
RUN npm install -g @anthropic-ai/claude-code

# Install Flutter SDK with world-readable permissions
RUN git clone https://github.com/flutter/flutter.git -b stable ${FLUTTER_HOME} && \
    ${FLUTTER_HOME}/bin/flutter --version && \
    ${FLUTTER_HOME}/bin/flutter config --enable-web --no-analytics && \
    ${FLUTTER_HOME}/bin/flutter precache && \
    chmod -R a+rX ${FLUTTER_HOME}

# Install Android SDK
RUN mkdir -p ${ANDROID_HOME} && \
    cd ${ANDROID_HOME} && \
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-9477386_latest.zip -O cmdline-tools.zip && \
    unzip -q cmdline-tools.zip && \
    rm cmdline-tools.zip && \
    mkdir -p cmdline-tools/latest && \
    mv cmdline-tools/* cmdline-tools/latest/ 2>/dev/null || true && \
    yes | ${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager --licenses >/dev/null 2>&1 || true && \
    ${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager "platform-tools" "platforms;android-33" "build-tools;33.0.0"

# Create non-root user
RUN useradd -m -s /bin/bash claude && \
    echo "claude ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Claude configuration will be mounted at runtime from host
# This ensures configuration is always up-to-date and not baked into the image

# Set up working directory
WORKDIR /app

# Copy project files
COPY --chown=claude:claude README.md ./
COPY --chown=claude:claude pyproject.toml uv.lock ./
COPY --chown=claude:claude claude_code_agent_farm.py ./
COPY --chown=claude:claude view_agents.sh ./
COPY --chown=claude:claude configs/ ./configs/
COPY --chown=claude:claude prompts/ ./prompts/
COPY --chown=claude:claude best_practices_guides/ ./best_practices_guides/
COPY --chown=claude:claude tool_setup_scripts/ ./tool_setup_scripts/

# Switch to non-root user
USER claude

# Create virtual environment and install dependencies
RUN python3 -m venv /home/claude/.venv && \
    /home/claude/.venv/bin/pip install --upgrade pip && \
    cd /app && \
    /home/claude/.venv/bin/pip install -e .

# Set up shell environment for claude user
RUN echo 'export PATH="/home/claude/.venv/bin:${PATH}"' >> /home/claude/.bashrc && \
    echo 'export PATH="${FLUTTER_HOME}/bin:${PATH}"' >> /home/claude/.bashrc && \
    echo 'export PATH="${ANDROID_HOME}/cmdline-tools/latest/bin:${PATH}"' >> /home/claude/.bashrc && \
    echo 'export PATH="${ANDROID_HOME}/platform-tools:${PATH}"' >> /home/claude/.bashrc && \
    echo 'export ANDROID_HOME="${ANDROID_HOME}"' >> /home/claude/.bashrc && \
    echo 'alias cc="ENABLE_BACKGROUND_TASKS=1 claude --dangerously-skip-permissions"' >> /home/claude/.bashrc && \
    echo '# cd to project directory is handled by entrypoint' >> /home/claude/.bashrc

# Create workspace directory for mounting projects
RUN mkdir -p /home/claude/workspace

# Copy entrypoint script (as root to allow dynamic user switching)
USER root
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy cc wrapper script to system bin
COPY cc-wrapper.sh /usr/local/bin/cc
RUN chmod +x /usr/local/bin/cc

# Working directory will be set dynamically by entrypoint

# Environment variables that can be overridden
ENV PROMPT_FILE=""
ENV PROMPT_TEXT=""
ENV CONFIG_FILE="/app/configs/flutter_config.json"
ENV AGENTS="1"
ENV AUTO_RESTART="true"

# Skip Claude permission prompts
ENV CLAUDE_DANGEROUSLY_SKIP_PERMISSIONS=1

# Expose any ports if needed (Flutter web runs on 8080 by default)
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command
CMD ["--help"]