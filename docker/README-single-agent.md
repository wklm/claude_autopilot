# Claude Single Agent Monitor

A simplified, single-agent version of Claude Code Agent Farm that runs one Claude agent in a Docker container with automatic monitoring and respawning.

## Features

- **Single Agent Focus**: Runs exactly one Claude agent in a tmux session
- **Automatic Respawning**: Restarts the agent when it completes, errors, or hits usage limits
- **Usage Limit Handling**: Parses Claude's usage limit messages and waits until the specified retry time
- **Docker Native**: Runs entirely in a Docker container
- **Real-time Monitoring**: Shows agent status, runs, restarts, and usage limit hits

## Quick Start

### 1. Build the Docker Image

```bash
cd /path/to/claude_code_agent_farm
docker build -f docker/Dockerfile.single -t claude-single-agent .
```

### 2. Run with Docker Compose

```bash
# With prompt.txt in your project
PROJECT_DIR=/path/to/your/project docker-compose -f docker/docker-compose-single.yml up

# With inline prompt text
PROJECT_DIR=/path/to/your/project \
PROMPT_TEXT="Fix all type errors and lint issues" \
docker-compose -f docker/docker-compose-single.yml up
```

### 3. Run with Docker CLI

```bash
# Basic usage with prompt.txt in project
docker run -it --rm \
  -v /path/to/your/project:/workspace \
  -v ~/.config/claude:/home/claude/.config/claude:ro \
  claude-single-agent

# With custom prompt text
docker run -it --rm \
  -v /path/to/your/project:/workspace \
  -v ~/.config/claude:/home/claude/.config/claude:ro \
  -e PROMPT_TEXT="Fix all type errors" \
  claude-single-agent

# With external prompt file
docker run -it --rm \
  -v /path/to/your/project:/workspace \
  -v ~/.config/claude:/home/claude/.config/claude:ro \
  -v /path/to/prompt.txt:/prompt.txt:ro \
  -e PROMPT_FILE=/prompt.txt \
  claude-single-agent
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_PATH` | `/workspace` | Project directory path |
| `PROMPT_FILE` | - | Path to prompt file |
| `PROMPT_TEXT` | - | Direct prompt text |
| `WAIT_ON_LIMIT` | `true` | Wait when usage limit is hit |
| `RESTART_ON_COMPLETE` | `true` | Restart when task completes |
| `RESTART_ON_ERROR` | `true` | Restart on errors |
| `CHECK_INTERVAL` | `5` | Seconds between status checks |
| `IDLE_TIMEOUT` | `300` | Seconds before considering agent idle |
| `TMUX_SESSION` | `claude-agent` | tmux session name |

### Prompt Configuration

The monitor accepts prompts in three ways (in order of precedence):

1. **PROMPT_FILE**: Path to a file containing the prompt
2. **PROMPT_TEXT**: Direct prompt text as environment variable
3. **Default**: `prompt.txt` in the project directory

## Usage Limit Handling

The monitor automatically detects and handles Claude usage limits by parsing messages like:

- "Usage limit reached. Try again at 3:00 PM PST"
- "Available after 2:00 PM EST"
- "Usage resets at midnight UTC"
- "Available in 2 hours"

When a usage limit is detected:
1. The monitor extracts the retry time
2. Calculates how long to wait
3. Waits until that time
4. Automatically restarts the agent

## Monitoring

The monitor displays real-time status including:

- Current agent status (Working, Ready, Idle, Error, Usage Limit)
- Total runtime
- Number of runs/restarts
- Usage limit hits
- Wait time remaining (when applicable)

## Attach to Running Session

To see what Claude is doing:

```bash
# Using docker-compose
docker exec -it claude-single-agent-monitor tmux attach-session -t claude-agent

# Using docker run (find container ID first)
docker ps
docker exec -it <container-id> tmux attach-session -t claude-agent

# Detach with: Ctrl+B, then D
```

## Advanced Usage

### Custom Configuration

Create a custom docker-compose override:

```yaml
# docker-compose.override.yml
version: '3.8'

services:
  claude-single-agent:
    environment:
      - CHECK_INTERVAL=10
      - IDLE_TIMEOUT=600
      - WAIT_ON_LIMIT=false
```

### Running Without Waiting on Limits

```bash
PROJECT_DIR=/path/to/project \
WAIT_ON_LIMIT=false \
docker-compose -f docker/docker-compose-single.yml up
```

### Running with One-Time Execution

```bash
PROJECT_DIR=/path/to/project \
RESTART_ON_COMPLETE=false \
RESTART_ON_ERROR=false \
docker-compose -f docker/docker-compose-single.yml up
```

## Troubleshooting

### Claude Configuration Not Found

Make sure to mount your Claude configuration:

```bash
# Modern location
-v ~/.config/claude:/home/claude/.config/claude:ro

# Legacy location
-v ~/.claude.json:/home/claude/.claude.json:ro
-v ~/.claude:/home/claude/.claude:ro
```

### Project Directory Issues

Ensure your project is properly mounted:

```bash
# Correct: Full paths on both sides
-v /home/user/my-project:/workspace

# Incorrect: Relative paths
-v ./my-project:/workspace
```

### View Logs

```bash
# With docker-compose
docker-compose -f docker/docker-compose-single.yml logs -f

# With docker
docker logs -f <container-id>
```

## Development

To run the monitor locally without Docker:

```bash
# Install dependencies
pip install -e .

# Run directly
python -m claude_code_agent_farm.single_agent_cli \
  --project-path /path/to/project \
  --prompt-text "Your prompt here"
```

## Architecture

The single-agent monitor consists of:

1. **DockerAgentMonitor**: Core monitoring class that manages the tmux session
2. **UsageLimitTimeParser**: Parses various time formats from Claude's messages
3. **SingleAgentConfig**: Pydantic model for configuration
4. **AgentSession**: Tracks session state and statistics

The monitor runs a continuous loop that:
1. Checks agent status every N seconds
2. Detects completion, errors, or usage limits
3. Takes appropriate action (restart, wait, etc.)
4. Updates the display with current status

## Differences from Multi-Agent Farm

| Feature | Multi-Agent Farm | Single Agent Monitor |
|---------|-----------------|---------------------|
| Agents | Multiple (1-100) | Single |
| Complexity | High | Low |
| Resource Usage | High | Low |
| Use Case | Parallel tasks | Continuous single task |
| Configuration | Complex | Simple |
| Monitoring | Per-agent status | Single status |