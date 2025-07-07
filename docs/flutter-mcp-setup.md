# Flutter MCP Integration for Claude Code Agent Farm

This guide explains how to use the Flutter MCP (Model Context Protocol) server with Claude Code Agent Farm to provide real-time Flutter and Dart documentation to your agents.

## Seamless Integration (NEW)

As of the latest update, MCP integration is now **automatic and seamless**:

- **Auto-detection**: MCP automatically starts for Flutter projects (detected by `pubspec.yaml`)
- **Lifecycle management**: MCP server starts with the first agent and stops when the last agent is removed
- **Zero configuration**: No manual setup required for Flutter projects
- **Override options**: Use `--mcp` to force enable or `--no-mcp` to disable

## Overview

Flutter MCP is a documentation server that provides AI assistants with up-to-date, version-specific documentation for Flutter and Dart packages from pub.dev. By integrating it with Claude Code Agent Farm, all your agents can access Flutter documentation without making external API calls.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Docker Network                      │
│                                                      │
│  ┌─────────────────┐      ┌────────────────────┐   │
│  │ Flutter MCP     │      │  Agent Container 1  │   │
│  │ Server          │◄────►│  (Claude Code)      │   │
│  │ (Port 8000)     │      └────────────────────┘   │
│  └─────────────────┘                                │
│          ▲                 ┌────────────────────┐   │
│          │                 │  Agent Container 2  │   │
│          └────────────────►│  (Claude Code)      │   │
│                           └────────────────────┘   │
│                                    ...              │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### Automatic Mode (Recommended)

For Flutter projects, MCP starts automatically:

```bash
# In your Flutter project directory (with pubspec.yaml)
./run-docker.sh "Add user authentication"

# MCP server starts automatically!
```

### Manual Control

Override automatic behavior:

```bash
# Force MCP for non-Flutter projects
./run-docker.sh --mcp "Work with Flutter docs"

# Disable MCP for Flutter projects
./run-docker.sh --no-mcp "Work without docs"
```

### Background Mode with Multiple Agents

```bash
# Start 5 agents with automatic MCP
./run-docker.sh -b 5 "Fix all Flutter issues"

# View agents in tmux grid
./ccfarm-agents.sh

# Stop all agents and MCP server
./ccfarm-stopremove.sh
```

## Manual Setup with Docker Compose

If you prefer to use Docker Compose directly:

```bash
cd docker
docker-compose up -d

# Scale to multiple agents
docker-compose up -d --scale ccfarm-agent=5
```

## Health Checks

Check if the MCP server is running properly:

```bash
./scripts/check-mcp-health.sh
```

This will verify:
- Container status
- Network connectivity
- HTTP endpoint availability
- MCP tools accessibility

## How It Works

1. **MCP Server**: A dedicated container runs the Flutter MCP server, exposing it on port 8000
2. **Agent Configuration**: Each agent container is configured with MCP environment variables
3. **Network Communication**: All containers share a Docker network for internal communication
4. **Claude Integration**: The `cc-wrapper.sh` script configures Claude to use the MCP server

## MCP Tools Available

The Flutter MCP server provides two main tools to Claude:

### flutter_search
Universal package and documentation search. Examples:
- "Search for state management packages"
- "Find packages for HTTP requests"
- "Look for animation libraries"

### flutter_docs
Smart documentation retrieval. Examples:
- "Get docs for Provider package"
- "Show me the API for http package version 0.13.5"
- "Explain the Navigator widget"

## Troubleshooting

### MCP Server Not Starting

Check Docker logs:
```bash
docker logs flutter-mcp-server
```

### Agents Can't Connect to MCP

Verify network connectivity:
```bash
docker network inspect claude-mcp-network
```

### Permission Issues

Ensure proper permissions on config files:
```bash
chmod 644 ~/.claude.json
chmod -R 755 ~/.claude/
```

## Environment Variables

### MCP Configuration
- `MCP_ENABLED`: Set to "true" to enable MCP
- `MCP_SERVER_URL`: URL of the MCP server (default: http://flutter-mcp:8000)

### Agent Configuration
- `PROJECT_PATH`: Path to your Flutter project
- `CONFIG_FILE`: Path to agent configuration file
- `PROMPT_FILE`: Path to prompt file
- `PROMPT_TEXT`: Direct prompt text
- `AGENTS`: Number of agents to run
- `AUTO_RESTART`: Enable auto-restart on failure

## Advanced Usage

### Custom MCP Server Port

To run the MCP server on a different port, modify the Dockerfile:

```dockerfile
ENV MCP_PORT=9000
EXPOSE 9000
CMD ["flutter-mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "9000"]
```

### Adding More MCP Servers

You can add additional MCP servers by:
1. Creating new Dockerfile in `docker/` directory
2. Adding service to `docker-compose.yml`
3. Configuring agents with multiple MCP endpoints

## Best Practices

1. **Resource Management**: The MCP server is lightweight but caches documentation. Monitor memory usage for large projects.

2. **Network Security**: The MCP server is only accessible within the Docker network by default. Don't expose port 8000 unless necessary.

3. **Agent Coordination**: While agents share the MCP server, they work independently. Use the todo system for task coordination.

4. **Error Handling**: Agents will continue working even if MCP is unavailable, but without Flutter documentation access.

## Example Workflow

1. Start MCP server and agents:
```bash
./scripts/run-agents-with-mcp.sh --agents 5 --path ~/my-flutter-app \
  --prompt-text "Add a new feature for user authentication with Firebase"
```

2. Monitor progress:
```bash
./ccfarm-agents.sh
```

3. Check individual agent logs:
```bash
docker logs ccfarm-mcp-1
```

4. Stop all agents:
```bash
docker stop $(docker ps -q --filter 'name=ccfarm-mcp-')