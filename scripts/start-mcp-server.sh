#!/bin/bash

# Script to start the Flutter MCP server container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Starting Flutter MCP Server..."

# Build the Flutter MCP image
echo "Building Flutter MCP Docker image..."
docker build -t flutter-mcp:latest "$PROJECT_ROOT/docker/flutter-mcp"

# Check if container is already running
if docker ps --format '{{.Names}}' | grep -q '^flutter-mcp-server$'; then
    echo "Flutter MCP server is already running"
    echo "To restart, run: docker restart flutter-mcp-server"
    exit 0
fi

# Remove existing container if it exists but is stopped
if docker ps -a --format '{{.Names}}' | grep -q '^flutter-mcp-server$'; then
    echo "Removing existing stopped container..."
    docker rm flutter-mcp-server
fi

# Run the MCP server
docker run -d \
    --name flutter-mcp-server \
    --restart unless-stopped \
    --network claude-mcp-network \
    -p 8000:8000 \
    flutter-mcp:latest

# Wait for server to be ready
echo "Waiting for MCP server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "Flutter MCP server is ready!"
        echo "Server is running at: http://localhost:8000"
        exit 0
    fi
    sleep 1
done

echo "Warning: MCP server may not be fully ready yet"
echo "Check logs with: docker logs flutter-mcp-server"