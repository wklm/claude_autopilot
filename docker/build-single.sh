#!/bin/bash
# Build script for Claude Single Agent Monitor

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Building Claude Single Agent Monitor..."
echo "Project root: $PROJECT_ROOT"

# Make entrypoint executable
chmod +x "$SCRIPT_DIR/docker-entrypoint-single.sh"

# Build the Docker image
cd "$PROJECT_ROOT"
docker build -f docker/Dockerfile.single -t claude-single-agent:latest .

echo ""
echo "Build complete!"
echo ""
echo "To run the single agent monitor:"
echo "  PROJECT_DIR=/path/to/your/project docker-compose -f docker/docker-compose-single.yml up"
echo ""
echo "Or with Docker directly:"
echo "  docker run -it --rm \\"
echo "    -v /path/to/your/project:/workspace \\"
echo "    -v ~/.config/claude:/home/claude/.config/claude:ro \\"
echo "    claude-single-agent"