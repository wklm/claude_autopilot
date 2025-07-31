#!/bin/bash
# Quick installer for bash aliases

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the setup script
"$SCRIPT_DIR/setup_aliases.sh"