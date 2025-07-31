#!/bin/bash
# Quick Task Runner for Carenji Development

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Show usage
usage() {
    echo "Usage: $0 <task-type> [custom-prompt]"
    echo ""
    echo "Task types:"
    echo "  fix-errors    - Fix all Flutter analyzer errors"
    echo "  implement     - Implement a new feature"
    echo "  test          - Write tests for existing code"
    echo "  review        - Review and optimize code"
    echo "  custom        - Run with custom prompt"
    echo ""
    echo "Examples:"
    echo "  $0 fix-errors"
    echo "  $0 implement 'Add medication reminder notifications'"
    echo "  $0 test 'Add tests for vitals monitoring'"
    echo "  $0 custom 'Your custom development task'"
    exit 1
}

# Check arguments
if [ $# -eq 0 ]; then
    usage
fi

TASK_TYPE=$1
CUSTOM_PROMPT=$2

# Set prompt based on task type
case $TASK_TYPE in
    fix-errors)
        PROMPT="Fix all Flutter analyzer errors and warnings in the carenji app following the coding standards in CLAUDE.md"
        ;;
    implement)
        if [ -z "$CUSTOM_PROMPT" ]; then
            echo -e "${RED}Error: Please specify what to implement${NC}"
            echo "Example: $0 implement 'medication reminder notifications'"
            exit 1
        fi
        PROMPT="Implement $CUSTOM_PROMPT in the carenji app following clean architecture with proper models, repositories, services, and ViewModels. Include comprehensive tests."
        ;;
    test)
        if [ -z "$CUSTOM_PROMPT" ]; then
            echo -e "${RED}Error: Please specify what to test${NC}"
            echo "Example: $0 test 'vitals monitoring feature'"
            exit 1
        fi
        PROMPT="Write comprehensive tests for $CUSTOM_PROMPT in the carenji app to achieve at least 80% coverage"
        ;;
    review)
        if [ -z "$CUSTOM_PROMPT" ]; then
            PROMPT="Review the carenji codebase for performance optimization opportunities and adherence to Flutter best practices"
        else
            PROMPT="Review and optimize $CUSTOM_PROMPT in the carenji app"
        fi
        ;;
    custom)
        if [ -z "$CUSTOM_PROMPT" ]; then
            echo -e "${RED}Error: Please provide a custom prompt${NC}"
            exit 1
        fi
        PROMPT="$CUSTOM_PROMPT"
        ;;
    *)
        echo -e "${RED}Error: Unknown task type '$TASK_TYPE'${NC}"
        usage
        ;;
esac

echo -e "${BLUE}🦋 Running Carenji Development Task${NC}"
echo -e "Task: ${YELLOW}$TASK_TYPE${NC}"
echo -e "Prompt: ${GREEN}$PROMPT${NC}"
echo ""

# Check if running locally or in Docker
if command -v claude-flutter-agent &> /dev/null; then
    # Running locally
    echo -e "${BLUE}Starting agent locally...${NC}"
    claude-flutter-agent run --prompt-text "$PROMPT"
else
    # Try Docker
    if docker ps | grep -q "claude-carenji-agent"; then
        echo -e "${BLUE}Using existing Docker container...${NC}"
        docker exec -it claude-carenji-agent claude-flutter-agent run --prompt-text "$PROMPT"
    else
        echo -e "${YELLOW}Starting Docker container...${NC}"
        export CLAUDE_PROMPT_TEXT="$PROMPT"
        docker-compose up claude-carenji-agent
    fi
fi