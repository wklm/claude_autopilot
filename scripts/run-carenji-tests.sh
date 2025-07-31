#!/bin/bash
# Run Carenji Tests with Coverage Report

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 Running Carenji Tests${NC}"
echo ""

# Check if running in Docker
if [ -f "/.dockerenv" ]; then
    CARENJI_PATH="/workspace"
else
    # Check if CARENJI_PATH is set
    if [ -z "$CARENJI_PATH" ]; then
        SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
        POSSIBLE_PATH="$SCRIPT_DIR/../../carenji"
        
        if [ -d "$POSSIBLE_PATH" ] && [ -f "$POSSIBLE_PATH/pubspec.yaml" ]; then
            export CARENJI_PATH="$POSSIBLE_PATH"
        else
            echo -e "${RED}Error: CARENJI_PATH not set${NC}"
            exit 1
        fi
    fi
fi

cd "$CARENJI_PATH"

# Run Flutter analyzer first
echo -e "${BLUE}Running Flutter analyzer...${NC}"
flutter analyze
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ No analyzer issues found${NC}"
else
    echo -e "${RED}✗ Analyzer found issues${NC}"
    exit 1
fi

echo ""

# Run tests with coverage
echo -e "${BLUE}Running tests with coverage...${NC}"
flutter test --coverage

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi

echo ""

# Generate coverage report
if [ -f "coverage/lcov.info" ]; then
    echo -e "${BLUE}Generating coverage report...${NC}"
    
    # Calculate coverage percentage
    if command -v lcov &> /dev/null; then
        COVERAGE=$(lcov --summary coverage/lcov.info 2>&1 | grep -E "lines.*:" | sed 's/.*: \([0-9.]*\)%.*/\1/')
        
        echo -e "Overall coverage: ${YELLOW}${COVERAGE}%${NC}"
        
        # Check if coverage meets threshold
        THRESHOLD=80
        if (( $(echo "$COVERAGE >= $THRESHOLD" | bc -l) )); then
            echo -e "${GREEN}✓ Coverage meets ${THRESHOLD}% threshold${NC}"
        else
            echo -e "${RED}✗ Coverage below ${THRESHOLD}% threshold${NC}"
        fi
    fi
    
    # Generate HTML report if genhtml is available
    if command -v genhtml &> /dev/null; then
        echo -e "${BLUE}Generating HTML coverage report...${NC}"
        genhtml coverage/lcov.info -o coverage/html --quiet
        echo -e "${GREEN}✓ HTML report generated at coverage/html/index.html${NC}"
    fi
fi

echo ""
echo -e "${GREEN}Test run complete!${NC}"