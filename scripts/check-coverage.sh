#!/bin/bash
# Check Carenji Test Coverage

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default threshold
THRESHOLD=${COVERAGE_THRESHOLD:-80}

echo -e "${BLUE}📊 Checking Carenji Test Coverage${NC}"
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

# Check if coverage data exists
if [ ! -f "coverage/lcov.info" ]; then
    echo -e "${YELLOW}No coverage data found. Running tests...${NC}"
    flutter test --coverage
fi

# Parse coverage data
if [ -f "coverage/lcov.info" ]; then
    # Use lcov if available
    if command -v lcov &> /dev/null; then
        echo -e "${BLUE}Coverage Summary:${NC}"
        echo ""
        lcov --summary coverage/lcov.info
        
        COVERAGE=$(lcov --summary coverage/lcov.info 2>&1 | grep -E "lines.*:" | sed 's/.*: \([0-9.]*\)%.*/\1/')
        
        echo ""
        echo -e "Overall coverage: ${YELLOW}${COVERAGE}%${NC}"
        echo -e "Required threshold: ${YELLOW}${THRESHOLD}%${NC}"
        
        if (( $(echo "$COVERAGE >= $THRESHOLD" | bc -l) )); then
            echo -e "${GREEN}✓ Coverage meets requirements${NC}"
            exit 0
        else
            echo -e "${RED}✗ Coverage below threshold${NC}"
            exit 1
        fi
    else
        # Fallback: Basic parsing
        echo -e "${BLUE}Analyzing coverage data...${NC}"
        
        # Count lines
        TOTAL_LINES=$(grep -c "^DA:" coverage/lcov.info || true)
        COVERED_LINES=$(grep "^DA:" coverage/lcov.info | grep -c ",1" || true)
        
        if [ $TOTAL_LINES -gt 0 ]; then
            COVERAGE=$(echo "scale=2; $COVERED_LINES * 100 / $TOTAL_LINES" | bc)
            echo -e "Lines covered: $COVERED_LINES / $TOTAL_LINES"
            echo -e "Coverage: ${YELLOW}${COVERAGE}%${NC}"
            
            if (( $(echo "$COVERAGE >= $THRESHOLD" | bc -l) )); then
                echo -e "${GREEN}✓ Coverage meets requirements${NC}"
                exit 0
            else
                echo -e "${RED}✗ Coverage below threshold${NC}"
                exit 1
            fi
        else
            echo -e "${RED}Error: Unable to calculate coverage${NC}"
            exit 1
        fi
    fi
else
    echo -e "${RED}Error: No coverage data found${NC}"
    exit 1
fi