# Agent Monitoring Reliability Improvements

This document outlines the improvements made to ensure reliable agent monitoring, particularly for handling restarts and usage limits.

## Overview

The agent monitoring system has been enhanced with several key features to improve reliability and resilience when dealing with usage limits and agent restarts.

## Key Improvements

### 1. Enhanced Usage Limit Detection

**File**: `src/claude_code_agent_farm/constants.py`

- Expanded `USAGE_LIMIT_INDICATORS` from 9 to 28 patterns
- Added more variations including:
  - "you've hit your rate limit"
  - "please try again at"
  - "come back at"
  - "retry after"
  - "available again at"
  - "usage will reset at"
  - "limit will reset"
  - "try again tomorrow"
  - And many more...

This ensures better detection of usage limit messages across different formats.

### 2. Exponential Backoff for Usage Limits

**Files**: 
- `src/claude_code_agent_farm/models/retry_strategy.py` (new)
- `src/claude_code_agent_farm/flutter_agent_monitor.py` (updated)

Implemented intelligent retry strategy with:
- Exponential backoff starting at 60 seconds
- Maximum delay of 3600 seconds (1 hour)
- Configurable backoff factor (default: 2.0)
- Random jitter (±10%) to prevent thundering herd
- Maximum retry attempts (default: 10)
- Retry window tracking (24 hours)

### 3. Fallback Retry Logic

**File**: `src/claude_code_agent_farm/models/retry_strategy.py`

The `UsageLimitRetryInfo` class provides:
- Attempts to parse retry time from message
- Falls back to exponential backoff if parsing fails
- Falls back if parsed time is in the past
- Always provides a valid retry time

### 4. Health Checks for Restart Reliability

**Files**:
- `src/claude_code_agent_farm/models/health_check.py` (new)
- `src/claude_code_agent_farm/flutter_agent_monitor.py` (updated)

Comprehensive health checking system:
- Pre-restart checks:
  - Tmux session existence
  - Disk space availability
- Post-restart checks:
  - Agent responsiveness
  - Tmux pane alive status
- Health status tracking (HEALTHY, DEGRADED, UNHEALTHY)
- Restart success/failure recording

### 5. Restart Attempt Limits with Cooldown

**File**: `src/claude_code_agent_farm/models/health_check.py`

The `RestartAttemptTracker` provides:
- Maximum restart attempts (default: 5)
- Cooldown period between restarts (default: 300 seconds)
- Attempt window tracking (default: 1 hour)
- Success rate calculation
- Automatic cleanup of old attempts

### 6. Watchdog Timer for Hung Agent Detection

**Files**:
- `src/claude_code_agent_farm/models/watchdog.py` (new)
- `src/claude_code_agent_farm/flutter_agent_monitor.py` (updated)

Advanced watchdog system:
- Configurable timeout (default: 5 minutes)
- Grace period after restart (default: 1 minute)
- Activity tracking (pane output, status changes, progress)
- Content change detection via hashing
- Stuck pattern detection (e.g., "Thinking..." repeated)
- Automatic restart on timeout

### 7. Session State Checkpointing

**Files**:
- `src/claude_code_agent_farm/models/checkpoint.py` (new)
- `src/claude_code_agent_farm/flutter_agent_monitor.py` (updated)

Checkpoint and recovery system:
- Automatic checkpoints every 5 minutes
- Saves complete session state:
  - Counters (runs, restarts, usage limits)
  - Retry strategy state
  - Health monitor state
  - Watchdog state
  - Last pane content
- Recovery strategy with:
  - Auto-recovery option
  - Checkpoint age validation
  - Selective state restoration
- Old checkpoint cleanup

### 8. Comprehensive Test Coverage

**New test files**:
- `tests/unit/test_retry_strategy.py` - Tests exponential backoff and fallback logic
- `tests/unit/test_health_check.py` - Tests health checks and restart tracking
- `tests/unit/test_watchdog.py` - Tests watchdog timer and hung detection
- `tests/unit/test_checkpoint.py` - Tests checkpointing and recovery

## Integration Example

The monitor now handles usage limits and restarts more reliably:

```python
# When a usage limit is detected:
1. Enhanced pattern matching identifies the limit
2. Time parser attempts to extract retry time
3. If parsing fails, exponential backoff calculates retry time
4. Retry attempt is recorded with backoff factor applied
5. Monitor waits until retry time
6. Health checks ensure system is ready
7. Restart is attempted with pre/post checks
8. Watchdog monitors for hung state
9. Checkpoint saves state periodically
10. Recovery restores state if needed
```

## Benefits

1. **Better Usage Limit Handling**: More patterns detected, intelligent retry timing
2. **Reliable Restarts**: Health checks prevent failed restarts, cooldown prevents rapid cycling
3. **Hung Detection**: Watchdog automatically recovers from stuck states
4. **State Persistence**: Checkpoints allow recovery from crashes
5. **Observability**: Comprehensive tracking of all events and states

## Configuration

All features are configurable through the monitor initialization:

```python
monitor = FlutterAgentMonitor(settings)
# Automatically includes:
# - RetryStrategy with exponential backoff
# - RestartAttemptTracker with limits
# - WatchdogTimer for hung detection
# - CheckpointManager for state persistence
# - AgentHealthMonitor for health tracking
```

## Monitoring Dashboard

The status display now shows:
- Current retry attempt count
- Time until next retry (with exponential backoff)
- Restart success rate
- Watchdog state
- Health status
- Last checkpoint time

This ensures operators have full visibility into the agent's state and can trust it will handle limits and restarts reliably.