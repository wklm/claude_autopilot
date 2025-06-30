# The Definitive Guide to Modern Bash/Zsh Scripting for Ubuntu (mid-2025 Edition)

This guide synthesizes cutting-edge shell scripting practices for Ubuntu 25.04 LTS and beyond, moving past outdated patterns to establish production-grade standards for modern infrastructure automation, DevOps workflows, and system administration.

## Prerequisites & Modern Shell Environment

This guide assumes **Ubuntu 24.10+** with **Bash 5.3+** or **Zsh 5.9+**. Ubuntu 25.04 LTS ships with Bash 5.3 by default, supporting new features like `BASH_ARGC_MAX` for improved function introspection and native JSON parsing via loadable builtins.

### Initial Setup: Choose Your Shell Wisely

```bash
# Check your current shell and version
echo $SHELL && $SHELL --version

# For Bash users - enable modern features globally
cat >> ~/.bashrc << 'EOF'
# Enable extended globbing and modern bash features
shopt -s extglob globstar nullglob dotglob
shopt -s inherit_errexit errtrace
set -o pipefail  # Critical for production scripts

# Load JSON builtin (Bash 5.3+)
enable -f /usr/lib/bash/json json
EOF

# For Zsh users - enable recommended options
cat >> ~/.zshrc << 'EOF'
# Modern Zsh options for scripting
setopt EXTENDED_GLOB NULL_GLOB DOT_GLOB
setopt PIPE_FAIL ERR_RETURN ERR_EXIT
setopt NO_UNSET WARN_CREATE_GLOBAL
EOF
```

### ✅ DO: Use a Proper Shebang with Explicit Flags

```bash
#!/usr/bin/env -S bash -euo pipefail
# -e: exit on error
# -u: exit on undefined variable
# -o pipefail: exit on pipe failure

# For scripts requiring Bash 5.3+ features
#!/usr/bin/env -S bash --version-check=5.3 -euo pipefail
```

### ❌ DON'T: Use Legacy Shebangs

```bash
# Bad - hardcoded path
#!/bin/bash

# Worse - no error handling
#!/bin/sh

# Worst - assumes bash is in PATH without env
#!bash
```

## 1. Modern Script Architecture

### Project Structure for Production Scripts

```
/project
├── bin/                    # User-facing scripts
│   ├── deploy              # Main entry point
│   └── rollback           
├── lib/                    # Shared libraries
│   ├── common.sh          # Common functions
│   ├── logging.sh         # Structured logging
│   └── validation.sh      # Input validation
├── config/                # Configuration
│   ├── defaults.conf      # Default values
│   └── production.conf    # Environment-specific
├── tests/                 # Test suite
│   ├── unit/             
│   └── integration/      
├── .shellcheckrc         # ShellCheck configuration
└── .gitlab-ci.yml        # CI/CD pipeline
```

### ✅ DO: Create Modular, Testable Functions

```bash
# lib/logging.sh - Structured logging with journald integration
declare -r LOG_LEVELS=(DEBUG INFO WARN ERROR FATAL)
declare -gi LOG_LEVEL=${LOG_LEVEL:-1}  # Default to INFO

log() {
    local level=$1; shift
    local message="$*"
    local level_index
    
    # Get numeric level
    for i in "${!LOG_LEVELS[@]}"; do
        [[ "${LOG_LEVELS[$i]}" == "$level" ]] && level_index=$i && break
    done
    
    # Skip if below threshold
    (( level_index < LOG_LEVEL )) && return 0
    
    # Structured output for systemd journal
    printf '{"timestamp":"%s","level":"%s","message":"%s","script":"%s","pid":%d}\n' \
        "$(date -Iseconds)" \
        "$level" \
        "$message" \
        "${BASH_SOURCE[-1]##*/}" \
        "$$" | systemd-cat -t "${0##*/}" -p "${level,,}"
}

# Usage
log INFO "Deployment started"
log ERROR "Failed to connect to database"
```

### ❌ DON'T: Write Monolithic Scripts

```bash
# Bad - 500+ lines in a single file with no functions
#!/bin/bash
# deploy.sh - everything in one place
echo "Starting deployment..."
cd /app
git pull
npm install
npm build
# ... 500 more lines
```

## 2. Error Handling: Beyond `set -e`

### ✅ DO: Implement Comprehensive Error Handling

```bash
#!/usr/bin/env -S bash -eEuo pipefail

# Global error handler with stack trace
error_handler() {
    local line_no=$1
    local bash_lineno=$2
    local last_command=$3
    local code=$4
    
    # Structured error output
    cat >&2 <<-EOF
	{
	  "error": true,
	  "timestamp": "$(date -Iseconds)",
	  "line": $line_no,
	  "bash_line": $bash_lineno,
	  "command": "$last_command",
	  "exit_code": $code,
	  "stack_trace": [
	    $(printf '"%s",' "${BASH_SOURCE[@]}" | sed 's/,$//')
	  ],
	  "function_stack": [
	    $(printf '"%s",' "${FUNCNAME[@]}" | sed 's/,$//')
	  ]
	}
	EOF
    
    # Cleanup on error
    cleanup_on_error
    exit "$code"
}

# Install error handler
trap 'error_handler ${LINENO} ${BASH_LINENO} "$BASH_COMMAND" $?' ERR

# Cleanup function
cleanup_on_error() {
    # Remove temporary files
    [[ -n "${TEMP_DIR:-}" ]] && rm -rf "$TEMP_DIR"
    
    # Release locks
    [[ -n "${LOCK_FD:-}" ]] && flock -u "$LOCK_FD"
    
    # Notify monitoring
    send_alert "Script failed: ${0##*/}"
}
```

### Advanced Error Recovery Patterns

```bash
# Retry with exponential backoff and jitter
retry_with_backoff() {
    local max_attempts=${1:-5}
    local timeout=${2:-1}
    local attempt=0
    local exitCode=0
    
    shift 2  # Remove first two arguments
    
    while (( attempt < max_attempts )); do
        if "$@"; then
            return 0
        else
            exitCode=$?
        fi
        
        attempt=$(( attempt + 1 ))
        local delay=$(( timeout * (2 ** (attempt - 1)) ))
        local jitter=$(( RANDOM % delay ))
        
        log WARN "Command failed (attempt $attempt/$max_attempts), retrying in ${delay}s (±${jitter}s)..."
        sleep $(( delay + jitter ))
        timeout=$(( timeout * 2 ))
    done
    
    log ERROR "Command failed after $max_attempts attempts"
    return $exitCode
}

# Usage
retry_with_backoff 5 2 curl -fsSL "https://api.example.com/health"
```

## 3. Modern Input Validation and Parsing

### ✅ DO: Use Proper Argument Parsing

```bash
# Modern argument parsing with validation
parse_args() {
    # Default values
    declare -g ACTION=""
    declare -g ENVIRONMENT="staging"
    declare -g DRY_RUN=false
    declare -g VERBOSE=false
    declare -ga TAGS=()
    
    # GNU-style long options
    local -r SHORT_OPTS="hve:t:n"
    local -r LONG_OPTS="help,verbose,env:,environment:,tag:,dry-run"
    
    # Parse options
    if ! PARSED=$(getopt --options="$SHORT_OPTS" --longoptions="$LONG_OPTS" --name "$0" -- "$@"); then
        usage >&2
        exit 2
    fi
    
    eval set -- "$PARSED"
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                LOG_LEVEL=0  # DEBUG
                shift
                ;;
            -e|--env|--environment)
                ENVIRONMENT="$2"
                validate_environment "$ENVIRONMENT"
                shift 2
                ;;
            -t|--tag)
                TAGS+=("$2")
                shift 2
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            --)
                shift
                break
                ;;
            *)
                log ERROR "Unexpected argument: $1"
                usage >&2
                exit 2
                ;;
        esac
    done
    
    # Remaining arguments
    if [[ $# -eq 0 ]]; then
        log ERROR "No action specified"
        usage >&2
        exit 2
    fi
    
    ACTION="$1"
    validate_action "$ACTION"
}

# Input validation with clear error messages
validate_environment() {
    local env=$1
    local -r VALID_ENVS=(development staging production)
    
    if [[ ! " ${VALID_ENVS[*]} " =~ " ${env} " ]]; then
        log ERROR "Invalid environment: $env"
        log ERROR "Valid environments: ${VALID_ENVS[*]}"
        exit 2
    fi
    
    # Additional validation for production
    if [[ "$env" == "production" ]] && [[ "$USER" != "deploy" ]]; then
        log ERROR "Production deployments must be run as 'deploy' user"
        exit 1
    fi
}
```

### ❌ DON'T: Use Positional Parameters Without Validation

```bash
# Bad - no validation, unclear parameters
#!/bin/bash
ENVIRONMENT=$1
ACTION=$2
TAG=$3

# No validation, will fail mysteriously later
deploy_to_$ENVIRONMENT
```

## 4. Secure Coding Practices

### ✅ DO: Sanitize All External Input

```bash
# Secure command execution with input validation
execute_safe() {
    local cmd=$1; shift
    local -a safe_args=()
    
    # Validate command against allowlist
    local -r ALLOWED_COMMANDS=(git docker kubectl helm terraform)
    if [[ ! " ${ALLOWED_COMMANDS[*]} " =~ " ${cmd} " ]]; then
        log ERROR "Command not in allowlist: $cmd"
        return 1
    fi
    
    # Sanitize arguments
    for arg in "$@"; do
        # Remove potentially dangerous characters
        if [[ "$arg" =~ ^[a-zA-Z0-9._/=-]+$ ]]; then
            safe_args+=("$arg")
        else
            log ERROR "Unsafe argument rejected: $arg"
            return 1
        fi
    done
    
    # Execute with timeout and resource limits
    timeout --kill-after=10s 300s \
        nice -n 10 \
        "$cmd" "${safe_args[@]}"
}

# Usage
execute_safe git checkout main
execute_safe kubectl get pods -n production
```

### ✅ DO: Use Secure Temporary Files

```bash
# Create secure temporary directory
create_temp_dir() {
    # Use mktemp with secure permissions
    TEMP_DIR=$(mktemp -d -t "${0##*/}.XXXXXXXX") || {
        log ERROR "Failed to create temporary directory"
        exit 1
    }
    
    # Ensure cleanup on exit
    trap 'rm -rf "$TEMP_DIR"' EXIT INT TERM
    
    # Set restrictive permissions
    chmod 700 "$TEMP_DIR"
    
    export TEMP_DIR
    log DEBUG "Created temporary directory: $TEMP_DIR"
}

# Secure file operations
secure_write() {
    local file=$1
    local content=$2
    
    # Create with restricted permissions
    (umask 077; echo "$content" > "$file")
    
    # Verify permissions
    if [[ "$(stat -c %a "$file")" != "600" ]]; then
        log ERROR "Failed to set secure permissions on $file"
        rm -f "$file"
        return 1
    fi
}
```

## 5. Modern JSON Handling

### ✅ DO: Use Native JSON Support (Bash 5.3+)

```bash
#!/usr/bin/env -S bash -euo pipefail

# Enable JSON builtin
enable -f /usr/lib/bash/json json

# Parse JSON response
parse_api_response() {
    local response=$1
    local -A data
    
    # Parse JSON into associative array
    json -a data <<< "$response"
    
    # Access parsed data
    echo "Status: ${data[status]}"
    echo "Message: ${data[message]}"
    
    # Handle nested objects
    json -a nested <<< "${data[details]}"
    echo "Error code: ${nested[code]}"
}

# Generate JSON
generate_payload() {
    local action=$1
    local timestamp=$(date -Iseconds)
    
    # Build JSON using native builtin
    json \
        action="$action" \
        timestamp="$timestamp" \
        host="$(hostname -f)" \
        user="$USER" \
        metadata="$(json \
            version="1.0.0" \
            environment="$ENVIRONMENT"
        )"
}
```

### Fallback for Older Systems: jq Integration

```bash
# Robust jq wrapper with error handling
json_query() {
    local json=$1
    local query=$2
    local default=${3:-null}
    
    if ! command -v jq &>/dev/null; then
        log ERROR "jq is required but not installed"
        return 1
    fi
    
    # Query with error handling
    if result=$(echo "$json" | jq -r "$query" 2>/dev/null); then
        echo "$result"
    else
        echo "$default"
        return 1
    fi
}

# Type-safe JSON extraction
extract_typed() {
    local json=$1
    local path=$2
    local type=$3
    local default=$4
    
    local value
    value=$(json_query "$json" "$path" "$default")
    
    # Validate type
    case "$type" in
        string)
            [[ "$value" =~ ^[^0-9]*$ ]] || return 1
            ;;
        number)
            [[ "$value" =~ ^-?[0-9]+(\.[0-9]+)?$ ]] || return 1
            ;;
        boolean)
            [[ "$value" =~ ^(true|false)$ ]] || return 1
            ;;
        array)
            [[ "$value" =~ ^\[.*\]$ ]] || return 1
            ;;
    esac
    
    echo "$value"
}
```

## 6. Parallel Processing and Performance

### ✅ DO: Use Modern Parallel Execution

```bash
# Parallel execution with GNU Parallel (installed by default in Ubuntu 25)
parallel_process() {
    local -r MAX_JOBS=${PARALLEL_JOBS:-$(nproc)}
    local -r JOB_LOG_DIR="$TEMP_DIR/jobs"
    
    mkdir -p "$JOB_LOG_DIR"
    
    # Process with progress bar and job logging
    parallel \
        --jobs "$MAX_JOBS" \
        --bar \
        --joblog "$JOB_LOG_DIR/joblog" \
        --results "$JOB_LOG_DIR/results" \
        --retries 3 \
        --timeout 300 \
        --memfree 1G \
        process_single_item {} ::: "${ITEMS[@]}"
    
    # Check results
    if [[ -f "$JOB_LOG_DIR/joblog" ]]; then
        local failed
        failed=$(awk '$7 != 0' "$JOB_LOG_DIR/joblog" | wc -l)
        if (( failed > 0 )); then
            log ERROR "$failed jobs failed"
            return 1
        fi
    fi
}

# Alternative: Native bash job control for simpler cases
parallel_simple() {
    local -r max_jobs=${1:-4}
    shift
    
    local -i active_jobs=0
    local -a pids=()
    
    for item in "$@"; do
        # Wait if at job limit
        while (( active_jobs >= max_jobs )); do
            wait -n  # Wait for any job to complete
            active_jobs=$((active_jobs - 1))
        done
        
        # Launch job in background
        process_item "$item" &
        pids+=($!)
        active_jobs=$((active_jobs + 1))
    done
    
    # Wait for remaining jobs
    local exit_code=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            exit_code=1
            log ERROR "Job $pid failed"
        fi
    done
    
    return $exit_code
}
```

### Performance Monitoring

```bash
# Built-in profiling with PS4
profile_script() {
    # Enhanced PS4 for timing information
    export PS4='+ $(date "+%s.%N") ${BASH_SOURCE}:${LINENO}: ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
    
    # Enable tracing
    set -x
    
    # Run the profiled code
    "$@"
    
    # Disable tracing
    set +x
}

# Memory usage tracking
track_memory() {
    local -r pid=$$
    local -r interval=${1:-1}
    
    # Background monitor
    (
        while kill -0 "$pid" 2>/dev/null; do
            ps -o pid,vsz,rss,comm -p "$pid" >> "$TEMP_DIR/memory.log"
            sleep "$interval"
        done
    ) &
    
    local monitor_pid=$!
    
    # Ensure cleanup
    trap "kill $monitor_pid 2>/dev/null" EXIT
}
```

## 7. Modern Testing Framework

### ✅ DO: Use BATS for Testing

```bash
# tests/unit/logging.bats
#!/usr/bin/env bats

setup() {
    load '../test_helper'
    source "$PROJECT_ROOT/lib/logging.sh"
}

@test "log INFO writes to journal" {
    run log INFO "Test message"
    assert_success
    assert_output --partial "INFO"
    assert_output --partial "Test message"
}

@test "log respects LOG_LEVEL" {
    LOG_LEVEL=2  # WARN and above
    run log INFO "Should not appear"
    assert_success
    refute_output
}

@test "log handles special characters" {
    run log ERROR 'Message with "quotes" and $variables'
    assert_success
    assert_output --partial 'Message with "quotes" and $variables'
}
```

### Integration Testing

```bash
# tests/integration/deployment.bats
#!/usr/bin/env bats

setup() {
    # Create test environment
    export TEST_ENV=$(mktemp -d)
    export ENVIRONMENT="test"
    
    # Mock external commands
    function kubectl() {
        echo "kubectl $*" >> "$TEST_ENV/kubectl.log"
        
        case "$1" in
            get)
                echo "NAME  READY  STATUS"
                echo "app   1/1    Running"
                ;;
            rollout)
                return 0
                ;;
            *)
                return 1
                ;;
        esac
    }
    export -f kubectl
}

teardown() {
    rm -rf "$TEST_ENV"
}

@test "deployment creates kubernetes resources" {
    run "$PROJECT_ROOT/bin/deploy" app
    assert_success
    
    # Verify kubectl was called correctly
    assert_file_contains "$TEST_ENV/kubectl.log" "apply -f"
    assert_file_contains "$TEST_ENV/kubectl.log" "rollout status"
}
```

## 8. Container and Cloud Native Integration

### ✅ DO: Build Container-Aware Scripts

```bash
# Detect if running in container
is_container() {
    # Multiple detection methods for reliability
    if [[ -f /.dockerenv ]] || 
       [[ -f /run/.containerenv ]] ||
       grep -q -E '(docker|lxc|containerd|podman)' /proc/1/cgroup 2>/dev/null; then
        return 0
    fi
    return 1
}

# Adapt behavior for containers
configure_for_environment() {
    if is_container; then
        # Container-specific settings
        export LOG_FORMAT="json"
        export COLOR_OUTPUT="false"
        
        # Write to stdout/stderr for container logs
        exec 1> >(systemd-cat -t "${0##*/}" -p info)
        exec 2> >(systemd-cat -t "${0##*/}" -p err)
    else
        # Traditional server settings
        export LOG_FORMAT="text"
        export COLOR_OUTPUT="true"
    fi
}

# Kubernetes-aware health checks
k8s_health_check() {
    local -r HEALTH_FILE="/tmp/healthy"
    local -r READY_FILE="/tmp/ready"
    
    # Liveness probe
    touch "$HEALTH_FILE"
    
    # Readiness probe
    if all_services_ready; then
        touch "$READY_FILE"
    else
        rm -f "$READY_FILE"
    fi
}
```

### Cloud Provider Integration

```bash
# AWS SSM Parameter Store integration
get_secret() {
    local param_name=$1
    local region=${AWS_REGION:-us-east-1}
    
    # Cache secrets to reduce API calls
    local cache_file="$TEMP_DIR/.secrets_cache"
    local cache_key
    cache_key=$(echo -n "$param_name" | sha256sum | cut -d' ' -f1)
    
    # Check cache (5 minute TTL)
    if [[ -f "$cache_file.$cache_key" ]]; then
        local age
        age=$(( $(date +%s) - $(stat -c %Y "$cache_file.$cache_key") ))
        if (( age < 300 )); then
            cat "$cache_file.$cache_key"
            return 0
        fi
    fi
    
    # Fetch from SSM
    local value
    if value=$(aws ssm get-parameter \
        --name "$param_name" \
        --with-decryption \
        --query 'Parameter.Value' \
        --output text \
        --region "$region" 2>/dev/null); then
        
        # Cache the result
        echo "$value" > "$cache_file.$cache_key"
        chmod 600 "$cache_file.$cache_key"
        
        echo "$value"
    else
        log ERROR "Failed to retrieve secret: $param_name"
        return 1
    fi
}
```

## 9. Observability and Monitoring

### ✅ DO: Implement Structured Logging and Metrics

```bash
# OpenTelemetry integration for traces
send_trace() {
    local operation=$1
    local duration=$2
    local status=$3
    local -A attributes=()
    
    # Parse additional attributes
    shift 3
    while [[ $# -gt 0 ]]; do
        attributes[$1]=$2
        shift 2
    done
    
    # Send to OTLP collector
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$(json \
            resourceSpans="[$(json \
                resource="$(json \
                    attributes="[
                        $(json key="service.name" value="${SERVICE_NAME:-script}")
                        $(json key="host.name" value="$(hostname)")
                    ]"
                )" \
                scopeSpans="[$(json \
                    spans="[$(json \
                        name="$operation" \
                        kind=2 \
                        startTimeUnixNano="$(date +%s%N)" \
                        endTimeUnixNano="$(($(date +%s%N) + duration * 1000000))" \
                        status="$(json code="$([[ $status == "ok" ]] && echo 1 || echo 2)")"
                    )]"
                )]"
            )]"
        )" \
        "${OTEL_EXPORTER_OTLP_ENDPOINT:-http://localhost:4318}/v1/traces" &
}

# Prometheus metrics
send_metric() {
    local metric_name=$1
    local value=$2
    local type=${3:-gauge}
    local help=${4:-""}
    local -A labels=()
    
    # Parse labels
    shift 4
    while [[ $# -gt 0 ]]; do
        labels[$1]=$2
        shift 2
    done
    
    # Format labels
    local label_str=""
    for key in "${!labels[@]}"; do
        label_str+=",${key}=\"${labels[$key]}\""
    done
    label_str=${label_str#,}  # Remove leading comma
    
    # Send to pushgateway
    cat <<EOF | curl -s --data-binary @- "${PROMETHEUS_PUSHGATEWAY:-http://localhost:9091}/metrics/job/${JOB_NAME:-script}"
# HELP $metric_name $help
# TYPE $metric_name $type
${metric_name}{${label_str}} $value
EOF
}

# Usage
time_operation() {
    local operation=$1; shift
    local start_time=$(date +%s.%N)
    
    # Run the operation
    local exit_code=0
    "$@" || exit_code=$?
    
    # Calculate duration
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    
    # Send metrics
    send_trace "$operation" "$duration" "$([[ $exit_code -eq 0 ]] && echo "ok" || echo "error")" \
        "exit_code" "$exit_code"
    
    send_metric "script_operation_duration_seconds" "$duration" "histogram" \
        "Duration of script operations" \
        "operation" "$operation" \
        "status" "$([[ $exit_code -eq 0 ]] && echo "success" || echo "failure")"
    
    return $exit_code
}

# Wrap operations
time_operation "database_backup" perform_backup
time_operation "file_sync" rsync -avz /source/ /dest/
```

## 10. Advanced Shell Features

### ✅ DO: Use Modern Bash Features

```bash
# Nameref for dynamic variable references (Bash 4.3+)
dynamic_config() {
    local env=$1
    local -n config="CONFIG_${env^^}"  # Nameref to CONFIG_STAGING, etc.
    
    echo "Database: ${config[database]}"
    echo "API URL: ${config[api_url]}"
}

# Associative arrays for configuration
declare -A CONFIG_STAGING=(
    [database]="staging.db.example.com"
    [api_url]="https://api-staging.example.com"
)

declare -A CONFIG_PRODUCTION=(
    [database]="prod.db.example.com"
    [api_url]="https://api.example.com"
)

# Usage
dynamic_config staging
dynamic_config production

# BASH_ARGV/BASH_ARGC for advanced debugging (Bash 5.3+)
debug_function_calls() {
    # Enable extended debugging
    set -o functrace
    shopt -s extdebug
    
    trap 'debug_info' DEBUG
}

debug_info() {
    local func="${FUNCNAME[1]}"
    local line="${BASH_LINENO[0]}"
    local args_count="${BASH_ARGC[1]}"
    
    if [[ -n "$func" ]] && [[ "$func" != "main" ]]; then
        log DEBUG "Calling $func at line $line with $args_count arguments"
    fi
}
```

### Coprocesses for Bidirectional Communication

```bash
# Advanced coprocess usage for database connections
maintain_db_connection() {
    # Start psql as a coprocess
    coproc PSQL {
        psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -A -t 2>&1
    }
    
    # Function to execute queries
    db_query() {
        local query=$1
        
        # Send query to coprocess
        echo "$query;" >&${PSQL[1]}
        
        # Read results
        local result
        local output=""
        while IFS= read -r -t 1 -u ${PSQL[0]} result; do
            output+="$result"$'\n'
        done
        
        echo "${output%$'\n'}"  # Remove trailing newline
    }
    
    # Keep connection alive
    trap 'echo "\\q" >&${PSQL[1]}; wait $PSQL_PID' EXIT
    
    # Usage
    db_query "SELECT COUNT(*) FROM users"
    db_query "SELECT name FROM users WHERE active = true LIMIT 5"
}
```

## 11. Systemd Integration

### ✅ DO: Create SystemD-Aware Scripts

```bash
# Notify systemd of service status
systemd_notify() {
    local state=$1
    
    # Only if running under systemd
    if [[ -n "${NOTIFY_SOCKET:-}" ]]; then
        systemd-notify "$state"
    fi
}

# Main service loop
run_service() {
    # Tell systemd we're ready
    systemd_notify "READY=1"
    systemd_notify "STATUS=Processing requests"
    
    # Update watchdog
    local watchdog_usec
    watchdog_usec=$(systemctl show -p WatchdogUSec --value "$SERVICE_NAME")
    
    if [[ "$watchdog_usec" != "0" ]]; then
        # Pet the watchdog at half the interval
        local watchdog_interval=$((${watchdog_usec%us} / 2000000))
        
        while true; do
            systemd_notify "WATCHDOG=1"
            process_work || systemd_notify "STATUS=Error processing work"
            sleep "$watchdog_interval"
        done
    else
        # No watchdog, just run
        while true; do
            process_work
            sleep 60
        done
    fi
}

# Generate systemd service file
generate_service_file() {
    cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=${SERVICE_DESCRIPTION:-Script Service}
After=network-online.target
Wants=network-online.target

[Service]
Type=notify
ExecStart=${SCRIPT_PATH}
Restart=on-failure
RestartSec=30s
WatchdogSec=120s

# Security hardening
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
NoNewPrivileges=yes
ReadWritePaths=/var/log/${SERVICE_NAME}

# Resource limits
MemoryLimit=512M
CPUQuota=50%

# Environment
Environment="LOG_LEVEL=INFO"
EnvironmentFile=-/etc/default/${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable "${SERVICE_NAME}.service"
}
```

## 12. Security Hardening

### ✅ DO: Implement Defense in Depth

```bash
# AppArmor integration for script confinement
generate_apparmor_profile() {
    local script_path=$1
    local profile_name
    profile_name=$(basename "$script_path" | tr '.' '_')
    
    cat > "/etc/apparmor.d/usr.local.bin.$profile_name" <<EOF
#include <tunables/global>

profile $profile_name /usr/local/bin/$profile_name {
  #include <abstractions/base>
  #include <abstractions/bash>
  
  # Allow reading config
  /etc/${profile_name}/** r,
  
  # Allow writing logs
  /var/log/${profile_name}/** w,
  
  # Allow network access (customize as needed)
  network inet stream,
  network inet dgram,
  
  # Temp files
  /tmp/${profile_name}.* rw,
  
  # Deny everything else
  deny /** w,
  deny @{HOME}/** rwx,
}
EOF

    apparmor_parser -r "/etc/apparmor.d/usr.local.bin.$profile_name"
}

# Seccomp filtering for system calls
apply_seccomp_filter() {
    # Use systemd-run with seccomp filter
    systemd-run \
        --scope \
        --property="SystemCallFilter=@system-service" \
        --property="SystemCallFilter=~@privileged @resources" \
        --property="NoNewPrivileges=yes" \
        "$@"
}

# Capability dropping
drop_capabilities() {
    # Drop all capabilities except the ones needed
    capsh \
        --drop=all \
        --caps="cap_net_raw+p" \
        --user="$SERVICE_USER" \
        -- -c "$*"
}
```

## 13. Modern Debugging Techniques

### ✅ DO: Use Advanced Debugging Tools

```bash
# Enhanced debugging with bash debugger
debug_mode() {
    # Check if running under bashdb
    if [[ -n "${BASHDB_DEBUGGER:-}" ]]; then
        echo "Running under bashdb"
        return 0
    fi
    
    # Enhanced debugging output
    export PS4='+ ${BASH_SOURCE}:${LINENO}:${FUNCNAME[0]:+${FUNCNAME[0]}():} '
    
    # Enable debugging features
    set -o functrace
    set -o errtrace
    shopt -s extdebug
    
    # Trap for interactive debugging
    trap 'debug_prompt $? "$BASH_COMMAND"' DEBUG
}

debug_prompt() {
    local last_status=$1
    local last_command=$2
    
    # Skip if not in debug mode
    [[ "${DEBUG_INTERACTIVE:-false}" == "false" ]] && return 0
    
    # Show context
    echo "---"
    echo "Command: $last_command"
    echo "Status: $last_status"
    echo "Location: ${BASH_SOURCE[1]}:${BASH_LINENO[0]}"
    echo "Function: ${FUNCNAME[1]:-main}"
    
    # Interactive prompt
    read -p "Debug> " -r debug_cmd
    case "$debug_cmd" in
        c|continue)
            return 0
            ;;
        v|vars)
            ( set -o posix ; set ) | less
            ;;
        s|stack)
            for i in "${!FUNCNAME[@]}"; do
                echo "$i: ${FUNCNAME[$i]} (${BASH_SOURCE[$i+1]}:${BASH_LINENO[$i]})"
            done
            ;;
        q|quit)
            exit 1
            ;;
        *)
            eval "$debug_cmd"
            ;;
    esac
    
    # Recurse for next command
    debug_prompt 0 ""
}

# Execution tracing with timing
trace_execution() {
    exec 3>&2 2> >(
        while IFS= read -r line; do
            echo "$(date '+%Y-%m-%d %H:%M:%S.%N') $line" >&3
        done
    )
    set -x
    "$@"
    set +x
}
```

## 14. Package Management and Distribution

### ✅ DO: Create Proper Debian Packages

```bash
# Modern .deb package creation
create_deb_package() {
    local name=$1
    local version=$2
    local arch=${3:-all}
    
    # Create package structure
    local pkg_dir="$TEMP_DIR/${name}_${version}_${arch}"
    mkdir -p "$pkg_dir"/{DEBIAN,usr/local/bin,etc,usr/share/doc/$name}
    
    # Copy files
    cp "$PROJECT_ROOT/bin/"* "$pkg_dir/usr/local/bin/"
    cp "$PROJECT_ROOT/config/defaults.conf" "$pkg_dir/etc/$name/"
    
    # Create control file
    cat > "$pkg_dir/DEBIAN/control" <<EOF
Package: $name
Version: $version
Architecture: $arch
Maintainer: $MAINTAINER_EMAIL
Depends: bash (>= 5.3), jq, parallel, curl
Section: utils
Priority: optional
Homepage: $PROJECT_URL
Description: $PROJECT_DESCRIPTION
 $(echo "$PROJECT_LONG_DESCRIPTION" | fold -s -w 76 | sed '2,$s/^/ /')
EOF

    # Create postinst script
    cat > "$pkg_dir/DEBIAN/postinst" <<'EOF'
#!/bin/bash
set -e

case "$1" in
    configure)
        # Create service user
        if ! getent passwd scriptrunner >/dev/null; then
            adduser --system --group --no-create-home scriptrunner
        fi
        
        # Set up logging directory
        mkdir -p /var/log/${name}
        chown scriptrunner:scriptrunner /var/log/${name}
        
        # Install systemd service
        systemctl daemon-reload
        ;;
esac

#DEBHELPER#
exit 0
EOF
    chmod 755 "$pkg_dir/DEBIAN/postinst"
    
    # Build package
    dpkg-deb --build --root-owner-group "$pkg_dir"
    
    # Lint the package
    lintian "${pkg_dir}.deb"
}

# APT repository integration
publish_to_apt() {
    local deb_file=$1
    local repo_path="/var/www/apt"
    
    # Sign package
    dpkg-sig --sign builder "$deb_file"
    
    # Add to repository
    reprepro -b "$repo_path" includedeb stable "$deb_file"
    
    # Update repository metadata
    reprepro -b "$repo_path" export
}
```

## 15. Zsh-Specific Advanced Features

### ✅ DO: Leverage Zsh's Unique Capabilities

```zsh
#!/usr/bin/env zsh
# Zsh-specific advanced patterns

# Extended globbing with qualifiers
cleanup_old_logs() {
    local days=${1:-30}
    
    # Delete files older than $days, larger than 100MB
    rm -f /var/log/**/*.log(.mtime+$days,L+100M)
    
    # Archive files between 7-30 days old
    tar -czf "logs_$(date +%Y%m%d).tar.gz" /var/log/**/*.log(.mtime+7,mtime-30)
}

# Advanced parameter expansion
safe_variable_expansion() {
    local input=$1
    
    # Nested expansion with defaults
    : ${DEPLOY_ENV:=${ENVIRONMENT:-${ENV:-development}}}
    
    # Array slicing and filtering
    local services=(web api worker database cache)
    local active_services=(${(M)services:#${~ACTIVE_PATTERN}})
    
    # Unique array elements
    local all_hosts=(${(u)${(f)"$(cat hosts.txt)"}})
}

# Zsh modules for system programming
load_system_modules() {
    zmodload zsh/system
    zmodload zsh/net/tcp
    zmodload zsh/stat
    
    # Direct TCP connections
    ztcp localhost 80
    local fd=$REPLY
    
    # Send HTTP request
    print -u $fd "GET / HTTP/1.0\r\n\r\n"
    
    # Read response
    while IFS= read -u $fd -r line; do
        echo "Received: $line"
    done
    
    # Close connection
    ztcp -c $fd
}

# Function profiling built-in
profile_functions() {
    zmodload zsh/zprof
    
    # Run code to profile
    complex_operation
    
    # Show profiling results
    zprof
}
```

## Production Deployment Checklist

### Pre-Deployment Validation

```bash
#!/usr/bin/env -S bash -euo pipefail

validate_script() {
    local script=$1
    local errors=0
    
    # ShellCheck validation
    if ! shellcheck -x -S error "$script"; then
        log ERROR "ShellCheck validation failed"
        ((errors++))
    fi
    
    # Check for required error handling
    if ! grep -q "set -.*e" "$script" || ! grep -q "pipefail" "$script"; then
        log ERROR "Missing error handling configuration"
        ((errors++))
    fi
    
    # Verify no hardcoded secrets
    if grep -E "(password|secret|key)\\s*=" "$script" | grep -v "^\\s*#"; then
        log ERROR "Potential hardcoded secrets found"
        ((errors++))
    fi
    
    # Check for proper cleanup handlers
    if ! grep -q "trap.*EXIT" "$script"; then
        log WARN "No cleanup trap found"
    fi
    
    # Validate dependencies
    local deps
    deps=$(grep -o "command -v [a-zA-Z0-9_-]*" "$script" | awk '{print $3}' | sort -u)
    
    for cmd in $deps; do
        if ! command -v "$cmd" &>/dev/null; then
            log ERROR "Missing dependency: $cmd"
            ((errors++))
        fi
    done
    
    return $errors
}

# Integration with CI/CD
generate_ci_pipeline() {
    cat > .gitlab-ci.yml <<'EOF'
stages:
  - validate
  - test
  - build
  - deploy

variables:
  DEBIAN_FRONTEND: noninteractive

validate:
  stage: validate
  image: koalaman/shellcheck-alpine:latest
  script:
    - shellcheck -x bin/* lib/*.sh
    
test:unit:
  stage: test
  image: bats/bats:latest
  script:
    - bats tests/unit/*.bats
  coverage: '/Total coverage: \d+\.\d+%/'
    
test:integration:
  stage: test
  image: ubuntu:25.04
  services:
    - postgres:15
    - redis:7
  script:
    - apt-get update && apt-get install -y bats
    - bats tests/integration/*.bats
    
build:package:
  stage: build
  image: ubuntu:25.04
  script:
    - ./scripts/build-package.sh
  artifacts:
    paths:
      - "*.deb"
    expire_in: 1 week
    
deploy:staging:
  stage: deploy
  script:
    - ./scripts/deploy.sh staging
  environment:
    name: staging
  only:
    - main
    
deploy:production:
  stage: deploy
  script:
    - ./scripts/deploy.sh production
  environment:
    name: production
  when: manual
  only:
    - tags
EOF
}
```

## Conclusion

This guide represents the state of the art in shell scripting for Ubuntu systems as of mid-2025. The patterns and practices outlined here will help you create maintainable, secure, and performant scripts that integrate seamlessly with modern cloud-native infrastructure.

Remember: **Always validate input, handle errors explicitly, and test thoroughly**. The shell is powerful but unforgiving—respect it, and it will serve you well.

For updates and community discussion, visit the project repository and join our forum where shell scripting professionals share advanced techniques and real-world solutions.