# Bash completion for Claude Flutter Firebase Agent

_cffa_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Main commands
    opts="run status stop logs attach test build docker help"
    
    case "${prev}" in
        cffa|claude-flutter-agent)
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            return 0
            ;;
        run)
            local run_opts="--prompt --project --max-runs --monitor-interval"
            COMPREPLY=( $(compgen -W "${run_opts}" -- ${cur}) )
            return 0
            ;;
        test)
            local test_opts="unit integration e2e docker carenji firebase quick coverage"
            COMPREPLY=( $(compgen -W "${test_opts}" -- ${cur}) )
            return 0
            ;;
        logs)
            local log_opts="--follow --tail --since"
            COMPREPLY=( $(compgen -W "${log_opts}" -- ${cur}) )
            return 0
            ;;
        --project)
            # Complete with directories
            COMPREPLY=( $(compgen -d -- ${cur}) )
            return 0
            ;;
    esac
    
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
}

# Register completions
complete -F _cffa_completion cffa
complete -F _cffa_completion claude-flutter-agent

# Completion for test command
_cffa_test_completion() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local opts="all unit integration e2e docker carenji firebase quick coverage"
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
}

complete -F _cffa_test_completion cffa-test
