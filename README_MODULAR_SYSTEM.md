# Claude Code Agent Farm - Modular System

The Claude Code Agent Farm has been updated to support multiple technology stacks and configurable problem generation commands.

## Tech Stack Support

The system now supports different technology stacks out of the box:

- **Next.js** (TypeScript) - Using `bun`, `eslint`, and `tsc`
- **Python** - Using `mypy` and `ruff` (with optional `uv` package manager)

## Configuration System

### Using Configuration Files

Create a JSON configuration file to customize the agent farm for your project:

```bash
python claude_code_agent_farm.py --path /path/to/project --config configs/python_config.json
```

### Configuration Options

#### Tech Stack Configuration

```json
{
  "tech_stack": "python",  // or "nextjs"
  "problem_commands": {
    "type_check": ["mypy", "."],
    "lint": ["ruff", "check", "."]
  }
}
```

#### Best Practices Guides

The system can automatically copy specific best practices guide files to your project:

```json
{
  "best_practices_files": [
    "./best_practices_guides/NEXTJS15_BEST_PRACTICES.md",
    "./best_practices_guides/TYPESCRIPT_BEST_PRACTICES.md"
  ]
}
```

Specify individual file paths, and they'll be copied to `{project}/best_practices_guides/` when the farm starts.

#### Full Configuration Example

```json
{
  "comment": "Configuration for Python project with uv",
  "tech_stack": "python",
  "problem_commands": {
    "type_check": ["uv", "run", "mypy", "src", "--strict"],
    "lint": ["uv", "run", "ruff", "check", "src", "tests"]
  },
  "best_practices_files": ["./best_practices_guides/PYTHON_BEST_PRACTICES.md"],
  "agents": 20,
  "session": "claude_agents",
  "stagger": 10.0,
  "wait_after_cc": 15.0,
  "check_interval": 10,
  "skip_regenerate": false,
  "skip_commit": false,
  "auto_restart": true,
  "no_monitor": false,
  "attach": false,
  "prompt_file": "prompts/default_prompt_python.txt",
  "context_threshold": 20,
  "idle_timeout": 60,
  "max_errors": 3,
  "git_branch": "main",
  "git_remote": "origin",
  "tmux_kill_on_exit": true,
  "tmux_mouse": true,
  "fast_start": false,
  "full_backup": false,
  "max_agents": 50
}
```

## Prompt System

### Default Prompts

The system looks for prompts in this order:

1. Explicitly specified prompt file (`--prompt-file`)
2. Tech-stack specific default: `prompts/default_prompt_{tech_stack}.txt`
3. Generic default: `prompts/default_prompt.txt`

### Custom Prompts

Create tech-stack specific prompts that reference the appropriate tools and best practices:

```bash
# For Python projects
python claude_code_agent_farm.py --path /path/to/project --prompt-file prompts/python_async_prompt.txt

# For Next.js projects  
python claude_code_agent_farm.py --path /path/to/project --prompt-file prompts/nextjs_app_router_prompt.txt
```

## Examples

### Python Project with uv

```bash
python claude_code_agent_farm.py \
  --path /path/to/python/project \
  --config configs/python_uv_config.json \
  --agents 15
```

### Next.js Project

```bash
python claude_code_agent_farm.py \
  --path /path/to/nextjs/project \
  --config configs/nextjs_config.json \
  --agents 20
```

### Custom Tech Stack

Create your own configuration for any tech stack:

```json
{
  "tech_stack": "rust",
  "problem_commands": {
    "type_check": ["cargo", "check"],
    "lint": ["cargo", "clippy", "--", "-D", "warnings"]
  },
  "best_practices_files": ["./best_practices_guides/RUST_BEST_PRACTICES.md"],
  "prompt_file": "prompts/default_prompt_rust.txt"
}
```

## Migration from Old System

If you were using the old system with hardcoded Next.js commands:

1. Your existing setup will continue to work (defaults to Next.js)
2. To make it explicit, use `--config configs/nextjs_config.json`
3. Move any custom prompts to the `prompts/` directory

## Problem File Generation

The `combined_typechecker_and_linter_problems.txt` file is generated using the configured commands:

- **type_check**: Command for type checking (e.g., `tsc`, `mypy`)
- **lint**: Command for linting (e.g., `eslint`, `ruff`)

Both commands are optional - you can run just type checking or just linting by omitting the other. 