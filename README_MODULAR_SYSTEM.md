# Claude Code Agent Farm - Modular System

The Claude Code Agent Farm has been updated to support multiple technology stacks, configurable problem generation commands, and different workflow types (bug fixing vs best practices implementation).

## Tech Stack Support

The system now supports different technology stacks out of the box:

- **Next.js** (TypeScript) - Using `bun`, `eslint`, and `tsc`
- **Python** - Using `mypy` and `ruff` (with optional `uv` package manager)

## Workflow Types

### 1. Bug Fixing Workflow (Traditional)
Agents work through type-checker and linter problems in parallel, fixing issues in batches.

### 2. Best Practices Implementation Workflow
Agents systematically implement modern best practices from a guide, tracking progress in a structured document.

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

#### Chunk Size Configuration

The `chunk_size` parameter controls how many lines/changes agents work on at once:

```json
{
  "chunk_size": 50  // Default is 50, can be adjusted based on task complexity
}
```

This value is substituted into prompts wherever `{chunk_size}` appears.

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
  "chunk_size": 50,
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

### Prompt Types

#### Bug Fixing Prompts
- `default_prompt.txt` - Generic bug fixing
- `default_prompt_nextjs.txt` - Next.js specific bug fixing
- `default_prompt_python.txt` - Python specific bug fixing
- `bug_fixing_prompt_for_nextjs.txt` - Original Next.js bug fixing prompt

#### Best Practices Implementation Prompts
- `default_best_practices_prompt.txt` - Generic best practices implementation
- `default_best_practices_prompt_nextjs.txt` - Next.js best practices implementation
- `default_best_practices_prompt_python.txt` - Python best practices implementation
- `continue_best_practices_prompt.txt` - For continuing existing progress

### Custom Prompts

Create tech-stack specific prompts that reference the appropriate tools and best practices:

```bash
# For bug fixing
python claude_code_agent_farm.py --path /path/to/project --config configs/nextjs_config.json

# For best practices implementation
python claude_code_agent_farm.py --path /path/to/project --config configs/nextjs_best_practices_config.json
```

### Variable Substitution

Prompts support variable substitution:
- `{chunk_size}` - Replaced with the configured chunk size (default: 50)

## Examples

### Bug Fixing Workflow

#### Python Project with uv

```bash
python claude_code_agent_farm.py \
  --path /path/to/python/project \
  --config configs/python_uv_config.json \
  --agents 15
```

#### Next.js Project

```bash
python claude_code_agent_farm.py \
  --path /path/to/nextjs/project \
  --config configs/nextjs_config.json \
  --agents 20
```

### Best Practices Implementation Workflow

#### Next.js Best Practices

```bash
# First, ensure you have NEXTJS15_BEST_PRACTICES.md in best_practices_guides/
python claude_code_agent_farm.py \
  --path /path/to/nextjs/project \
  --config configs/nextjs_best_practices_config.json \
  --agents 10
```

This will:
1. Have agents read the best practices guide
2. Create a progress tracking document
3. Systematically implement the practices
4. Track completion status accurately

#### Continuing Best Practices Work

```bash
python claude_code_agent_farm.py \
  --path /path/to/project \
  --prompt-file prompts/continue_best_practices_prompt.txt \
  --config configs/nextjs_best_practices_config.json \
  --agents 10
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