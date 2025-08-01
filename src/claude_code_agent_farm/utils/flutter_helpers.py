"""Flutter and Firebase helper functions."""

from pathlib import Path

from rich.console import Console

from claude_code_agent_farm.utils.shell import run

console = Console(stderr=True)


def check_flutter_project(path: Path) -> bool:
    """Check if the given path is a Flutter project."""
    pubspec = path / "pubspec.yaml"
    if not pubspec.exists():
        return False

    # Check for Flutter in pubspec
    try:
        content = pubspec.read_text()
        return "flutter:" in content and "sdk: flutter" in content
    except Exception:
        return False


def check_firebase_project(path: Path) -> bool:
    """Check if the given path has Firebase configuration."""
    firebase_json = path / "firebase.json"
    return firebase_json.exists()


def check_carenji_project(path: Path) -> bool:
    """Check if the given path is the carenji project."""
    if not check_flutter_project(path):
        return False

    # Check for carenji-specific files
    pubspec = path / "pubspec.yaml"
    try:
        content = pubspec.read_text()
        return "name: carenji" in content
    except Exception:
        return False


def get_firebase_emulator_status() -> dict:
    """Check status of Firebase emulators."""
    from claude_code_agent_farm.constants import CARENJI_FIREBASE_EMULATOR_PORTS

    status = {}
    for service, port in CARENJI_FIREBASE_EMULATOR_PORTS.items():
        try:
            result = run(f"curl -s http://localhost:{port}", check=False, capture_output=True)
            status[service] = result is not None and result.returncode == 0
        except Exception:
            status[service] = False

    return status


def start_firebase_emulators(project_path: Path) -> bool:
    """Start Firebase emulators for the project."""
    # Check if docker-compose.emulators.yml exists
    docker_compose = project_path / "docker-compose.emulators.yml"
    if docker_compose.exists():
        console.print("[cyan]Starting Firebase emulators via Docker...[/cyan]")
        result = run(f"docker-compose -f {docker_compose} up -d", cwd=project_path, check=False)
        return result is not None
    # Try starting with Firebase CLI
    console.print("[cyan]Starting Firebase emulators via Firebase CLI...[/cyan]")
    result = run("firebase emulators:start --only auth,firestore,functions,storage", cwd=project_path, check=False)
    return result is not None


def get_flutter_mcp_command(project_path: Path) -> str:
    """Get the Flutter run command with MCP flags."""
    from claude_code_agent_farm.constants import FLUTTER_RUN_FLAGS

    return f"cd {project_path} && flutter run {' '.join(FLUTTER_RUN_FLAGS)}"


def get_carenji_prompt_template(task_type: str) -> str:
    """Get a prompt template for specific carenji development tasks."""
    from claude_code_agent_farm.constants import CARENJI_PROMPT_TEMPLATES

    return CARENJI_PROMPT_TEMPLATES.get(
        task_type, "Help with carenji Flutter app development following the guidelines in CLAUDE.md",
    )
