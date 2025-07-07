"""Shell command execution utilities."""

import shlex
import subprocess
from typing import List, Tuple, Union

from rich.console import Console

console = Console(stderr=True)


def run(cmd: str, *, check: bool = True, quiet: bool = False, capture: bool = False) -> Tuple[int, str, str]:
    """Execute shell command with optional output capture.

    When capture=False, output is streamed to terminal unless quiet=True
    When capture=True, output is captured and returned
    
    Args:
        cmd: Shell command to execute
        check: Raise exception on non-zero exit code
        quiet: Suppress command logging and output
        capture: Capture output instead of streaming
        
    Returns:
        Tuple of (returncode, stdout, stderr)
    """
    if not quiet:
        console.log(cmd, style="cyan")

    # Parse command for shell safety when possible
    cmd_arg: Union[str, List[str]]
    try:
        # Try to parse as a list of arguments for safer execution
        cmd_list = shlex.split(cmd)
        use_shell = False
        cmd_arg = cmd_list
    except ValueError:
        # Fall back to shell=True for complex commands with pipes, redirects, etc.
        cmd_list = []  # Not used when shell=True
        use_shell = True
        cmd_arg = cmd

    if capture:
        result = subprocess.run(cmd_arg, shell=use_shell, capture_output=True, text=True, check=check)
        return result.returncode, result.stdout or "", result.stderr or ""
    else:
        # Stream output to terminal when not capturing
        # Preserve stderr even in quiet-mode so that exceptions contain detail
        if quiet:
            result = subprocess.run(cmd_arg, shell=use_shell, capture_output=True, text=True, check=check)
            return result.returncode, result.stdout or "", result.stderr or ""
        stdout_pipe = None
        stderr_pipe = subprocess.STDOUT
        try:
            result = subprocess.run(
                cmd_arg, shell=use_shell, check=check, stdout=stdout_pipe, stderr=stderr_pipe, text=True
            )
            return result.returncode, "", ""
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Command failed with exit code {e.returncode}: {cmd}[/red]")
            raise