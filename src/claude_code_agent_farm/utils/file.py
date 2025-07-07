"""File operation utilities."""

from pathlib import Path

from rich.console import Console

console = Console(stderr=True)


def line_count(file_path: Path) -> int:
    """Count lines in a file with robust encoding handling.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Number of lines in the file, or 0 if error
    """
    try:
        # Try UTF-8 first, then fall back to latin-1
        try:
            with file_path.open("r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except UnicodeDecodeError:
            # Try common encodings
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    with file_path.open("r", encoding=encoding) as f:
                        return sum(1 for _ in f)
                except UnicodeDecodeError:
                    continue
            # Last resort: ignore errors
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not count lines in {file_path}: {e}[/yellow]")
        return 0