"""Resolve explicitly configured local input and output paths."""
from pathlib import Path


def resolve_project_path(root: str | Path, relative_path: str | Path) -> Path:
    """Resolve explicit user input paths or bounded root-relative paths."""

    root_path = Path(root).resolve()
    requested = Path(relative_path)
    if requested.is_absolute():
        return requested.resolve()

    resolved = (root_path / requested).resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"Project path escapes project root: {relative_path}") from exc
    return resolved

