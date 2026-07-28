#!/usr/bin/env python3
"""
Script to rename 'quantum_agent' / 'quantum-agent' to 'heraldo' across the entire codebase.

This script performs:
1. Text replacements inside source files, configuration files, and scripts.
2. Directory and file renamings (e.g., renaming the package directory `quantum_agent` to `heraldo`).

Safety Features:
- Restricted strictly to the current working directory and its subfolders.
- Excludes hidden dot folders (e.g., .git, .vscode, .venv).
- Excludes itself from modification and renaming.
"""

import os
from pathlib import Path

# Absolute path of this script to ensure it never modifies or renames itself
SCRIPT_PATH = Path(__file__).resolve()

# List of string replacement pairs (case variants)
REPLACEMENTS = [
    ("quantum_agent", "heraldo"),
    ("quantum-agent", "heraldo"),
    ("QuantumAgent", "Heraldo"),
    ("QUANTUM_AGENT", "HERALDO"),
]

# Standard directories to exclude in addition to any dot-prefixed directories
EXCLUDE_DIRS = {
    "__pycache__",
    "venv",
    "env",
    "build",
    "dist",
}

# File extensions to include for text replacement
FILE_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".rst", ".toml", ".sh", ".bat", ".ini", ".cfg", ".tex"
}


def is_within_root(path: Path, root_dir: Path) -> bool:
    """Guarantees a path is strictly inside root_dir and not above it."""
    try:
        path.resolve().relative_to(root_dir.resolve())
        return True
    except ValueError:
        return False


def should_skip(path: Path) -> bool:
    """
    Checks whether a given relative path or its ancestors should be excluded.
    Excludes any folder starting with '.' (e.g., .git, .vscode, .venv) as well as EXCLUDE_DIRS.
    """
    for part in path.parts:
        if part.startswith(".") or part in EXCLUDE_DIRS or part.endswith(".egg-info"):
            return True
    return False


def replace_text_in_files(root_dir: Path):
    """Replaces target string occurrences in all matching text files."""
    modified_files = 0
    for file_path in root_dir.rglob("*"):
        if not file_path.is_file():
            continue

        # Strict boundary check: ensure file is strictly inside root_dir
        if not is_within_root(file_path, root_dir):
            continue

        # Skip this script itself
        if file_path.resolve() == SCRIPT_PATH:
            continue

        rel_path = file_path.relative_to(root_dir)
        if should_skip(rel_path):
            continue

        if file_path.suffix.lower() not in FILE_EXTENSIONS and file_path.name not in {"Dockerfile", "Makefile"}:
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        new_content = content
        for old, new in REPLACEMENTS:
            new_content = new_content.replace(old, new)

        if new_content != content:
            file_path.write_text(new_content, encoding="utf-8")
            print(f"Updated content: {rel_path}")
            modified_files += 1

    print(f"Subtotal: {modified_files} file(s) updated.")


def rename_paths(root_dir: Path):
    """Renames directories and files matching target names bottom-up."""
    renamed_count = 0
    # Sort paths bottom-up (deepest paths first) so parent directories aren't moved before children
    all_paths = sorted(list(root_dir.rglob("*")), key=lambda p: len(p.parts), reverse=True)

    for path in all_paths:
        # Strict boundary check: ensure path is strictly inside root_dir
        if not is_within_root(path, root_dir):
            continue

        # Skip this script itself
        if path.resolve() == SCRIPT_PATH:
            continue

        rel_path = path.relative_to(root_dir)
        if should_skip(rel_path):
            continue

        name = path.name
        new_name = name
        for old, new in REPLACEMENTS:
            new_name = new_name.replace(old, new)

        if new_name != name:
            new_path = path.parent / new_name
            path.rename(new_path)
            print(f"Renamed path: {rel_path} -> {new_path.relative_to(root_dir)}")
            renamed_count += 1

    print(f"Subtotal: {renamed_count} path(s) renamed.")


def main():
    # Target is strictly the Current Working Directory
    root_dir = Path.cwd().resolve()

    print("=" * 70)
    print(" CODEBASE RENAME: quantum_agent -> heraldo ")
    print("=" * 70)
    print(f"Working Directory (root): {root_dir}")
    print("Parent directories will NOT be scanned or modified.\n")

    print("[Step 1/2] Updating text references inside project files...")
    replace_text_in_files(root_dir)

    print("\n[Step 2/2] Renaming directories and file paths...")
    rename_paths(root_dir)

    print("\n" + "=" * 70)
    print(" Codebase successfully renamed to 'heraldo'! ")
    print("=" * 70)


if __name__ == "__main__":
    main()
