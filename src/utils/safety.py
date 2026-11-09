"""Safety controls and path assertions for EEF_analyze."""

from __future__ import annotations

import os
from pathlib import Path

# External raw dataset path - strictly read-only
READONLY_INPUT_PREFIX = Path("J:/2026二康").resolve()
READONLY_INPUT_STR = str(READONLY_INPUT_PREFIX).lower().replace("/", "\\")

# Permitted write root
ALLOWED_WRITE_ROOT = Path("D:/Python_File/EEF_analyze").resolve()
ALLOWED_WRITE_STR = str(ALLOWED_WRITE_ROOT).lower().replace("/", "\\")


def assert_readonly_path(target_path: str | Path) -> None:
    """Ensure that the given target path is strictly treated as read-only.

    Raises PermissionError if a write operation is attempted to J:\\2026二康.
    """
    resolved = Path(target_path).resolve()
    resolved_str = str(resolved).lower().replace("/", "\\")
    if resolved_str.startswith(READONLY_INPUT_STR):
        raise PermissionError(
            f"CRITICAL SAFETY VIOLATION: Write operation blocked! '{resolved}' is inside strictly read-only external storage."
        )


def assert_safe_write_path(target_path: str | Path, allow_scratch: bool = True) -> Path:
    """Ensure target path resolves within the allowed project directory.

    Returns the resolved Path if safe.
    """
    assert_readonly_path(target_path)
    resolved = Path(target_path).resolve()
    resolved_str = str(resolved).lower().replace("/", "\\")
    if not (resolved_str.startswith(ALLOWED_WRITE_STR) or (allow_scratch and "tmp" in resolved_str)):
        raise PermissionError(
            f"SAFETY ALERT: Output path '{resolved}' is outside permitted workspace '{ALLOWED_WRITE_ROOT}'."
        )
    return resolved


def verify_cwd_safe() -> None:
    """Verify that current working directory is safe for execution."""
    cwd = Path.cwd().resolve()
    cwd_str = str(cwd).lower().replace("/", "\\")
    if cwd_str.startswith(READONLY_INPUT_STR):
        raise RuntimeError(
            f"DANGEROUS EXECUTION CWD: cwd is inside strictly read-only external storage '{cwd}'. Switch cwd to '{ALLOWED_WRITE_ROOT}' before running."
        )
