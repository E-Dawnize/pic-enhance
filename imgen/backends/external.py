"""Run an external program and capture its output.

Kept deliberately small: no shell, no pipes beyond stdout/stderr, a hard
timeout, and a decoded log line the caller can print or stash in the report.
"""

from __future__ import annotations

import locale
import os
import subprocess
import time
from pathlib import Path

from ..registry import StageError

__all__ = ["run", "RunResult"]

_ENCODING = locale.getpreferredencoding(False) or "utf-8"


class RunResult:
    __slots__ = ("command", "returncode", "output", "seconds")

    def __init__(self, command: list[str], returncode: int, output: str, seconds: float):
        self.command = command
        self.returncode = returncode
        self.output = output
        self.seconds = seconds

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return f"{' '.join(self.command)} -> rc={self.returncode} ({self.seconds:.1f}s)"


def run(command: list[str], log=None, timeout: float | None = 3600.0, cwd: Path | None = None) -> RunResult:
    """Execute ``command``; raise :class:`StageError` on a non-zero exit."""
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = time.time()
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding=_ENCODING,
            errors="replace",
            creationflags=creationflags,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise StageError(f"executable not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise StageError(f"timed out after {timeout:.0f}s: {' '.join(str(c) for c in command)}") from exc

    result = RunResult([str(c) for c in command], completed.returncode, (completed.stdout or "").strip(), time.time() - started)
    if log is not None and result.output:
        for line in result.output.splitlines():
            log(f"       | {line}")
    return result


def run_checked(command: list[str], log=None, timeout: float | None = 3600.0, cwd: Path | None = None) -> RunResult:
    result = run(command, log=log, timeout=timeout, cwd=cwd)
    if not result.ok:
        tail = "\n".join(result.output.splitlines()[-12:]) if result.output else "(no output)"
        raise StageError(
            "command failed with exit code "
            f"{result.returncode}:\n  {' '.join(result.command)}\n{tail}"
        )
    return result
