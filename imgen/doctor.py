"""``imgen doctor`` — what this machine can and cannot do.

Prints one row per dependency, and for anything missing prints the exact fix.
The point is to answer "why did my pipeline stop" before it stops.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from importlib import metadata

from .backends import onnx_sr
from .config import REPO_ROOT, Tools

__all__ = ["report", "format_report"]


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return ""


def _gpu() -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "nvidia-smi not on PATH"
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
            check=False,
        ).stdout.strip()
        return out or "no GPU reported"
    except Exception as exc:  # pragma: no cover
        return f"query failed: {exc}"


def report(tools: Tools) -> list[tuple[str, str, str]]:
    """``(item, status, detail)`` rows."""
    rows: list[tuple[str, str, str]] = []
    rows.append(("python", "ok", f"{platform.python_version()} ({sys.executable})"))
    for package, note in (
        ("numpy", "required"),
        ("pillow", "required"),
        ("onnxruntime", "optional: ONNX upscaling"),
        ("onnxruntime-gpu", "optional: CUDA/DML providers"),
        ("torch", "optional: semantic HDR inference"),
        ("diffusers", "optional: semantic HDR inference"),
    ):
        found = _version(package)
        rows.append((package, "ok" if found else ("missing" if note == "required" else "optional"), found or note))
    providers = onnx_sr.providers_available()
    rows.append(("onnx providers", "ok" if providers else "optional", ", ".join(providers) or "none"))
    rows += tools.table()
    rows.append(("gpu", "info", _gpu()))
    rows.append(("repo", "info", str(REPO_ROOT)))
    return rows


def format_report(rows: list[tuple[str, str, str]]) -> str:
    width = max(len(name) for name, _, _ in rows) + 2
    lines = ["imgen doctor", "=" * (width + 46)]
    for name, status, detail in rows:
        marker = {"ok": "ok  ", "missing": "MISS", "optional": "--  ", "info": "info"}.get(status, status)
        lines.append(f"  {marker}  {name:<{width}}{detail}")
    missing = [name for name, status, _ in rows if status == "missing"]
    lines.append("")
    if missing:
        lines.append(f"  missing required: {', '.join(missing)}")
    else:
        lines.append("  required dependencies are present.")
    lines.append("  tip: `imgen stages` lists every stage; `imgen doctor --json` for machine-readable output.")
    return "\n".join(lines)
