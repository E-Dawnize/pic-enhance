"""``imgen doctor`` — what this machine can and cannot do.

Prints one row per dependency, and for anything missing prints the exact fix.
The point is to answer "why did my pipeline stop" before it stops.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from .backends import onnx_sr
from .config import REPO_ROOT, Tools

__all__ = ["report", "format_report"]


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return ""


def _versions_in(python: Path, packages: tuple[str, ...]) -> dict[str, str] | None:
    """Ask another interpreter for its package versions (no imports of the packages)."""
    code = (
        "import json, importlib.metadata as md\n"
        "out = {}\n"
        f"for name in {list(packages)!r}:\n"
        "    try: out[name] = md.version(name)\n"
        "    except Exception: out[name] = ''\n"
        "print(json.dumps(out))\n"
    )
    try:
        proc = subprocess.run(
            [str(python), "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return None


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
    ):
        found = _version(package)
        rows.append((package, "ok" if found else ("missing" if note == "required" else "optional"), found or note))

    # torch/diffusers only matter where hdr.semantic.external will run them, which
    # is usually the repository venv — not this interpreter.
    semantic_python = Path(tools.python)
    local = semantic_python.resolve() == Path(sys.executable).resolve()
    versions = (
        {name: _version(name) for name in ("torch", "diffusers")}
        if local
        else _versions_in(semantic_python, ("torch", "diffusers"))
    )
    if versions is None:
        rows.append(("semantic python", "missing", f"cannot run {semantic_python}"))
        versions = {"torch": "", "diffusers": ""}
    else:
        where = "this interpreter" if local else str(semantic_python)
        rows.append(("semantic python", "ok", where))
    for package, note in (("torch", "optional: semantic HDR inference"), ("diffusers", "optional: semantic HDR inference")):
        found = versions.get(package, "")
        rows.append((package, "ok" if found else "optional", found or note))
    providers = onnx_sr.providers_available()
    rows.append(("onnx providers", "ok" if providers else "optional", ", ".join(providers) or "none"))
    rows += tools.table()
    rows.append(("lumipic model", *_lumipic()))
    rows.append(("gpu", "info", _gpu()))
    rows.append(("repo", "info", str(REPO_ROOT)))
    return rows


def _lumipic() -> tuple[str, str]:
    """Is the semantic model on disk inside the repository?"""
    from .stages.s_hdr import resolve_semantic_paths

    base, lora = resolve_semantic_paths(None, None)
    if base and lora:
        size = sum(f.stat().st_size for f in base.rglob("*.safetensors") if f.is_file())
        return "ok", f"{base} ({size / 1024 ** 3:.1f} GiB) + {lora.name}"
    missing = [] if base else ["base model"]
    if not lora:
        missing.append("LoRA")
    return "optional", "missing " + ", ".join(missing) + " (python scripts/fetch_models.py --lumipic)"


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
