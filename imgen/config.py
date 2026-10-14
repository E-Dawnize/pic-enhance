"""Where the external tools live.

Nothing in this toolkit is allowed to hard-code a machine path. Tool locations
are resolved in this order:

1. ``--config FILE`` (a JSON file, see ``config/tools.example.json``)
2. environment variables (``IMGEN_REALESRGAN``, ``IMGEN_FFMPEG``, …)
3. ``config/tools.json`` inside the repository
4. **auto-discovery** — a small candidate list built from a few roots, which
   includes the parent directory of the repository (the layout this project was
   developed in: tools one level up, in ``../tools``).

Every stage that needs a tool checks for it and raises with the exact path it
looked for, so a missing dependency is a one-line fix rather than a stack trace.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Tools", "load_tools", "REPO_ROOT", "CONFIG_PATH"]

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "tools.json"

#: The repository's own virtual environment, when the semantic HDR stage lives in
#: it. ``scripts/setup_env.py`` creates exactly this one, so a clone that has run
#: the setup script needs no ``--python`` override.
VENV_PYTHON = (
    REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if os.name == "nt"
    else REPO_ROOT / ".venv" / "bin" / "python"
)

# Candidate relative paths per tool, tried against every discovery root.
_CANDIDATES: dict[str, tuple[str, ...]] = {
    "realesrgan": (
        "tools/realesrgan-ncnn-vulkan.exe",
        "realesrgan-ncnn-vulkan.exe",
        "tools/realesrgan-ncnn-vulkan-20220424-windows/realesrgan-ncnn-vulkan.exe",
    ),
    "realcugan": (
        "tools/realcugan-ncnn-vulkan.exe",
        "realcugan-ncnn-vulkan.exe",
        "tools/realcugan-ncnn-vulkan-20220728-windows/realcugan-ncnn-vulkan.exe",
    ),
    "waifu2x": (
        "tools/waifu2x-ncnn-vulkan.exe",
        "waifu2x-ncnn-vulkan.exe",
        "tools/waifu2x-ncnn-vulkan-20250915-windows/waifu2x-ncnn-vulkan.exe",
        "tools/waifu2x-ncnn-vulkan-20250915-windows/waifu2x-ncnn-vulkan-20250915-windows/waifu2x-ncnn-vulkan.exe",
    ),
    "jxr_encode": (
        "tools/jxr-encode-wic.exe",
        "jxr-encode-wic.exe",
        "HDR-JXR-工作流与技术文档/tools/jxr-encode-wic.exe",
    ),
    "jxr_inspect": (
        "tools/jxr-inspect.exe",
        "jxr-inspect.exe",
        "HDR-JXR-工作流与技术文档/tools/jxr-inspect.exe",
    ),
    "jxr_verify": (
        "tools/verify-jxr.exe",
        "verify-jxr.exe",
        "HDR-JXR-工作流与技术文档/tools/verify-jxr.exe",
    ),
    "ffmpeg": (
        "tools/ffmpeg.exe",
        "ffmpeg.exe",
        "tools/ffmpeg/bin/ffmpeg.exe",
    ),
}

_ENV_VARS = {
    "realesrgan": "IMGEN_REALESRGAN",
    "realcugan": "IMGEN_REALCUGAN",
    "waifu2x": "IMGEN_WAIFU2X",
    "jxr_encode": "IMGEN_JXR_ENCODE",
    "jxr_inspect": "IMGEN_JXR_INSPECT",
    "jxr_verify": "IMGEN_JXR_VERIFY",
    "ffmpeg": "IMGEN_FFMPEG",
}


def _discovery_roots(extra: list[str] | None = None) -> list[Path]:
    roots: list[Path] = []
    for item in (extra or []):
        roots.append(Path(item).expanduser())
    env_root = os.environ.get("IMGEN_TOOLS_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots += [
        REPO_ROOT,               # ./tools/...
        REPO_ROOT.parent,        # ../tools/...  (the layout used in development)
        REPO_ROOT.parent.parent,
        Path.home() / "Documents",
        Path.home() / "Documents" / "HDR-JXR-工作流与技术文档",
        Path.cwd(),
    ]
    seen, unique = set(), []
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _discover(tool: str, roots: list[Path]) -> Path | None:
    for root in roots:
        for rel in _CANDIDATES.get(tool, ()):
            candidate = root / rel
            if candidate.is_file():
                return candidate
    # last resort: PATH
    exe = shutil.which(tool.replace("_", "-")) or shutil.which(tool)
    return Path(exe) if exe else None


@dataclass
class Tools:
    """Resolved locations of optional external binaries."""

    realesrgan: Path | None = None
    realcugan: Path | None = None
    waifu2x: Path | None = None
    jxr_encode: Path | None = None
    jxr_inspect: Path | None = None
    jxr_verify: Path | None = None
    ffmpeg: Path | None = None
    models_dir: Path | None = None
    python: Path = field(default_factory=lambda: Path(sys.executable))
    extra: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)

    # -- capability checks ------------------------------------------------ #
    def ncnn_exe(self, tool: str) -> Path:
        from .registry import StageError

        path = getattr(self, tool, None)
        hint = {
            "realesrgan": "https://github.com/xinntao/Real-ESRGAN/releases (realesrgan-ncnn-vulkan)",
            "realcugan": "https://github.com/nihui/realcugan-ncnn-vulkan/releases",
            "waifu2x": "https://github.com/nihui/waifu2x-ncnn-vulkan/releases",
        }.get(tool, "")
        if not path or not Path(path).is_file():
            raise StageError(
                f"the {tool} ncnn-vulkan binary was not found.\n"
                f"  fix: drop it in {REPO_ROOT / 'tools'} or set IMGEN_{tool.upper()}=<path>\n"
                f"  download: {hint}"
            )
        return Path(path)

    def require(self, capability: str) -> None:
        """Raise a helpful error when ``capability`` is unavailable."""
        from .registry import StageError

        if capability == "ncnn":
            if not any(getattr(self, n, None) for n in ("realesrgan", "realcugan", "waifu2x")):
                raise StageError("no ncnn-vulkan binary configured; see `imgen doctor`")
        elif capability == "jxr":
            if not self.jxr_encode:
                raise StageError(
                    "jxr-encode-wic.exe was not found, so .jxr output is unavailable.\n"
                    "  fix: set IMGEN_JXR_ENCODE=<path> (and IMGEN_JXR_INSPECT / IMGEN_JXR_VERIFY)\n"
                    "  note: the pipeline can still write scRGB floats with `encode.float`"
                )
        elif capability == "ffmpeg":
            if not self.ffmpeg:
                raise StageError(
                    "ffmpeg was not found (needed by this stage).\n"
                    "  fix: set IMGEN_FFMPEG=<path to ffmpeg.exe>, or use the pure-numpy "
                    "alternative stage for this step"
                )
        elif capability == "onnx":
            try:
                import onnxruntime  # noqa: F401
            except ImportError as exc:  # pragma: no cover
                raise StageError("onnxruntime is not installed (pip install onnxruntime)") from exc
        elif capability == "torch":
            try:
                import torch  # noqa: F401
            except ImportError as exc:  # pragma: no cover
                raise StageError(
                    "torch is not installed in this interpreter.\n"
                    "  fix: point the stage's `python` parameter at the interpreter that has "
                    "diffusers + torch (see scripts/lumipic_infer.py)"
                ) from exc

    def table(self) -> list[tuple[str, str, str]]:
        """``(capability, status, detail)`` rows for `imgen doctor`."""
        rows = []
        for name in ("realesrgan", "realcugan", "waifu2x"):
            path = getattr(self, name, None)
            rows.append((name, "ok" if path else "missing", str(path or "-")))
        # ffmpeg only backs the optional denoise.ffmpeg / deband.ffmpeg stages;
        # every pipeline here has a pure-numpy alternative, so it is never required.
        rows.append(
            (
                "ffmpeg",
                "ok" if self.ffmpeg else "optional",
                str(self.ffmpeg) if self.ffmpeg else "optional: denoise.ffmpeg / deband.ffmpeg only",
            )
        )
        jxr = [self.jxr_encode, self.jxr_inspect, self.jxr_verify]
        rows.append(
            (
                "jxr (WIC)",
                "ok" if jxr[0] else "missing",
                str(self.jxr_encode or "-") + (f"  (+{sum(1 for j in jxr[1:] if j)} helpers)" if jxr[0] else ""),
            )
        )
        rows.append(("models_dir", "ok" if self.models_dir else "auto", str(self.models_dir or "-")))
        return rows


def _read_config_file(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"config file {path} is not valid JSON: {exc}") from exc


def load_tools(config_path: str | Path | None = None, roots: list[str] | None = None) -> Tools:
    """Resolve tool locations (see module docstring for the precedence)."""
    explicit: dict = {}
    for candidate in (CONFIG_PATH, Path(config_path) if config_path else None):
        if candidate:
            data = _read_config_file(Path(candidate))
            explicit.update(data.get("tools", data))

    discovery_roots = _discovery_roots(roots or explicit.get("search_roots"))
    tools = Tools()
    sources: dict[str, str] = {}

    for tool, env in _ENV_VARS.items():
        value = explicit.get(tool) or os.environ.get(env)
        if value:
            path = Path(str(value)).expanduser()
            setattr(tools, tool, path if path.is_file() else None)
            sources[tool] = "config" if explicit.get(tool) else f"env:{env}"
        else:
            found = _discover(tool, discovery_roots)
            setattr(tools, tool, found)
            sources[tool] = "discovered" if found else "missing"

    models_dir = explicit.get("models_dir") or os.environ.get("IMGEN_MODELS_DIR")
    if models_dir:
        tools.models_dir = Path(str(models_dir)).expanduser()
    else:
        for root in discovery_roots:
            if (root / "tools" / "models").is_dir():
                tools.models_dir = root / "tools" / "models"
                break

    if explicit.get("python"):
        tools.python = Path(str(explicit["python"])).expanduser()
        sources["python"] = "config"
    elif VENV_PYTHON.is_file():
        tools.python = VENV_PYTHON
        sources["python"] = "repo venv"
    tools.extra = {k: v for k, v in explicit.items() if k not in _ENV_VARS and k not in ("models_dir", "python")}
    tools.sources = sources
    return tools
