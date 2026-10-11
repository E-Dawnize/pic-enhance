"""ncnn-vulkan super-resolution backends (Real-ESRGAN · Real-CUGAN · waifu2x).

The three tools share a Vulkan/ncnn core but disagree about what ``-n`` and
``-m`` mean, so the command line is built per tool rather than generated from a
template:

=============  ==========================================  =========================
tool           model selection                             noise / denoise
=============  ==========================================  =========================
realesrgan     ``-n <model name>`` (e.g. x4plus-anime)     n/a — baked into model
realcugan      ``-m <models-se|models-pro|models-nose>``   ``-n -1..3``
waifu2x        ``-m <models-cunet|models-upconv_7_...>``   ``-n -1..3``
=============  ==========================================  =========================

Facts worth remembering (verified against upstream help output and READMEs):

* Real-ESRGAN's ``-s`` is *not* a native scale — the network runs at 4× and the
  tool then resizes with Lanczos, so ``-s 2`` is a 4× pass plus a downscale.
* Real-CUGAN's denoise strength is encoded twice: in the model folder name
  (``up2x-no-denoise`` / ``conservative`` / ``denoise1x..3x``) and in ``-n``.
* ``-x`` (TTA) costs ~8× the compute; no project documents a quality win.
* Tile seams are a long-standing ncnn bug, so a large ``--tile`` (or 0 = auto)
  is usually safer than a small one.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Tools
from ..registry import StageError
from . import external

__all__ = ["TOOLS", "MODEL_KINDS", "resolve_model_dir", "build_command", "upscale_file"]

TOOLS = ("realesrgan", "realcugan", "waifu2x")

#: Which kinds of model identifier each tool accepts.
MODEL_KINDS = {
    "realesrgan": ("name",),          # -n
    "realcugan": ("dir",),            # -m
    "waifu2x": ("dir",),              # -m
}

_DEFAULT_MODEL_DIRS = {
    "realcugan": ("models-pro", "models-se", "models-nose"),
    "waifu2x": ("models-cunet", "models-upconv_7_anime_style_art_rgb", "models-upconv_7_photo"),
}


def _search_dir(name: str, roots: list[Path], max_depth: int = 4) -> Path | None:
    """Bounded breadth-first search for a directory called ``name``."""
    wanted = name.lower()
    frontier = [(root, 0) for root in roots if root.is_dir()]
    while frontier:
        current, depth = frontier.pop(0)
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name.lower() == wanted:
                return entry
            if depth + 1 < max_depth and not entry.name.startswith("."):
                frontier.append((entry, depth + 1))
    return None


def resolve_model_dir(tools: Tools, tool: str, name: str | None) -> Path | None:
    """Find a model folder for the ``-m`` tools without hard-coding paths."""
    if not name:
        return None
    direct = Path(name)
    if direct.is_dir():
        return direct
    exe = getattr(tools, tool, None)
    roots = [Path.cwd()]
    if exe:
        roots.insert(0, Path(exe).resolve().parent)
    from ..config import REPO_ROOT

    roots += [REPO_ROOT, REPO_ROOT.parent, REPO_ROOT.parent / "tools"]
    if tools.models_dir:
        roots.insert(0, Path(tools.models_dir))
    found = _search_dir(name, roots)
    if found:
        return found
    # let the tool try to resolve it itself (its own working directory rules)
    return Path(name)


def build_command(
    tools: Tools,
    tool: str,
    input_png: Path,
    output_png: Path,
    *,
    model: str | None = None,
    scale: int | None = None,
    noise: int | None = None,
    tile: int | None = None,
    tta: bool = False,
    gpu: int | None = None,
    threads: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Assemble the command line for one of the three ncnn tools."""
    if tool not in TOOLS:
        raise StageError(f"unknown ncnn tool {tool!r}; expected one of {', '.join(TOOLS)}")
    exe = tools.ncnn_exe(tool)
    command: list[str] = [str(exe), "-i", str(input_png), "-o", str(output_png)]

    if model:
        if tool == "realesrgan":
            command += ["-n", str(model)]
        else:
            resolved = resolve_model_dir(tools, tool, model)
            command += ["-m", str(resolved) if resolved else str(model)]
    else:
        default = _DEFAULT_MODEL_DIRS.get(tool)
        if default:
            resolved = resolve_model_dir(tools, tool, default[0])
            command += ["-m", str(resolved) if resolved else default[0]]

    if scale:
        command += ["-s", str(int(scale))]
    if noise is not None and tool in ("realcugan", "waifu2x"):
        command += ["-n", str(int(noise))]
    if tile:
        command += ["-t", str(int(tile))]
    if gpu is not None:
        command += ["-g", str(int(gpu))]
    if threads:
        command += ["-j", str(threads)]
    if tta:
        command += ["-x"]
    if extra_args:
        command += [str(a) for a in extra_args]
    return command


def upscale_file(
    tools: Tools,
    tool: str,
    input_png: Path,
    output_png: Path,
    log=None,
    timeout: float = 7200.0,
    **kwargs,
) -> external.RunResult:
    """Run an ncnn upscaler, raising if it produced no output."""
    command = build_command(tools, tool, input_png, output_png, **kwargs)
    if log is not None:
        log("       $ " + " ".join(command))
    result = external.run_checked(command, log=log, timeout=timeout)
    if not Path(output_png).is_file():
        raise StageError(f"{tool} reported success but wrote no file to {output_png}")
    return result
