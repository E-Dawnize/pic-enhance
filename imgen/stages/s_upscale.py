"""Upscaling stages: ncnn-vulkan executables and tiled ONNX models.

Both stages work through an 8-bit PNG (ncnn) or a float tile loop (ONNX) and
write the result back into the ``rgb`` plane. Repeat the stage to chain passes —
``2× then 2×`` is a legitimate and often better choice than one ``4×`` pass,
because the second pass *sees* the first pass's output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..backends import ncnn, onnx_sr
from ..config import REPO_ROOT
from ..registry import StageError, stage
from ..state import Frame

__all__ = ["ncnn_tool", "onnx_model"]


def _write_png(rgb: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGB").save(path, compress_level=1)
    return path


def _read_png(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), np.float32) / 255.0


def resolve_model_path(name: str, tools) -> Path:
    """Find an ONNX model by filename, bare name or search."""
    direct = Path(name)
    if direct.is_file():
        return direct
    stem = direct.stem if direct.suffix else direct.name
    roots = [REPO_ROOT, REPO_ROOT / "models", REPO_ROOT.parent, REPO_ROOT.parent / "tools", Path.cwd()]
    if tools and tools.models_dir:
        roots.insert(0, Path(tools.models_dir))
    for root in roots:
        for candidate in (root / f"{stem}.onnx", root / name):
            if candidate.is_file():
                return candidate
    raise StageError(
        f"ONNX model {name!r} not found. Looked in:\n  "
        + "\n  ".join(str(r) for r in roots)
        + "\n  fix: pass an absolute path, or put the file next to the pipeline."
    )


@stage(
    "upscale.ncnn",
    requires=("rgb",),
    produces=("rgb",),
    needs=("ncnn",),
    params={
        "tool": "realesrgan",
        "model": "realesrgan-x4plus-anime",
        "scale": 4,
        "noise": None,
        "tta": False,
        "tile": 0,
        "gpu": None,
        "threads": None,
        "timeout": 7200,
        "extra_args": [],
    },
    doc="Upscale with an ncnn-vulkan binary (realesrgan | realcugan | waifu2x). Repeat the stage to chain passes.",
)
def ncnn_tool(
    frame: Frame,
    ctx,
    tool: str = "realesrgan",
    model: str = "realesrgan-x4plus-anime",
    scale: int = 4,
    noise: int | None = None,
    tta: bool = False,
    tile: int = 0,
    gpu: int | None = None,
    threads: str | None = None,
    timeout: float = 7200,
    extra_args: list | None = None,
) -> Frame:
    if frame.rgb is None:
        raise StageError("upscale.ncnn needs an rgb plane")
    source = _write_png(frame.rgb, ctx.temp(f"{ctx.index:02d}-sr-in.png"))
    target = ctx.temp(f"{ctx.index:02d}-sr-out.png")
    before = frame.size
    kwargs = dict(
        model=model or None,
        scale=int(scale) if scale else None,
        noise=noise,
        tile=int(tile) if tile else None,
        tta=bool(tta),
        gpu=gpu,
        threads=threads,
        extra_args=list(extra_args or []),
    )
    if tool == "realesrgan" and scale and int(scale) != 4:
        ctx.log(f"       note: realesrgan has no native {scale}x — it renders 4x then Lanczos-resizes")
    ncnn.upscale_file(ctx.tools, tool, source, target, log=ctx.log, timeout=float(timeout), **kwargs)
    frame.rgb = _read_png(target)
    frame.invalidate_after("rgb")
    frame.meta.setdefault("passes", []).append(
        {"stage": f"upscale.ncnn:{tool}", "model": model, "scale": scale, "tta": bool(tta), "size": list(frame.size)}
    )
    frame.note(f"{tool} {model} -> {frame.size[0]}x{frame.size[1]} (from {before[0]}x{before[1]})")
    return frame


@stage(
    "upscale.onnx",
    requires=("rgb",),
    produces=("rgb",),
    needs=("onnx",),
    params={
        "model": "",
        "scale": 2,
        "tile": 1024,
        "overlap": 64,
        "fp16": True,
        "provider": "auto",
        "timeout": 7200,
    },
    doc="Tiled ONNX super-resolution with feathered (seam-free) blending.",
)
def onnx_model(
    frame: Frame,
    ctx,
    model: str = "",
    scale: int = 2,
    tile: int = 1024,
    overlap: int = 64,
    fp16: bool = True,
    provider: str = "auto",
    timeout: float = 7200,
) -> Frame:
    if frame.rgb is None:
        raise StageError("upscale.onnx needs an rgb plane")
    if not model:
        raise StageError("upscale.onnx needs a model (name or path to .onnx)")
    path = resolve_model_path(model, ctx.tools)
    ctx.log(f"       model: {path.name}  providers: {', '.join(onnx_sr.providers_available()) or 'cpu'}")
    session = onnx_sr.load_session(path, provider=provider)
    before = frame.size
    frame.rgb = onnx_sr.tiled_upscale(
        session,
        frame.rgb,
        scale=int(scale),
        tile=int(tile),
        overlap=int(overlap),
        fp16=bool(fp16),
        log=ctx.log,
    )
    frame.invalidate_after("rgb")
    frame.meta.setdefault("passes", []).append(
        {"stage": "upscale.onnx", "model": path.name, "scale": int(scale), "size": list(frame.size)}
    )
    frame.note(f"onnx {path.name} -> {frame.size[0]}x{frame.size[1]} (from {before[0]}x{before[1]})")
    return frame
