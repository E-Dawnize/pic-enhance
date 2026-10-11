"""Source and sink stages: load, crop, resize, save."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .. import ops
from ..registry import StageError, stage
from ..state import Frame

__all__ = ["load", "crop", "resize", "save"]


def _stem(frame: Frame, fallback: str = "image") -> str:
    source = frame.meta.get("input_path") or frame.meta.get("source") or ""
    return Path(source).stem or fallback if source else fallback


def _fill(template: str, ctx, frame: Frame) -> str:
    """Expand ``{stem}``/``{name}``/``{index}``/``{stage}`` in a filename."""
    return template.format(
        stem=_stem(frame),
        name=frame.meta.get("name", _stem(frame)),
        index=ctx.index,
        stage=ctx.label or "stage",
    )


@stage(
    "io.load",
    requires=(),
    produces=("rgb",),
    kind="source",
    params={"path": "", "name": ""},
    doc="Read an image (PNG/JPEG/WebP/BMP/TIFF) into the rgb plane. Uses --input when path is empty.",
)
def load(frame: Frame, ctx, path: str = "", name: str = "") -> Frame:
    candidate = path or ctx.env.get("input")
    if not candidate:
        raise StageError("io.load needs a path: pass `--input FILE` or set params.path")
    file = Path(candidate)
    if not file.is_file():
        raise StageError(f"input image not found: {file}")
    with Image.open(file) as handle:
        rgb = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    frame.rgb = rgb
    frame.path = file
    frame.meta.update(
        {
            "input_path": str(file.resolve()),
            "name": name or file.stem,
            "source_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "source_mode": "sRGB 8-bit",
        }
    )
    frame.note(f"loaded {file.name} {rgb.shape[1]}x{rgb.shape[0]}")
    return frame


@stage(
    "io.crop",
    requires=("rgb",),
    produces=("rgb",),
    params={"x": 0, "y": 0, "width": 0, "height": 0, "anchor": "top-left"},
    doc="Crop the rgb plane (width/height 0 = to the edge). Derived planes are dropped.",
)
def crop(frame: Frame, ctx, x: int = 0, y: int = 0, width: int = 0, height: int = 0, anchor: str = "top-left") -> Frame:
    if frame.rgb is None:
        raise StageError("io.crop needs an rgb plane")
    h, w = frame.rgb.shape[:2]
    cw = int(width) or (w - int(x))
    ch = int(height) or (h - int(y))
    if anchor == "center":
        x = (w - cw) // 2
        y = (h - ch) // 2
    elif anchor == "bottom-right":
        x, y = w - cw, h - ch
    x, y = max(0, int(x)), max(0, int(y))
    cw, ch = min(cw, w - x), min(ch, h - y)
    if cw <= 0 or ch <= 0:
        raise StageError(f"io.crop produced an empty region ({cw}x{ch})")
    frame.rgb = np.ascontiguousarray(frame.rgb[y : y + ch, x : x + cw, :])
    frame.invalidate_after("rgb")
    frame.note(f"cropped to {cw}x{ch} at ({x},{y})")
    return frame


@stage(
    "io.resize",
    requires=("rgb",),
    produces=("rgb",),
    params={"width": 0, "height": 0, "long_edge": 0, "scale": 0.0, "interp": "auto", "align": 1},
    doc="Resize the rgb plane. Give width+height, or long_edge, or scale (0 = unused).",
)
def resize(
    frame: Frame,
    ctx,
    width: int = 0,
    height: int = 0,
    long_edge: int = 0,
    scale: float = 0.0,
    interp: str = "auto",
    align: int = 1,
) -> Frame:
    if frame.rgb is None:
        raise StageError("io.resize needs an rgb plane")
    h, w = frame.rgb.shape[:2]
    if long_edge:
        factor = float(long_edge) / max(w, h)
        width, height = round(w * factor), round(h * factor)
    elif scale:
        width, height = round(w * float(scale)), round(h * float(scale))
    if not width or not height:
        raise StageError("io.resize needs width+height, long_edge or scale")
    if align and int(align) > 1:
        step = int(align)
        width = max(step, int(round(width / step)) * step)
        height = max(step, int(round(height / step)) * step)
    frame.rgb = ops.resize(frame.rgb, (int(width), int(height)), interp=interp)
    frame.invalidate_after("rgb")
    frame.meta["scale"] = float(frame.meta.get("scale", 1.0)) * (frame.rgb.shape[1] / w)
    frame.note(f"resized {w}x{h} -> {frame.rgb.shape[1]}x{frame.rgb.shape[0]}")
    return frame


@stage(
    "io.save",
    requires=("rgb",),
    produces=(),
    kind="sink",
    params={"filename": "{stem}.png", "format": "", "quality": 95, "subdir": ""},
    doc="Write the rgb plane (PNG is lossless; JPEG/WebP use `quality`).",
)
def save(frame: Frame, ctx, filename: str = "{stem}.png", format: str = "", quality: int = 95, subdir: str = "") -> Frame:
    if frame.rgb is None:
        raise StageError("io.save needs an rgb plane")
    name = _fill(filename, ctx, frame)
    target = ctx.out(name if not subdir else str(Path(subdir) / name))
    fmt = (format or target.suffix.lstrip(".") or "png").lower()
    if fmt in ("jpg", "jpeg"):
        fmt = "jpeg"
    array = np.clip(frame.rgb, 0.0, 1.0)
    if fmt in ("png", "tiff", "bmp"):
        image = Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), "RGB")
        image.save(target, compress_level=6)
    else:
        image = Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), "RGB")
        image.save(target, quality=int(quality), subsampling=0 if fmt == "jpeg" else None)
    frame.path = target
    ctx.record(target)
    ctx.log(f"       wrote {target}")
    return frame
