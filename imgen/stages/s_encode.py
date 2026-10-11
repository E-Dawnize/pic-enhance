"""Delivery stages: scRGB floats, JPEG XR (Windows HDR wallpaper) and reports.

Windows has exactly one built-in path for an HDR wallpaper — a JPEG XR file in
scRGB half-float — and the compositor *clips* rather than tone-maps anything
above the panel's peak, so the mapping has to be right before the encode. These
stages do the last mile: write the interchange float, encode the .jxr, and verify
that what came out of WIC is what went in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..backends import jxr
from ..colors import LUMA
from ..registry import StageError, stage
from ..state import Frame

__all__ = ["float_raw", "jxr_encode"]


@stage(
    "encode.float",
    requires=("scrgb",),
    produces=(),
    kind="sink",
    params={"filename": "{stem}.scrgb.f32", "format": "f32", "subdir": ""},
    doc="Write the scRGB plane as packed float32 (.f32) or a numpy array (.npy) for other encoders.",
)
def float_raw(
    frame: Frame,
    ctx,
    filename: str = "{stem}.scrgb.f32",
    format: str = "f32",
    subdir: str = "",
) -> Frame:
    if frame.scrgb is None:
        raise StageError("encode.float needs the scrgb plane (run hdr.map first)")
    name = filename.format(stem=frame.meta.get("name", "image"))
    target = ctx.out(str(Path(subdir) / name) if subdir else name)
    if format == "npy":
        np.save(target, np.ascontiguousarray(frame.scrgb, dtype=np.float32))
    elif format == "f32":
        jxr.write_raw_float(frame.scrgb, target)
    else:
        raise StageError(f"unknown format {format!r}; use f32 | npy")
    ctx.record(target)
    ctx.log(f"       wrote {target}")
    frame.note(f"scRGB float written to {target.name}")
    return frame


@stage(
    "encode.jxr",
    requires=("scrgb",),
    produces=(),
    kind="sink",
    needs=("jxr",),
    params={"filename": "{stem}.jxr", "quality": 1.0, "verify": True, "keep_raw": False, "subdir": ""},
    doc="Encode scRGB to .jxr (64bppRGBAHalf/scRGB) via WIC and verify the decode.",
)
def jxr_encode(
    frame: Frame,
    ctx,
    filename: str = "{stem}.jxr",
    quality: float = 1.0,
    verify: bool = True,
    keep_raw: bool = False,
    subdir: str = "",
) -> Frame:
    if frame.scrgb is None:
        raise StageError("encode.jxr needs the scrgb plane (run hdr.map first)")
    name = filename.format(stem=frame.meta.get("name", "image"))
    target = ctx.out(str(Path(subdir) / name) if subdir else name)
    raw = ctx.temp(target.with_suffix(".f32").name) if not keep_raw else target.with_suffix(".f32")
    jxr.encode_scrgb(frame.scrgb, target, ctx.tools, quality=float(quality), raw_path=Path(raw) if keep_raw else None)

    info = jxr.inspect(target, ctx.tools)
    if info and ctx.log:
        for line in info.splitlines():
            ctx.log(f"       | {line}")
    ok, message = (True, "verify disabled")
    if verify:
        ok, message = jxr.verify(target, ctx.tools)
        if ctx.log:
            ctx.log(f"       verify: {'OK' if ok else 'FAILED'} — {message.splitlines()[0] if message else ''}")
    if not ok:
        raise StageError(f"JXR verification failed for {target}:\n{message}")

    size_mb = target.stat().st_size / (1024 * 1024)
    stats = frame.meta.setdefault("stats", {})
    stats["jxr"] = {
        "file": str(target),
        "megabytes": round(size_mb, 2),
        "verified": bool(ok),
        "size": list(frame.size),
    }
    if frame.nits is not None:
        stats["peak_nits"] = float(frame.nits.max())
        stats["mean_luma_nits"] = float((frame.nits @ LUMA).mean())
    frame.path = target
    frame.note(f"encoded {target.name} ({size_mb:.1f} MB)")
    ctx.record(target)
    ctx.log(f"       wrote {target}  ({size_mb:.1f} MB)")
    return frame
