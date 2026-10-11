"""Deband stages.

Banding is a *quantisation* artifact: a smooth gradient arrives as flat steps,
and any contrast boost (including HDR expansion) makes the steps obvious. Two
implementations:

``deband.gradient``  pure numpy, four-direction neighbour test in the spirit of
                     ffmpeg's ``vf_deband`` (default threshold 0.02, range 16),
                     followed by optional triangular dither.
``deband.ffmpeg``    ffmpeg's own ``deband`` / ``gradfun`` filters.

Note from the field (verified against the FFmpeg sources): ``gradfun`` is
**8-bit only** and dithers internally; for 10-bit or better, ``deband`` with
``neo_f3kdb``/VapourSynth is the community standard. ``deband`` has no ``planes``
option, whatever the folklore says.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import ops
from ..colors import LUMA
from ..registry import StageError, stage
from ..state import Frame

__all__ = ["gradient", "ffmpeg"]

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = a.shape[:2]
    out = np.zeros_like(a)
    src_y0, src_y1 = max(0, -dy), h - max(0, dy)
    src_x0, src_x1 = max(0, -dx), w - max(0, dx)
    dst_y0, dst_y1 = max(0, dy), h - max(0, -dy)
    dst_x0, dst_x1 = max(0, dx), w - max(0, -dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = a[src_y0:src_y1, src_x0:src_x1]
    return out


def _deband_plane(plane: np.ndarray, threshold: float, span: int, iterations: int, strength: float) -> np.ndarray:
    out = plane.astype(np.float32, copy=True)
    for _ in range(max(1, int(iterations))):
        acc = np.zeros_like(out)
        count = np.zeros_like(out)
        for dy, dx in _DIRECTIONS:
            ahead = _shift(out, dy * span, dx * span)
            behind = _shift(out, -dy * span, -dx * span)
            average = 0.5 * (ahead + behind)
            close = (np.abs(out - average) < threshold).astype(np.float32)
            acc += average * close
            count += close
        has = count > 0
        blended = np.where(has, acc / np.maximum(count, 1e-6), out)
        out = np.where(has, out + (blended - out) * float(strength), out)
    return out


@stage(
    "deband.gradient",
    requires=("rgb",),
    produces=("rgb",),
    params={
        "threshold": 0.02,
        "span": 16,
        "iterations": 2,
        "strength": 0.6,
        "dither": 0.6,
        "seed": 0,
        "luma_only": True,
    },
    doc="Four-direction gradient deband with optional dither (pure numpy).",
)
def gradient(
    frame: Frame,
    ctx,
    threshold: float = 0.02,
    span: int = 16,
    iterations: int = 2,
    strength: float = 0.6,
    dither: float = 0.6,
    seed: int = 0,
    luma_only: bool = True,
) -> Frame:
    if frame.rgb is None:
        raise StageError("deband.gradient needs an rgb plane")
    rgb = frame.rgb.astype(np.float32, copy=True)
    luma = rgb @ LUMA
    fixed = _deband_plane(luma, float(threshold), int(span), int(iterations), float(strength))
    if luma_only:
        ratio = fixed / np.maximum(luma, 1e-6)
        rgb = np.clip(rgb * ratio[..., None], 0.0, 1.0)
    else:
        for channel in range(3):
            rgb[..., channel] = _deband_plane(rgb[..., channel], float(threshold), int(span), int(iterations), float(strength))
        rgb = np.clip(rgb, 0.0, 1.0)
    if dither:
        rng = np.random.default_rng(int(seed))
        # Triangular noise (sum of two uniforms) is the classic dither shape.
        noise = (rng.random(rgb.shape, np.float32) - rng.random(rgb.shape, np.float32)) * (float(dither) / 255.0)
        rgb = np.clip(rgb + noise, 0.0, 1.0)
    frame.rgb = rgb
    frame.invalidate_after("rgb")
    frame.note(f"gradient deband thr={threshold} span={span} x{iterations} dither={dither}")
    return frame


_PRESETS = {
    "gradfun": "gradfun=strength=1.2:radius=16",
    "deband": "deband=1thr=0.02:2thr=0.02:3thr=0.02:4thr=0.02:range=16:blur=1",
    "deband-soft": "deband=1thr=0.012:2thr=0.012:3thr=0.012:4thr=0.012:range=16:blur=1",
    "deband-strong": "deband=1thr=0.04:2thr=0.04:3thr=0.04:4thr=0.04:range=16:blur=1",
}


@stage(
    "deband.ffmpeg",
    requires=("rgb",),
    produces=("rgb",),
    needs=("ffmpeg",),
    params={"preset": "gradfun", "filter": ""},
    doc="Deband through ffmpeg (gradfun / deband). gradfun is 8-bit only.",
)
def ffmpeg(frame: Frame, ctx, preset: str = "gradfun", filter: str = "") -> Frame:
    from ..backends import external

    if frame.rgb is None:
        raise StageError("deband.ffmpeg needs an rgb plane")
    chain = filter or _PRESETS.get(preset)
    if not chain:
        raise StageError(f"unknown preset {preset!r}; known: {', '.join(sorted(_PRESETS))}")
    source = ctx.temp(f"{ctx.index:02d}-deband-in.png")
    target = ctx.temp(f"{ctx.index:02d}-deband-out.png")
    Image.fromarray((np.clip(frame.rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8), "RGB").save(source)
    command = [str(ctx.tools.ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-vf", chain, "-pix_fmt", "rgb24", str(target)]
    if ctx.log:
        ctx.log("       $ " + " ".join(command))
    external.run_checked(command, log=ctx.log, timeout=3600.0)
    with Image.open(target) as handle:
        frame.rgb = np.asarray(handle.convert("RGB"), np.float32) / 255.0
    frame.invalidate_after("rgb")
    frame.note(f"ffmpeg deband: {chain}")
    return frame
