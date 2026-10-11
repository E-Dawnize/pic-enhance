"""Denoise stages.

Two families:

* **Pure numpy** (``denoise.nlmeans``, ``denoise.bilateral``) — no dependency,
  lossless float path, operate on luma with hue-preserving chroma. Slow at 4K:
  they are meant for the *pre-upscale* image, which is where noise actually
  hurts (the upscaler would otherwise amplify it).
* **ffmpeg** (``denoise.ffmpeg``) — the muscle most people already have; works
  on 8-bit PNG intermediates, so place it before an upscale, not after.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .. import ops
from ..colors import LUMA, linear_to_srgb, srgb_to_linear
from ..registry import StageError, stage
from ..state import Frame

__all__ = ["nlmeans", "bilateral", "ffmpeg"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _split_luma(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(luma_srgb, chroma_scale)`` where ``rgb = luma * chroma_scale``."""
    y = rgb @ LUMA
    return y, rgb / np.maximum(y[..., None], 1e-6)


def _merge_luma(y_new: np.ndarray, chroma: np.ndarray, rgb_original: np.ndarray) -> np.ndarray:
    out = chroma * np.maximum(y_new, 0.0)[..., None]
    return np.clip(np.where((rgb_original @ LUMA)[..., None] <= 1e-6, 0.0, out), 0.0, 1.0)


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift with zero padding (invalid neighbours contribute a 0 weight)."""
    h, w = a.shape[:2]
    out = np.zeros_like(a)
    src_y0, src_y1 = max(0, -dy), h - max(0, dy)
    src_x0, src_x1 = max(0, -dx), w - max(0, dx)
    dst_y0, dst_y1 = max(0, dy), h - max(0, -dy)
    dst_x0, dst_x1 = max(0, dx), w - max(0, -dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = a[src_y0:src_y1, src_x0:src_x1]
    return out


# --------------------------------------------------------------------------- #
# pure-numpy denoisers
# --------------------------------------------------------------------------- #
@stage(
    "denoise.nlmeans",
    requires=("rgb",),
    produces=("rgb",),
    params={
        "strength": 0.010,
        "patch": 5,
        "window": 9,
        "search_stride": 2,
        "passes": 1,
        "chroma": False,
        "sigma_floor": 0.004,
    },
    doc="Luma NL-means (non-local means). Hue-preserving; best applied before upscaling.",
)
def nlmeans(
    frame: Frame,
    ctx,
    strength: float = 0.010,
    patch: int = 5,
    window: int = 9,
    search_stride: int = 2,
    passes: int = 1,
    chroma: bool = False,
    sigma_floor: float = 0.004,
) -> Frame:
    if frame.rgb is None:
        raise StageError("denoise.nlmeans needs an rgb plane")
    pixels = frame.rgb.shape[0] * frame.rgb.shape[1]
    if pixels > 12_000_000 and search_stride < 2:
        raise StageError(
            f"refusing to run NL-means with search_stride={search_stride} on {pixels/1e6:.1f} MP "
            "(hours of CPU). Raise search_stride, downscale first, or use denoise.ffmpeg."
        )
    y, chroma_scale = _split_luma(frame.rgb)
    baseline = y.copy()
    half = max(1, int(window) // 2)
    stride = max(1, int(search_stride))
    offsets = [
        (dy, dx)
        for dy in range(-half, half + 1, stride)
        for dx in range(-half, half + 1, stride)
        if not (dy == 0 and dx == 0)
    ]
    h2 = max(float(strength), 1e-5) ** 2
    var_floor = 2.0 * float(sigma_floor) ** 2
    for _ in range(max(1, int(passes))):
        acc = baseline.copy()
        weight = np.ones_like(baseline)
        for dy, dx in offsets:
            shifted = _shift(baseline, dy, dx)
            diff = (baseline - shifted) ** 2
            distance = ops.box_mean(diff, max(3, int(patch)))
            w = np.exp(-np.maximum(distance - var_floor, 0.0) / h2).astype(np.float32)
            acc += w * shifted
            weight += w
        y = acc / weight
    if chroma:
        rgb = np.clip(chroma_scale * y[..., None], 0.0, 1.0)
    else:
        rgb = _merge_luma(y, chroma_scale, frame.rgb)
    frame.rgb = rgb.astype(np.float32)
    frame.invalidate_after("rgb")
    frame.note(f"nlmeans strength={strength} window={window} stride={stride} ({len(offsets)} offsets)")
    return frame


@stage(
    "denoise.bilateral",
    requires=("rgb",),
    produces=("rgb",),
    params={"radius": 2, "sigma_color": 0.06, "sigma_space": 1.5, "passes": 1},
    doc="Fast small-kernel bilateral filter on luma (edge preserving, no ringing).",
)
def bilateral(
    frame: Frame,
    ctx,
    radius: int = 2,
    sigma_color: float = 0.06,
    sigma_space: float = 1.5,
    passes: int = 1,
) -> Frame:
    if frame.rgb is None:
        raise StageError("denoise.bilateral needs an rgb plane")
    y, chroma_scale = _split_luma(frame.rgb)
    radius = max(1, min(int(radius), 6))
    spatial = np.zeros((2 * radius + 1, 2 * radius + 1), np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial[dy + radius, dx + radius] = np.exp(-(dy * dy + dx * dx) / (2.0 * sigma_space**2))
    colour_scale = -1.0 / (2.0 * max(sigma_color, 1e-6) ** 2)
    for _ in range(max(1, int(passes))):
        acc = np.zeros_like(y)
        weight = np.zeros_like(y)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shifted = _shift(y, dy, dx)
                w = spatial[dy + radius, dx + radius] * np.exp(colour_scale * (y - shifted) ** 2)
                acc += w * shifted
                weight += w
        y = acc / np.maximum(weight, 1e-8)
    frame.rgb = _merge_luma(y, chroma_scale, frame.rgb).astype(np.float32)
    frame.invalidate_after("rgb")
    frame.note(f"bilateral r={radius} sigma_color={sigma_color} x{max(1, int(passes))}")
    return frame


# --------------------------------------------------------------------------- #
# ffmpeg denoisers
# --------------------------------------------------------------------------- #
_PRESETS = {
    "light": "hqdn3d=2:1:2:3",
    "medium": "hqdn3d=4:3:6:4.5",
    "strong": "hqdn3d=8:6:12:9",
    "hqdn3d-light": "hqdn3d=2:1:2:3",
    "hqdn3d-medium": "hqdn3d=4:3:6:4.5",
    "hqdn3d-strong": "hqdn3d=8:6:12:9",
    "nlmeans8": "nlmeans=s=8:p=7:r=15",
    "nlmeans12": "nlmeans=s=12:p=7:r=15",
    "nlmeans16": "nlmeans=s=16:p=7:r=15",
}


@stage(
    "denoise.ffmpeg",
    requires=("rgb",),
    produces=("rgb",),
    needs=("ffmpeg",),
    params={"preset": "hqdn3d-light", "filter": "", "threads": 0},
    doc="Denoise through ffmpeg (hqdn3d / nlmeans presets). 8-bit PNG intermediate.",
)
def ffmpeg(frame: Frame, ctx, preset: str = "hqdn3d-light", filter: str = "", threads: int = 0) -> Frame:
    from ..backends import external

    if frame.rgb is None:
        raise StageError("denoise.ffmpeg needs an rgb plane")
    chain = filter or _PRESETS.get(preset)
    if not chain:
        raise StageError(f"unknown preset {preset!r}; known: {', '.join(sorted(_PRESETS))}")
    source = ctx.temp(f"{ctx.index:02d}-dn-in.png")
    target = ctx.temp(f"{ctx.index:02d}-dn-out.png")
    Image.fromarray((np.clip(frame.rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8), "RGB").save(source)
    command = [str(ctx.tools.ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-vf", chain]
    if threads:
        command += ["-threads", str(int(threads))]
    command += ["-pix_fmt", "rgb24", str(target)]
    if ctx.log:
        ctx.log("       $ " + " ".join(command))
    external.run_checked(command, log=ctx.log, timeout=3600.0)
    with Image.open(target) as handle:
        frame.rgb = np.asarray(handle.convert("RGB"), np.float32) / 255.0
    frame.invalidate_after("rgb")
    frame.note(f"ffmpeg denoise: {chain}")
    return frame
