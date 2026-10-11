"""Sharpening stages.

Sharpening is applied to **luma only** by default: sharpening R, G and B
independently is what produces coloured fringes on high-contrast anime line art.
The classic unsharp mask is used because its two parameters (radius, amount)
map directly onto the two failure modes people actually see — halos (radius too
large) and crunch (amount too high).
"""

from __future__ import annotations

import numpy as np

from .. import ops
from ..colors import LUMA
from ..registry import StageError, stage
from ..state import Frame

__all__ = ["unsharp", "highpass"]


@stage(
    "sharpen.unsharp",
    requires=("rgb",),
    produces=("rgb",),
    params={"radius": 1.0, "amount": 0.5, "threshold": 0.0, "luma_only": True, "clamp": True},
    doc="Unsharp mask on luma. `threshold` (in 0-1 units) protects smooth areas from noise amplification.",
)
def unsharp(
    frame: Frame,
    ctx,
    radius: float = 1.0,
    amount: float = 0.5,
    threshold: float = 0.0,
    luma_only: bool = True,
    clamp: bool = True,
) -> Frame:
    if frame.rgb is None:
        raise StageError("sharpen.unsharp needs an rgb plane")
    rgb = frame.rgb.astype(np.float32, copy=True)
    if luma_only:
        y = rgb @ LUMA
        blurred = ops.gaussian_blur(y, float(radius))
        detail = y - blurred
        if threshold > 0:
            detail = np.where(np.abs(detail) >= float(threshold), detail, 0.0)
        y_new = y + detail * float(amount)
        ratio = y_new / np.maximum(y, 1e-6)
        rgb = rgb * ratio[..., None]
    else:
        blurred = ops.gaussian_blur(rgb, float(radius))
        detail = rgb - blurred
        if threshold > 0:
            detail = np.where(np.abs(detail) >= float(threshold), detail, 0.0)
        rgb = rgb + detail * float(amount)
    frame.rgb = np.clip(rgb, 0.0, 1.0) if clamp else rgb.astype(np.float32)
    frame.invalidate_after("rgb")
    frame.note(f"unsharp radius={radius} amount={amount} threshold={threshold} luma_only={luma_only}")
    return frame


@stage(
    "sharpen.highpass",
    requires=("rgb",),
    produces=("rgb",),
    params={"radius": 2.0, "amount": 0.25, "luma_only": True},
    doc="Add a blurred high-pass layer (soft local contrast, no edge overshoot control).",
)
def highpass(frame: Frame, ctx, radius: float = 2.0, amount: float = 0.25, luma_only: bool = True) -> Frame:
    if frame.rgb is None:
        raise StageError("sharpen.highpass needs an rgb plane")
    blur_radius = max(0.3, float(radius))
    rgb = frame.rgb.astype(np.float32, copy=True)
    if luma_only:
        y = rgb @ LUMA
        low = ops.gaussian_blur(y, blur_radius)
        y_new = y + (y - low) * float(amount)
        rgb = rgb * (y_new / np.maximum(y, 1e-6))[..., None]
    else:
        low = ops.gaussian_blur(rgb, blur_radius)
        rgb = rgb + (rgb - low) * float(amount)
    frame.rgb = np.clip(rgb, 0.0, 1.0)
    frame.invalidate_after("rgb")
    frame.note(f"highpass radius={radius} amount={amount}")
    return frame
