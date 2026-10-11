"""The frame that travels through a pipeline, and the plane vocabulary.

A :class:`Frame` is a small bag of *planes*. Each stage declares which planes it
**requires** and which it **produces**; the runner refuses to build a pipeline
whose chain is inconsistent, so inserting or deleting a stage gives an immediate
error instead of a mysterious crash three stages later.

Planes
------
``rgb``       float32 (H, W, 3) sRGB-encoded [0, 1] — the working SDR image.
``semantic``  float32 (H, W, 3) scene-linear, may be low resolution — a
              generative model's opinion of *where* light comes from.
``highlight`` float32 (H, W) [0, 1] — the resolved highlight fraction, the
              thing that decides how many nits each pixel gets.
``nits``      float32 (H, W, 3) absolute luminance (cd/m²).
``scrgb``     float32 (H, W, 3) Windows scRGB (1.0 == 80 cd/m²).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Every plane a stage may declare.
PLANES: tuple[str, ...] = ("rgb", "semantic", "highlight", "nits", "scrgb")

#: Planes whose pixels are floats that must survive a round trip losslessly.
FLOAT_PLANES: tuple[str, ...] = ("semantic", "highlight", "nits", "scrgb")


@dataclass
class Frame:
    """Image state passed from stage to stage."""

    rgb: np.ndarray | None = None
    semantic: np.ndarray | None = None
    highlight: np.ndarray | None = None
    nits: np.ndarray | None = None
    scrgb: np.ndarray | None = None

    #: Path of the file this frame was last read from or written to.
    path: Path | None = None
    #: Free-form bookkeeping: source name, scale factor, stage statistics.
    meta: dict[str, Any] = field(default_factory=dict)
    #: Human-readable log lines emitted by stages for the run report.
    notes: list[str] = field(default_factory=list)

    # -- helpers ---------------------------------------------------------- #
    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` of the first plane that exists."""
        for name in PLANES:
            plane = getattr(self, name)
            if plane is not None:
                return int(plane.shape[1]), int(plane.shape[0])
        return (0, 0)

    def has(self, plane: str) -> bool:
        return getattr(self, plane, None) is not None

    def get(self, plane: str) -> np.ndarray | None:
        return getattr(self, plane, None)

    def set(self, plane: str, value: np.ndarray | None) -> None:
        if plane not in PLANES:
            raise KeyError(f"unknown plane {plane!r}; known: {', '.join(PLANES)}")
        setattr(self, plane, value)

    def clear(self, plane: str) -> None:
        self.set(plane, None)

    def invalidate_after(self, changed: str) -> None:
        """Drop planes that can no longer be trusted after ``changed`` is edited.

        Editing ``rgb`` (denoise, upscale, deband, sharpen …) invalidates every
        derived plane, because they were computed against different pixels.
        """
        order = list(PLANES)
        if changed not in order:
            return
        for plane in order[order.index(changed) + 1 :]:
            setattr(self, plane, None)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def summary(self) -> str:
        parts = []
        for name in PLANES:
            plane = getattr(self, name)
            if plane is not None:
                parts.append(f"{name}{tuple(plane.shape)}:{plane.dtype}")
        return f"Frame({self.size[0]}x{self.size[1]}, " + ", ".join(parts) + ")"
