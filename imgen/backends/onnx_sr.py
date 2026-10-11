"""Tiled ONNX super-resolution with feathered blending.

Why tiling: a 4K anime frame through a 2× RCAN needs more VRAM than most
consumer cards have, and a single pass at that size is where "CUDA out of
memory" comes from. Why *feathering*: naive tile averaging (or overlapping
without weights) leaves visible seam lines, the single most reported artifact of
tiled upscaling. Here every tile is accumulated with a raised-cosine window, so
the overlap ramps smoothly from one tile to the next; the accumulated weights
are divided out at the end, which also fixes the borders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..registry import StageError

__all__ = ["providers_available", "load_session", "tiled_upscale", "feather_window"]


def providers_available() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover
        return []
    return list(ort.get_available_providers())


def _pick_providers(preference: str) -> list[str]:
    available = providers_available()
    if not available:
        raise StageError("onnxruntime is not installed (pip install onnxruntime or onnxruntime-gpu)")
    order = {
        "auto": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"],
        "gpu": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"],
        "cuda": ["CUDAExecutionProvider"],
        "dml": ["DmlExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }.get(preference)
    if order is None:
        raise StageError(f"unknown provider preference {preference!r}; use auto|gpu|cuda|dml|cpu")
    chosen = [p for p in order if p in available]
    return chosen or ["CPUExecutionProvider"]


def load_session(model_path: Path, provider: str = "auto"):
    """Create an onnxruntime session, preferring a GPU provider when present."""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise StageError("onnxruntime is not installed (pip install onnxruntime)") from exc
    if not Path(model_path).is_file():
        raise StageError(f"ONNX model not found: {model_path}")
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=_pick_providers(provider))
    return session


def feather_window(height: int, width: int, overlap: int) -> np.ndarray:
    """Raised-cosine 2-D window; 1 in the tile interior, tapering over ``overlap``."""
    wy = np.ones(height, np.float32)
    wx = np.ones(width, np.float32)
    if overlap > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(overlap, dtype=np.float32) + 0.5) / overlap)
        if 2 * overlap <= height:
            wy[:overlap] = ramp
            wy[-overlap:] = ramp[::-1]
        if 2 * overlap <= width:
            wx[:overlap] = ramp
            wx[-overlap:] = ramp[::-1]
    return np.outer(wy, wx)


def tiled_upscale(
    session,
    rgb: np.ndarray,
    scale: int,
    tile: int = 1024,
    overlap: int = 64,
    fp16: bool = True,
    log=None,
) -> np.ndarray:
    """Run ``session`` over ``rgb`` (float32 HxWx3 in [0, 1]) tile by tile."""
    height, width = rgb.shape[:2]
    tile = max(64, int(tile))
    overlap = max(0, min(int(overlap), tile // 2 - 1))
    step = max(1, tile - overlap)

    out = np.zeros((height * scale, width * scale, 3), np.float32)
    weight = np.zeros((height * scale, width * scale, 1), np.float32)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    use_half = fp16 and "float16" in str(input_meta.type).lower()

    tiles = 0
    for y0 in range(0, height, step):
        for x0 in range(0, width, step):
            y1 = min(y0 + tile, height)
            x1 = min(x0 + tile, width)
            patch = rgb[y0:y1, x0:x1, :]
            feed = np.ascontiguousarray(patch.transpose(2, 0, 1)[None])
            feed = feed.astype(np.float16) if use_half else feed.astype(np.float32)
            produced = session.run(None, {input_name: feed})[0]
            result = np.asarray(produced[0]).transpose(1, 2, 0).astype(np.float32)
            oy0, ox0 = y0 * scale, x0 * scale
            oy1, ox1 = oy0 + result.shape[0], ox0 + result.shape[1]
            window = feather_window(result.shape[0], result.shape[1], overlap * scale)[..., None]
            out[oy0:oy1, ox0:ox1] += result * window
            weight[oy0:oy1, ox0:ox1] += window
            tiles += 1
    if log:
        log(f"       {tiles} tiles of {tile}px (overlap {overlap}px, {input_meta.type if hasattr(input_meta, 'type') else '?'})")
    out /= np.maximum(weight, 1e-6)
    return np.clip(out, 0.0, 1.0)
