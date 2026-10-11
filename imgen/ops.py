"""Pure-numpy image operations — no OpenCV, no SciPy.

All helpers take float32 arrays shaped ``(H, W)`` or ``(H, W, 3)`` and treat the
first two axes as spatial. Blurs are integral-image based, so their cost is
independent of radius: a 121-pixel flat-region probe on a 4K frame costs the
same as a 3-pixel one.

The functions mirror the OpenCV calls the original research scripts used
(``cv2.blur``, ``cv2.GaussianBlur``, ``cv2.resize`` with INTER_AREA/LINEAR,
``cv2.morphologyEx`` with an ellipse kernel, the guided filter) so that results
stay comparable without requiring OpenCV.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "as_float32",
    "box_mean",
    "gaussian_blur",
    "guided_filter",
    "morph_dilate",
    "morph_erode",
    "morph_open",
    "morph_close",
    "resize",
    "smoothstep",
    "clip01",
    "percentiles",
]


def as_float32(a) -> np.ndarray:
    """Return ``a`` as a C-contiguous float32 array (no copy when already so)."""
    arr = np.asarray(a)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


# --------------------------------------------------------------------------- #
# blurs
# --------------------------------------------------------------------------- #
def _box1d(a: np.ndarray, r: int, axis: int) -> np.ndarray:
    """Moving average of window ``2r+1`` along ``axis`` with edge clamping."""
    if r < 1:
        return a
    n = a.shape[axis]
    k = 2 * r + 1
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    padded = np.pad(a, pad, mode="edge")

    # float64 accumulation: a float32 running sum loses precision on an 8K row.
    c = np.cumsum(padded, axis=axis, dtype=np.float64)
    zshape = list(c.shape)
    zshape[axis] = 1
    c = np.concatenate([np.zeros(zshape, np.float64), c], axis=axis)

    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    hi[axis] = slice(k, k + n)
    lo[axis] = slice(0, n)
    out = (c[tuple(hi)] - c[tuple(lo)]) / float(k)
    return out.astype(np.float32)


def box_mean(a, size: int) -> np.ndarray:
    """Uniform (box) blur — equivalent to ``cv2.blur(a, (size, size))``."""
    arr = as_float32(a)
    size = int(size)
    if size <= 1:
        return arr
    if size % 2 == 0:
        size += 1
    r = size // 2
    return _box1d(_box1d(arr, r, 0), r, 1)


def _convolve1d(a: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Separable 1-D convolution with edge clamping."""
    n = a.shape[axis]
    radius = len(kernel) // 2
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(a, pad, mode="edge")
    out = np.zeros_like(a)
    for index, weight in enumerate(kernel):
        window = [slice(None)] * a.ndim
        window[axis] = slice(index, index + n)
        out += float(weight) * padded[tuple(window)]
    return out


def gaussian_blur(a, sigma: float) -> np.ndarray:
    """Gaussian blur.

    Small sigmas use a true truncated Gaussian kernel (accuracy matters for the
    unsharp mask, where the kernel *is* the effect). Large sigmas switch to a
    three-pass box approximation with mixed widths, which costs the same at any
    radius — the flat gate needs a 121-pixel probe on 4K frames and cannot
    afford 100+ taps per axis.
    """
    arr = as_float32(a)
    if sigma is None or sigma <= 0:
        return arr
    sigma = float(sigma)
    if sigma <= 16.0:
        radius = max(1, int(np.ceil(3.0 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
        kernel /= kernel.sum()
        return _convolve1d(_convolve1d(arr, kernel, 0), kernel, 1)
    # Three box passes whose widths sum to the requested variance (the classic
    # "ideal width + parity correction" construction), instead of three equal
    # passes, which are systematically a few percent too wide.
    passes = 3
    ideal = np.sqrt(12.0 * sigma * sigma / passes + 1.0)
    w1 = int(np.floor(ideal))
    if w1 % 2 == 0:
        w1 -= 1
    w1 = max(1, w1)
    w2 = w1 + 2
    m_ideal = (12.0 * sigma * sigma - passes * w1 * w1 - 4.0 * passes * w1 - 3.0 * passes) / (-4.0 * w1 - 4.0)
    m = int(round(m_ideal))
    m = max(0, min(passes, m))
    out = arr
    for width in [w1] * m + [w2] * (passes - m):
        out = box_mean(out, width)
    return out


def guided_filter(guide, src, radius: int = 10, eps: float = 2e-4) -> np.ndarray:
    """Grayscale guided filter (He et al.) — edge-aware signal refinement.

    Used to lock a low-resolution semantic highlight map onto the exact
    high-resolution geometry of the source image.
    """
    g = as_float32(guide)
    s = as_float32(src)
    size = 2 * int(radius) + 1
    mean_g = box_mean(g, size)
    mean_s = box_mean(s, size)
    cov = box_mean(g * s, size) - mean_g * mean_s
    var = box_mean(g * g, size) - mean_g * mean_g
    a = cov / (var + float(eps))
    b = mean_s - a * mean_g
    return box_mean(a, size) * g + box_mean(b, size)


# --------------------------------------------------------------------------- #
# binary morphology (ellipse kernels approximated by window means)
# --------------------------------------------------------------------------- #
# A disc of diameter k covers pi/4 ~= 0.785 of its bounding square; using a
# slightly lower fraction keeps erosion tolerant of the square-vs-disc
# difference while still removing thin structures.
_ELLIPSE_FILL = 0.72


def _check_odd(size: int) -> int:
    size = int(size)
    if size < 1:
        raise ValueError("kernel size must be >= 1")
    return size + 1 if size % 2 == 0 else size


def morph_erode(binary, size: int) -> np.ndarray:
    """Erode a binary/float mask with an ellipse-ish kernel of diameter ``size``."""
    size = _check_odd(size)
    m = box_mean(as_float32(binary), size)
    return (m >= _ELLIPSE_FILL).astype(np.float32)


def morph_dilate(binary, size: int) -> np.ndarray:
    """Dilate a binary/float mask with an ellipse-ish kernel."""
    size = _check_odd(size)
    m = box_mean(as_float32(binary), size)
    return (m > 1e-6).astype(np.float32)


def morph_open(binary, size: int) -> np.ndarray:
    """Opening = erode then dilate: drops specks and thin protrusions."""
    return morph_dilate(morph_erode(binary, size), size)


def morph_close(binary, size: int) -> np.ndarray:
    """Closing = dilate then erode: fills pinholes."""
    return morph_erode(morph_dilate(binary, size), size)


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #
def _linear1d(a: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    n_in = a.shape[axis]
    if n_in == n_out:
        return a
    pos = (np.arange(n_out, dtype=np.float32) + 0.5) * (n_in / n_out) - 0.5
    pos = np.clip(pos, 0.0, n_in - 1.0)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    wshape = [1] * a.ndim
    wshape[axis] = n_out
    w = (pos - i0).reshape(wshape).astype(np.float32)
    return np.take(a, i0, axis=axis) * (1.0 - w) + np.take(a, i1, axis=axis) * w


def _area1d(a: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Exact area average — equivalent to ``cv2.INTER_AREA``."""
    n_in = a.shape[axis]
    if n_in == n_out:
        return a
    idx = np.arange(n_out, dtype=np.int64)
    lo = np.floor(idx * n_in / n_out).astype(np.int64)
    hi = np.maximum(np.ceil((idx + 1) * n_in / n_out).astype(np.int64), lo + 1)
    hi = np.minimum(hi, n_in)
    c = np.cumsum(a, axis=axis, dtype=np.float64)
    zshape = list(c.shape)
    zshape[axis] = 1
    c = np.concatenate([np.zeros(zshape, np.float64), c], axis=axis)
    counts = (hi - lo).astype(np.float64)
    cshape = [1] * a.ndim
    cshape[axis] = n_out
    out = (np.take(c, hi, axis=axis) - np.take(c, lo, axis=axis)) / counts.reshape(cshape)
    return out.astype(np.float32)


def resize(a, size, interp: str = "auto") -> np.ndarray:
    """Resize to ``(width, height)``.

    ``interp``: ``"auto"`` (area when shrinking, linear when growing),
    ``"area"`` or ``"linear"``.
    """
    arr = as_float32(a)
    width, height = int(size[0]), int(size[1])
    h, w = arr.shape[:2]
    if (w, h) == (width, height):
        return arr
    shrinking = width < w * 0.75 or height < h * 0.75
    use_area = interp == "area" or (interp == "auto" and shrinking)
    step = _area1d if use_area else _linear1d
    out = step(arr, height, 0) if height != h else arr
    out = step(out, width, 1) if width != w else out
    return as_float32(out)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def smoothstep(edge0: float, edge1: float, value) -> np.ndarray:
    """Hermite ramp between two edges (0 below, 1 above)."""
    v = np.asarray(value, dtype=np.float32)
    if edge1 == edge0:
        return (v >= edge1).astype(np.float32)
    t = np.clip((v - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def clip01(a) -> np.ndarray:
    return np.clip(as_float32(a), 0.0, 1.0)


def percentiles(a, qs=(0, 1, 50, 90, 95, 99, 99.9, 100)) -> dict:
    """Percentile summary of a finite array, as a dict keyed by the percentile."""
    arr = np.asarray(a, dtype=np.float32).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {float(q): float("nan") for q in qs}
    vals = np.percentile(arr, qs)
    return {float(q): float(v) for q, v in zip(qs, vals)}
