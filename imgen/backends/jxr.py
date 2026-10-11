"""JPEG XR delivery through the Windows Imaging Component.

Windows composites HDR in **scRGB with FP16 precision**, and its HDR wallpaper
path only accepts JPEG XR, so ``.jxr`` is the one format the OS renders as HDR
without a third-party viewer. The encoder used here is the small WIC-based CLI
that ships with the HDR-JXR toolchain:

    jxr-encode-wic.exe WIDTH HEIGHT input.f32 output.jxr QUALITY

``input.f32`` is packed little-endian RGB float32, row-major — which is exactly
what ``np.ndarray.astype('<f4').tofile()`` writes, and what
:func:`imgen.colors.nits_to_scrgb` produces (1.0 == 80 cd/m²).

The optional ``jxr-inspect`` / ``verify-jxr`` helpers re-open the file through
WIC, so a delivery stage can assert "this decodes, at this size, as scRGB
half-float" instead of hoping.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..config import Tools
from ..registry import StageError
from . import external

__all__ = ["encode_scrgb", "inspect", "verify", "write_raw_float"]


def write_raw_float(scrgb: np.ndarray, path: Path) -> Path:
    """Write packed little-endian RGB float32 (the WIC encoder's input format)."""
    arr = np.ascontiguousarray(scrgb, dtype="<f4")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise StageError(f"expected an (H, W, 3) scRGB array, got {arr.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(str(path))
    return path


def encode_scrgb(
    scrgb: np.ndarray,
    output: Path,
    tools: Tools,
    quality: float = 1.0,
    raw_path: Path | None = None,
) -> Path:
    """Encode an scRGB float image to ``.jxr`` (64bppRGBAHalf/scRGB)."""
    if not tools.jxr_encode or not Path(tools.jxr_encode).is_file():
        tools.require("jxr")
    height, width = scrgb.shape[:2]
    raw = write_raw_float(scrgb, raw_path or output.with_suffix(".f32"))
    external.run_checked(
        [str(tools.jxr_encode), str(width), str(height), str(raw), str(output), str(quality)],
        timeout=1800.0,
    )
    if raw_path is None:
        raw.unlink(missing_ok=True)
    if not output.is_file():
        raise StageError(f"JXR encoder produced no file at {output}")
    return output


def inspect(path: Path, tools: Tools) -> str:
    """Human-readable summary of a JXR file (empty string when unavailable)."""
    if not tools.jxr_inspect:
        return ""
    result = external.run([str(tools.jxr_inspect), str(path)], timeout=300.0)
    return result.output


def verify(path: Path, tools: Tools) -> tuple[bool, str]:
    """Decode the file through WIC and report ``(ok, message)``."""
    if not tools.jxr_verify:
        return (True, "verify-jxr.exe not configured; skipped")
    result = external.run([str(tools.jxr_verify), str(path)], timeout=300.0)
    text = result.output
    ok = result.ok and not re.search(r"\b(fail|error|invalid)\b", text, re.IGNORECASE)
    return (ok, text.strip() or f"exit {result.returncode}")
