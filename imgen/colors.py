"""Colour science for the SDR -> scene-linear -> scRGB/PQ path.

Conventions used across the toolkit
-----------------------------------
* ``rgb``      : 8-bit-derived sRGB, float32 in [0, 1] — display-referred SDR.
* ``linear``   : scene-linear sRGB primaries (D65), float32, 1.0 = diffuse white.
* ``nits``     : absolute display luminance, float32, 1.0 = 1 cd/m².
* ``scrgb``    : Windows scRGB, float32 — **1.0 = 80 cd/m²** (BT.2100/Windows),
                 so ``scrgb = nits / 80``.

The 80-nit scRGB scale is what the Windows HDR compositor and the JPEG XR
half-float container expect; it is *not* the same as normalising to peak.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "LUMA",
    "SCRGB_NITS",
    "srgb_to_linear",
    "linear_to_srgb",
    "luma",
    "smoothstep",
    "LogC3",
    "nits_to_scrgb",
    "scrgb_to_nits",
    "nits_to_pq",
    "pq_to_nits",
    "bt2390_tonemap",
]

#: Rec.709 / sRGB luminance weights.
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

#: Windows scRGB reference: 1.0 == 80 cd/m².
SCRGB_NITS = 80.0


def srgb_to_linear(value) -> np.ndarray:
    """sRGB electro-optical transfer function (the exact piecewise curve)."""
    v = np.asarray(value, dtype=np.float32)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(value) -> np.ndarray:
    """Inverse of :func:`srgb_to_linear`."""
    v = np.asarray(value, dtype=np.float32)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(np.maximum(v, 0.0), 1 / 2.4) - 0.055).astype(
        np.float32
    )


def luma(rgb) -> np.ndarray:
    """Rec.709 luminance of linear RGB (last axis is colour)."""
    return np.asarray(rgb, dtype=np.float32) @ LUMA


def smoothstep(edge0: float, edge1: float, value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32) if edge1 == edge0 else (
        lambda t: t * t * (3.0 - 2.0 * t)
    )(np.clip((np.asarray(value, dtype=np.float32) - edge0) / (edge1 - edge0), 0.0, 1.0))


class LogC3:
    """ARRI LogC3 (EI 800) transfer function.

    LumiPic — and the HDR LoRAs it derives from — emit values in LogC3. Mapping
    the model output back to scene-linear is what makes the "how many nits
    should this pixel be" arithmetic meaningful.

    Verified anchors (see ``tests/test_ops.py``): ``0.18 -> 0.391``,
    ``1.0 -> 0.5706``, ``10.0 -> 0.7384``, ``55.1 -> 1.0``.
    """

    A = 5.555556
    B = 0.052272
    C = 0.247190
    D = 0.385537
    E = 5.367655
    F = 0.092809
    CUT = 0.010591
    CUT_LOG = E * CUT + F  # 0.149658...

    @classmethod
    def compress(cls, linear) -> np.ndarray:
        """Scene-linear -> LogC3 [0, 1]."""
        lin = np.maximum(np.asarray(linear, dtype=np.float32), 0.0)
        log_val = cls.C * np.log10(cls.A * lin + cls.B) + cls.D
        lin_val = cls.E * lin + cls.F
        return np.where(lin > cls.CUT, log_val, lin_val).astype(np.float32)

    @classmethod
    def decompress(cls, logc) -> np.ndarray:
        """LogC3 [0, 1] -> scene-linear."""
        v = np.clip(np.asarray(logc, dtype=np.float32), 0.0, 1.0)
        from_log = (np.power(10.0, (v - cls.D) / cls.C) - cls.B) / cls.A
        from_lin = (v - cls.F) / cls.E
        return np.where(v >= cls.CUT_LOG, from_log, from_lin).astype(np.float32)


def nits_to_scrgb(nits) -> np.ndarray:
    """Absolute luminance -> Windows scRGB (1.0 = 80 cd/m²)."""
    return (np.asarray(nits, dtype=np.float32) / SCRGB_NITS).astype(np.float32)


def scrgb_to_nits(scrgb) -> np.ndarray:
    """Windows scRGB -> absolute luminance in cd/m²."""
    return (np.asarray(scrgb, dtype=np.float32) * SCRGB_NITS).astype(np.float32)


# --------------------------------------------------------------------------- #
# PQ (SMPTE ST 2084) — needed for HDR10 / AVIF / gain-map deliverables
# --------------------------------------------------------------------------- #
_PQ_M1 = 2610.0 / 16384.0
_PQ_M2 = 2523.0 / 4096.0 * 128.0
_PQ_C1 = 3424.0 / 4096.0
_PQ_C2 = 2413.0 / 4096.0 * 32.0
_PQ_C3 = 2392.0 / 4096.0 * 32.0


def nits_to_pq(nits, peak: float = 10000.0):
    """Absolute luminance -> PQ signal [0, 1] (1.0 == ``peak`` cd/m²)."""
    y = np.clip(np.asarray(nits, dtype=np.float32) / float(peak), 0.0, 1.0)
    ym = np.power(y, _PQ_M1)
    num = _PQ_C1 + _PQ_C2 * ym
    den = 1.0 + _PQ_C3 * ym
    return np.power(num / den, _PQ_M2).astype(np.float32)


def pq_to_nits(pq, peak: float = 10000.0):
    """PQ signal [0, 1] -> absolute luminance in cd/m²."""
    e = np.clip(np.asarray(pq, dtype=np.float32), 0.0, 1.0)
    em = np.power(e, 1.0 / _PQ_M2)
    num = np.maximum(em - _PQ_C1, 0.0)
    den = _PQ_C2 - _PQ_C3 * em
    return (np.power(num / den, 1.0 / _PQ_M1) * float(peak)).astype(np.float32)


def bt2390_tonemap(nits, src_peak: float, dst_peak: float):
    """ITU-R BT.2390 EETF: map ``src_peak`` -> ``dst_peak`` with a Hermite knee.

    Only needed when a deliverable must be squeezed into a lower peak (for
    example a 1000-nit master shown on a 600-nit panel). Below the knee
    ``ks = 1.5*dst - 0.5*src`` values pass through untouched; above it a cubic
    Hermite blends smoothly onto ``dst_peak``.
    """
    y = np.asarray(nits, dtype=np.float32)
    if dst_peak >= src_peak:
        return y
    ks = 1.5 * float(dst_peak) - 0.5 * float(src_peak)
    e1 = np.clip((y - ks) / max(float(src_peak) - ks, 1e-6), 0.0, 1.0)
    hermite = (2.0 * e1**3 - 3.0 * e1**2 + 1.0) * ks + (-2.0 * e1**3 + 3.0 * e1**2) * float(dst_peak)
    return np.where(y < ks, y, hermite).astype(np.float32)
