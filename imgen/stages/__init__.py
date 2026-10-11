"""Bundled stages.

Importing this package registers every stage in the process-wide registry
(:mod:`imgen.registry`). The order below is the order ``imgen stages`` prints,
and it mirrors the order stages normally appear in a pipeline.
"""

from __future__ import annotations

from . import s_analyze, s_deband, s_denoise, s_encode, s_hdr, s_io, s_sharpen, s_upscale  # noqa: F401

__all__ = [
    "s_io",
    "s_denoise",
    "s_deband",
    "s_sharpen",
    "s_upscale",
    "s_hdr",
    "s_encode",
    "s_analyze",
]
