"""Backends that talk to native binaries, model runtimes and the OS.

Each module here wraps one class of external dependency and converts its failure
modes into :class:`imgen.registry.StageError` with an actionable message:

``ncnn``     Real-ESRGAN / Real-CUGAN / waifu2x ncnn-vulkan executables
``onnx_sr``  onnxruntime, tiled and feathered
``jxr``      the WIC-based JPEG XR encoder plus its inspect/verify helpers
``external`` the generic "run this program" helper every backend shares
"""

from __future__ import annotations

__all__ = ["ncnn", "onnx_sr", "jxr", "external"]
