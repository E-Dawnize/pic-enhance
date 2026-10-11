"""imgen — a pluggable image-enhancement pipeline (restore · upscale · HDR · encode).

Design goals
------------
* **Insertable / removable stages.** A pipeline is a JSON list of stages. Add,
  drop or reorder one line and the run changes; nothing else is coupled.
* **Dependency-light.** numpy + Pillow are the only hard requirements.
  onnxruntime is needed for ONNX models. Every external binary (ncnn-vulkan
  tools, the WIC JXR encoder, ffmpeg) is optional and resolved at runtime.
* **Honest about what it cannot do.** A stage that needs a missing tool fails
  with a clear message and a hint, instead of silently degrading.
* **Reproducible.** Every stage is deterministic given its parameters; the
  semantic-HDR stage caches its expensive output so mapping parameters can be
  re-tuned without re-running the model.

Public API
----------
>>> from imgen import run_pipeline
>>> frame, report = run_pipeline("pipelines/anime-4k-hdr.json", input_path="in.png")

Status plane vocabulary (see ``imgen.state.PLANES``): ``rgb``, ``semantic``,
``highlight``, ``nits``, ``scrgb``.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__", "run_pipeline", "load_pipeline"]


def __getattr__(name: str):  # lazy re-exports: keep `import imgen` cheap
    if name in {"run_pipeline", "load_pipeline"}:
        from . import runner

        return getattr(runner, name)
    raise AttributeError(name)
