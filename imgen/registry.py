"""Stage registry — the mechanism that makes a pipeline pluggable.

A *stage* is a plain function decorated with :func:`stage`. The decorator
records what the stage needs, what it produces, which parameters it accepts and
which external capability it depends on. The runner uses that metadata to

* validate a pipeline before running it (plane chain must line up),
* list/describe stages for the CLI,
* fail early with an actionable message when a required tool is missing.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .state import PLANES

__all__ = ["Stage", "stage", "get", "all_stages", "has", "load_builtin_stages", "StageError"]

#: Capabilities a stage may declare in ``needs``; doctor/CLI maps them to tools.
CAPABILITIES = ("ncnn", "onnx", "jxr", "ffmpeg", "torch", "network")

_ORDER = itertools.count()
_REGISTRY: dict[str, "Stage"] = {}


class StageError(RuntimeError):
    """Raised when a stage cannot run (missing tool, bad plane, bad parameter)."""


@dataclass(frozen=True)
class Stage:
    name: str
    func: Callable
    requires: tuple[str, ...]
    produces: tuple[str, ...]
    params: dict[str, Any]
    doc: str
    needs: tuple[str, ...] = ()
    kind: str = "process"  # source | process | sink
    order: int = field(default=0, compare=False)

    @property
    def summary(self) -> str:
        """First non-empty line of the docstring."""
        for line in (self.doc or "").splitlines():
            line = line.strip()
            if line:
                return line
        return "(no description)"

    def with_params(self, overrides: dict[str, Any]) -> "Stage":
        if not overrides:
            return self
        unknown = [k for k in overrides if k not in self.params]
        if unknown:
            raise StageError(
                f"{self.name}: unknown parameter(s) {', '.join(sorted(unknown))}; "
                f"known: {', '.join(sorted(self.params)) or '(none)'}"
            )
        return Stage(
            name=self.name,
            func=self.func,
            requires=self.requires,
            produces=self.produces,
            params={**self.params, **overrides},
            doc=self.doc,
            needs=self.needs,
            kind=self.kind,
            order=self.order,
        )


def stage(
    name: str,
    *,
    requires: Iterable[str] = ("rgb",),
    produces: Iterable[str] = ("rgb",),
    params: dict[str, Any] | None = None,
    needs: Iterable[str] = (),
    kind: str = "process",
    doc: str | None = None,
):
    """Register a pipeline stage.

    >>> @stage("sharpen.unsharp", requires=("rgb",), produces=("rgb",),
    ...        params={"amount": 0.5})
    ... def unsharp(frame, ctx, amount=0.5):
    ...     ...
    """
    req, prod = tuple(requires), tuple(produces)
    for plane in req + prod:
        if plane not in PLANES:
            raise ValueError(f"{name}: unknown plane {plane!r}; known: {', '.join(PLANES)}")
    for cap in needs:
        if cap not in CAPABILITIES:
            raise ValueError(f"{name}: unknown capability {cap!r}; known: {', '.join(CAPABILITIES)}")
    if kind not in ("source", "process", "sink"):
        raise ValueError(f"{name}: kind must be source|process|sink")

    def decorator(func: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"stage {name!r} is already registered")
        _REGISTRY[name] = Stage(
            name=name,
            func=func,
            requires=req,
            produces=prod,
            params=dict(params or {}),
            doc=(doc if doc is not None else (func.__doc__ or "")),
            needs=tuple(needs),
            kind=kind,
            order=next(_ORDER),
        )
        return func

    return decorator


def get(name: str) -> Stage:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none loaded)"
        raise StageError(f"unknown stage {name!r}; available: {known}")
    return _REGISTRY[name]


def has(name: str) -> bool:
    return name in _REGISTRY


def all_stages() -> list[Stage]:
    """Registered stages, in registration (dependency-friendly) order."""
    return sorted(_REGISTRY.values(), key=lambda s: s.order)


_LOADED = False


def load_builtin_stages() -> None:
    """Import the bundled stage modules exactly once.

    Import order matters only for the ``imgen stages`` listing, because each
    module registers its stages at import time.
    """
    global _LOADED
    if _LOADED:
        return
    from . import stages  # noqa: F401  (registers everything)

    _LOADED = True
