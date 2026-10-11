"""Pipeline definition, validation and execution.

A pipeline is JSON:

```json
{
  "name": "anime-4k-hdr",
  "description": "...",
  "stages": [
    {"stage": "io.load"},
    {"stage": "upscale.ncnn", "params": {"tool": "realesrgan", "model": "realesrgan-x4plus-anime", "scale": 4}},
    {"stage": "hdr.map", "params": {"peak_nits": 800}}
  ]
}
```

Inserting, removing or reordering stages is the whole point: edit the list, or
use ``imgen add`` / ``imgen remove`` / ``imgen move``, which rewrite the file for
you. :func:`Pipeline.validate` walks the plane chain and reports exactly which
stage breaks it, so a stale pipeline fails loudly before any pixel is touched.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .config import Tools, load_tools
from .registry import Stage, StageError, get, load_builtin_stages
from .state import PLANES, Frame

__all__ = ["StageCall", "Pipeline", "RunContext", "load_pipeline", "run_pipeline", "validate_pipeline"]


@dataclass
class StageCall:
    """One entry of the pipeline: a stage name plus its parameter overrides."""

    stage: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None
    enabled: bool = True
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "StageCall":
        if not isinstance(data, dict) or "stage" not in data:
            raise SystemExit(f"each pipeline entry must look like {{'stage': 'name', ...}}; got {data!r}")
        return cls(
            stage=data["stage"],
            params=dict(data.get("params") or {}),
            label=data.get("label") or data.get("as"),
            enabled=bool(data.get("enabled", True)),
            note=data.get("note", ""),
        )

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"stage": self.stage}
        if self.params:
            out["params"] = self.params
        if self.label:
            out["label"] = self.label
        if not self.enabled:
            out["enabled"] = False
        if self.note:
            out["note"] = self.note
        return out

    @property
    def display(self) -> str:
        return self.label or self.stage


@dataclass
class Pipeline:
    name: str
    stages: list[StageCall]
    description: str = ""
    defaults: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    # -- (de)serialisation ------------------------------------------------- #
    @classmethod
    def from_dict(cls, data: dict, source: Path | None = None) -> "Pipeline":
        if "stages" not in data:
            raise SystemExit("pipeline JSON has no 'stages' list")
        return cls(
            name=data.get("name") or (source.stem if source else "pipeline"),
            description=data.get("description", ""),
            defaults=dict(data.get("defaults") or {}),
            stages=[StageCall.from_dict(item) for item in data["stages"]],
            source=source,
        )

    def to_dict(self) -> dict:
        data: dict[str, Any] = {"name": self.name}
        if self.description:
            data["description"] = self.description
        if self.defaults:
            data["defaults"] = self.defaults
        data["stages"] = [call.to_dict() for call in self.stages]
        return data

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path or self.source or f"{self.name}.json")
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.source = target
        return target

    # -- introspection ----------------------------------------------------- #
    def specs(self, include_disabled: bool = False) -> list[tuple[int, StageCall, Stage]]:
        """``(index, call, spec)`` with the call's overrides already merged in.

        Merging here (rather than at call time) means validation, ``describe``
        and execution all see exactly the parameters that will be used — a typo
        in a pipeline file is caught before the run, not by a crash in a stage.
        """
        load_builtin_stages()
        out = []
        for index, call in enumerate(self.stages):
            if not call.enabled and not include_disabled:
                continue
            base = get(call.stage)
            out.append((index, call, base.with_params(call.params) if call.params else base))
        return out

    def validate(self, initial_planes: Sequence[str] = ("rgb",)) -> list[str]:
        """Return a list of problems; empty means the chain is consistent."""
        problems: list[str] = []
        available = set(initial_planes)
        seen_source = False
        for index, call, spec in self.specs():
            missing = [p for p in spec.requires if p not in available]
            if missing:
                problems.append(
                    f"[{index}] {call.stage}: needs plane(s) {', '.join(missing)} which no earlier stage "
                    f"produces (available: {', '.join(sorted(available)) or 'none'})"
                )
            if spec.kind == "source":
                if seen_source:
                    problems.append(f"[{index}] {call.stage}: a second 'source' stage — only one is allowed")
                seen_source = True
            available.update(spec.produces)
            try:
                spec.with_params(call.params)
            except StageError as exc:
                problems.append(f"[{index}] {exc}")
        if not seen_source and "rgb" not in initial_planes:
            problems.append("no 'source' stage (e.g. io.load) and no RGB plane supplied by the caller")
        return problems

    def describe(self) -> str:
        lines = [f"pipeline: {self.name}"]
        if self.description:
            lines.append(f"  {self.description}")
        available = {"rgb"}
        for index, call, spec in self.specs(include_disabled=True):
            flag = "" if call.enabled else "  (disabled)"
            lines.append(
                f"  [{index:>2}] {call.stage:<22} {spec.kind:<7} "
                f"in={','.join(spec.requires) or '-':<18} out={','.join(spec.produces) or '-':<20}{flag}"
            )
            if call.params:
                lines.append(f"        params: {json.dumps(call.params, ensure_ascii=False)}")
            if call.note:
                lines.append(f"        note:   {call.note}")
        return "\n".join(lines)


@dataclass
class RunContext:
    """Everything a stage may need from the outside world."""

    out_dir: Path
    work_dir: Path
    tools: Tools
    log: Callable[[str], None] = print
    save_stages: bool = True
    use_cache: bool = False
    index: int = 0
    label: str = ""
    dry_run: bool = False
    env: dict[str, Any] = field(default_factory=dict)
    #: Files written by sink stages, in order (used for the run report).
    wrote: list[str] = field(default_factory=list)

    def record(self, path) -> None:
        """Note that ``path`` was written, so the report can list it."""
        self.wrote.append(str(path))

    def artifact_dir(self) -> Path:
        path = self.work_dir / "stages"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def temp(self, name: str) -> Path:
        path = self.work_dir / "tmp"
        path.mkdir(parents=True, exist_ok=True)
        return path / name

    def out(self, name: str) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self.out_dir / name


def load_pipeline(path: str | Path) -> Pipeline:
    """Read a pipeline JSON file."""
    file = Path(path)
    if not file.is_file():
        raise SystemExit(f"pipeline file not found: {file}")
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{file} is not valid JSON: {exc}") from exc
    return Pipeline.from_dict(data, source=file)


def validate_pipeline(path: str | Path) -> list[str]:
    return load_pipeline(path).validate()


def _coerce(text: str) -> Any:
    """Interpret a ``--set`` value: JSON first, then plain string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _resolve_targets(pipeline: Pipeline, target: str) -> list[int]:
    """Resolve a ``--set`` target: ``#2`` (index), a stage name, or ``*``."""
    if target in ("*", "all"):
        return list(range(len(pipeline.stages)))
    if target.startswith("#"):
        try:
            index = int(target[1:])
        except ValueError as exc:
            raise SystemExit(f"bad stage index {target!r}") from exc
        if not 0 <= index < len(pipeline.stages):
            raise SystemExit(f"stage index {index} out of range (0..{len(pipeline.stages) - 1})")
        return [index]
    hits = [i for i, call in enumerate(pipeline.stages) if call.stage == target or call.label == target]
    if not hits:
        raise SystemExit(f"no stage named {target!r} in this pipeline (use #index to disambiguate)")
    return hits


def apply_overrides(pipeline: Pipeline, overrides: Iterable[str]) -> None:
    """Apply ``--set stage.param=value`` overrides in place."""
    known: set[str] = set()
    for index, call, spec in pipeline.specs(include_disabled=True):
        known.update(spec.params)
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects stage.param=value, got {item!r}")
        target_and_param, value = item.split("=", 1)
        if "." not in target_and_param:
            raise SystemExit(f"--set expects stage.param=value, got {item!r}")
        target, param = target_and_param.rsplit(".", 1)
        for index in _resolve_targets(pipeline, target):
            call = pipeline.stages[index]
            spec = get(call.stage)
            if param not in spec.params:
                raise SystemExit(
                    f"{call.stage} has no parameter {param!r}; known: {', '.join(sorted(spec.params)) or '(none)'}"
                )
            call.params[param] = _coerce(value)
        if param not in known and target != "*":
            raise SystemExit(f"no stage in this pipeline accepts {param!r}")


def run_pipeline(
    pipeline: Pipeline | str | Path,
    input_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    work_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
    skip: Sequence[str] = (),
    only: Sequence[str] = (),
    until: str | None = None,
    save_stages: bool = True,
    use_cache: bool = False,
    tools: Tools | None = None,
    overrides: Iterable[str] = (),
    dry_run: bool = False,
) -> tuple[Frame, dict]:
    """Execute a pipeline and return ``(final_frame, report)``.

    ``skip``/``only``/``until`` accept stage names, labels or ``#index`` values.
    """
    load_builtin_stages()
    if not isinstance(pipeline, Pipeline):
        pipeline = load_pipeline(pipeline)
    if overrides:
        apply_overrides(pipeline, overrides)

    def _selected(target: str) -> set[int]:
        return set(_resolve_targets(pipeline, target))

    skip_idx: set[int] = set()
    for item in skip:
        skip_idx |= _selected(item)
    only_idx: set[int] | None = None
    if only:
        only_idx = set()
        for item in only:
            only_idx |= _selected(item)
    until_idx = max(_selected(until)) if until else None

    out_path = Path(out_dir or pipeline.defaults.get("out_dir") or "out")
    work_path = Path(work_dir or pipeline.defaults.get("work_dir") or "work")
    ctx = RunContext(
        out_dir=out_path.resolve(),
        work_dir=work_path.resolve(),
        tools=tools or load_tools(),
        log=log,
        save_stages=save_stages,
        use_cache=use_cache,
        dry_run=dry_run,
    )
    ctx.env["input"] = str(input_path) if input_path else None

    frame = Frame()
    if input_path:
        frame.meta["input_path"] = str(Path(input_path).resolve())

    problems = pipeline.validate(initial_planes=("rgb",) if input_path else ())
    if problems:
        raise SystemExit("pipeline is inconsistent:\n  " + "\n  ".join(problems))

    report: dict[str, Any] = {
        "pipeline": pipeline.name,
        "input": str(input_path) if input_path else None,
        "out_dir": str(ctx.out_dir),
        "work_dir": str(ctx.work_dir),
        "stages": [],
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for index, call, spec in pipeline.specs():
        if index in skip_idx or (only_idx is not None and index not in only_idx):
            log(f"  [{index:>2}] {call.display:<24} skipped")
            report["stages"].append({"index": index, "stage": call.stage, "status": "skipped"})
            continue
        ctx.index, ctx.label = index, call.display
        missing_caps = [cap for cap in spec.needs if cap != "network"]
        for cap in missing_caps:
            ctx.tools.require(cap)
        params = spec.params
        wrote_before = len(ctx.wrote)
        started = time.time()
        log(f"  [{index:>2}] {call.display:<24} {spec.summary}")
        try:
            result = spec.func(frame, ctx, **params)
        except TypeError as exc:
            raise SystemExit(f"stage {call.stage} rejected its parameters: {exc}") from exc
        if isinstance(result, Frame):
            frame = result
        elif result is not None and not isinstance(result, tuple):
            raise SystemExit(f"stage {call.stage} returned {type(result).__name__}, expected a Frame or None")
        elapsed = time.time() - started
        entry = {
            "index": index,
            "stage": call.stage,
            "label": call.display,
            "status": "ok",
            "seconds": round(elapsed, 3),
            "params": params,
            "notes": list(frame.notes),
            "wrote": ctx.wrote[wrote_before:],
        }
        entry["size"] = frame.size if frame.size != (0, 0) else None
        report["stages"].append(entry)
        log(f"       done in {elapsed:.2f}s  {frame.summary()}")
        if until_idx is not None and index >= until_idx:
            log(f"  --until {until} reached; stopping")
            break

    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["total_seconds"] = round(
        sum(item.get("seconds", 0) for item in report["stages"]), 3
    )
    report["notes"] = list(frame.notes)
    report["outputs"] = sorted({str(p) for item in report["stages"] for p in item.get("wrote", [])})
    report["stats"] = frame.meta.get("stats", {})
    return frame, report


def _stage_help(names: Sequence[str] | None = None) -> str:
    from .registry import all_stages

    lines = []
    for spec in all_stages():
        if names and spec.name not in names:
            continue
        lines.append(f"{spec.name:<24} {spec.kind:<7} {spec.summary()}")
        lines.append(f"{'':<24} needs: {', '.join(spec.requires) or '-'} -> {', '.join(spec.produces) or '-'}")
    return "\n".join(lines)


def stage_help(names: Sequence[str] | None = None) -> str:  # pragma: no cover - alias
    return _stage_help(names)
