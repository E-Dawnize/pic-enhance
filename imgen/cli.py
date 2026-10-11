"""Command line interface.

    imgen stages                      list every registered stage
    imgen describe hdr.map            parameters and defaults for one stage
    imgen validate pipelines/x.json   check the plane chain without running it
    imgen run pipelines/x.json -i in.png -o out
    imgen add pipelines/x.json sharpen.unsharp --after upscale.ncnn
    imgen remove pipelines/x.json sharpen.unsharp
    imgen move pipelines/x.json sharpen.unsharp --to 4
    imgen toggle pipelines/x.json deband.gradient
    imgen doctor                      what is installed on this machine
    imgen new my-pipeline.json        write a starter pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import load_tools
from .registry import all_stages, get, load_builtin_stages
from .runner import Pipeline, load_pipeline, run_pipeline

__all__ = ["main"]

PROG = "imgen"


def _print_stages(pattern: str | None = None) -> None:
    stages = all_stages()
    if pattern:
        needle = pattern.lower()
        stages = [s for s in stages if needle in s.name.lower() or needle in s.summary.lower()]
    if not stages:
        print("no stages matched")
        return
    width = max(len(s.name) for s in stages) + 2
    print(f"{'stage':<{width}}{'kind':<9}{'needs':<10}description")
    print("-" * 110)
    for spec in stages:
        needs = ",".join(spec.needs) or "-"
        print(f"{spec.name:<{width}}{spec.kind:<9}{needs:<10}{spec.summary}")
    print(f"\n{len(stages)} stage(s). `imgen describe <stage>` for parameters.")


def _print_stage(name: str, as_json: bool = False) -> None:
    spec = get(name)
    if as_json:
        print(
            json.dumps(
                {
                    "name": spec.name,
                    "kind": spec.kind,
                    "requires": list(spec.requires),
                    "produces": list(spec.produces),
                    "needs": list(spec.needs),
                    "params": spec.params,
                    "doc": spec.doc,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    print(f"{spec.name}  [{spec.kind}]")
    print(f"  {spec.summary}")
    for line in (spec.doc or "").splitlines()[1:]:
        if line.strip():
            print(f"  {line.rstrip()}")
    print(f"\n  requires: {', '.join(spec.requires) or '(nothing)'}")
    print(f"  produces: {', '.join(spec.produces) or '(nothing — sink)'}")
    print(f"  needs:    {', '.join(spec.needs) or '(nothing external)'}")
    print("\n  parameters:")
    if not spec.params:
        print("    (none)")
    for key, value in spec.params.items():
        print(f"    {key:<24} default = {value!r}")


def _cmd_stages(args) -> int:
    _print_stages(args.filter)
    return 0


def _cmd_describe(args) -> int:
    _print_stage(args.stage, as_json=args.json)
    return 0


def _cmd_validate(args) -> int:
    pipeline = load_pipeline(args.pipeline)
    problems = pipeline.validate()
    print(pipeline.describe())
    print()
    if problems:
        print("INVALID:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("chain is consistent: every stage finds the planes it needs.")
    return 0


def _cmd_run(args) -> int:
    tools = load_tools(args.config)
    if args.input:
        candidate = Path(args.input)
        if not candidate.is_file():
            print(f"input not found: {candidate}", file=sys.stderr)
            return 2
    frame, report = run_pipeline(
        args.pipeline,
        input_path=args.input,
        out_dir=args.out,
        work_dir=args.work,
        skip=args.skip or (),
        only=args.only or (),
        until=args.until,
        save_stages=not args.no_save_stages,
        use_cache=args.cache,
        tools=tools,
        overrides=args.set or (),
        dry_run=args.dry_run,
    )
    print()
    print(f"pipeline '{report['pipeline']}' finished in {report['total_seconds']:.1f}s")
    if report.get("stats"):
        print("stats:")
        for key, value in report["stats"].items():
            if isinstance(value, float):
                print(f"  {key:<22}{value:.4f}")
            elif isinstance(value, dict):
                print(f"  {key:<22}{json.dumps(value, ensure_ascii=False)}")
            else:
                print(f"  {key:<22}{value}")
    outputs = report.get("outputs", [])
    if outputs:
        print("files:")
        for item in outputs:
            print(f"  {item}")
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"report: {target}")
    return 0


def _cmd_add(args) -> int:
    pipeline = load_pipeline(args.pipeline)
    get(args.stage)  # raises for unknown stages
    params = json.loads(args.params) if args.params else {}
    from .runner import StageCall

    call = StageCall(stage=args.stage, params=params, label=args.label, enabled=not args.disabled, note=args.note or "")
    if args.index is not None:
        index = max(0, min(int(args.index), len(pipeline.stages)))
    elif args.after:
        index = max(i for i, c in enumerate(pipeline.stages) if c.stage == args.after or c.label == args.after) + 1
    elif args.before:
        index = min(i for i, c in enumerate(pipeline.stages) if c.stage == args.before or c.label == args.before)
    else:
        index = len(pipeline.stages)
    pipeline.stages.insert(index, call)
    problems = pipeline.validate()
    if problems and not args.force:
        print("refusing to write an inconsistent pipeline (use --force to override):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    target = pipeline.save(args.output or args.pipeline)
    print(f"inserted {args.stage} at index {index} -> {target}")
    return 0


def _resolve_index(pipeline: Pipeline, target: str) -> list[int]:
    if target.startswith("#"):
        return [int(target[1:])]
    return [i for i, c in enumerate(pipeline.stages) if c.stage == target or c.label == target]


def _cmd_remove(args) -> int:
    pipeline = load_pipeline(args.pipeline)
    hits = _resolve_index(pipeline, args.target)
    if not hits:
        print(f"no stage matching {args.target!r}", file=sys.stderr)
        return 1
    for index in sorted(hits, reverse=True):
        removed = pipeline.stages.pop(index)
        print(f"removed [{index}] {removed.stage}")
    target = pipeline.save(args.output or args.pipeline)
    problems = pipeline.validate()
    if problems:
        print("warning: the remaining chain is inconsistent:")
        for problem in problems:
            print(f"  - {problem}")
    print(f"pipeline saved to {target}")
    return 0


def _cmd_move(args) -> int:
    pipeline = load_pipeline(args.pipeline)
    hits = _resolve_index(pipeline, args.target)
    if len(hits) != 1:
        print(f"{args.target!r} matched {len(hits)} stages; use #index", file=sys.stderr)
        return 1
    call = pipeline.stages.pop(hits[0])
    pipeline.stages.insert(max(0, min(int(args.to), len(pipeline.stages))), call)
    target = pipeline.save(args.output or args.pipeline)
    print(f"moved {call.stage} to index {args.to} -> {target}")
    problems = pipeline.validate()
    if problems:
        print("warning: the chain is now inconsistent:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    return 0


def _cmd_toggle(args) -> int:
    pipeline = load_pipeline(args.pipeline)
    hits = _resolve_index(pipeline, args.target)
    if not hits:
        print(f"no stage matching {args.target!r}", file=sys.stderr)
        return 1
    for index in hits:
        call = pipeline.stages[index]
        call.enabled = not call.enabled
        print(f"[{index}] {call.stage}: {'enabled' if call.enabled else 'disabled'}")
    target = pipeline.save(args.output or args.pipeline)
    print(f"pipeline saved to {target}")
    return 0


def _cmd_doctor(args) -> int:
    from . import doctor

    tools = load_tools(args.config)
    rows = doctor.report(tools)
    if args.json:
        print(json.dumps([{"item": a, "status": b, "detail": c} for a, b, c in rows], indent=2, ensure_ascii=False))
    else:
        print(doctor.format_report(rows))
    return 0


def _cmd_new(args) -> int:
    template = {
        "name": Path(args.path).stem,
        "description": "Starter pipeline: denoise -> 2x -> 2x -> sharpening -> save.",
        "defaults": {"out_dir": "out", "work_dir": "work"},
        "stages": [
            {"stage": "io.load"},
            {"stage": "denoise.nlmeans", "params": {"strength": 0.008, "search_stride": 2}},
            {
                "stage": "upscale.ncnn",
                "params": {"tool": "realesrgan", "model": "realesrgan-x4plus-anime", "scale": 2, "tile": 0},
            },
            {"stage": "sharpen.unsharp", "params": {"radius": 1.0, "amount": 0.35}},
            {"stage": "io.save", "params": {"filename": "{stem}-2x.png"}},
        ],
    }
    target = Path(args.path)
    if target.exists() and not args.force:
        print(f"{target} exists (use --force)", file=sys.stderr)
        return 1
    target.write_text(json.dumps(template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {target}")
    return 0


def _cmd_pipelines(args) -> int:
    directory = Path(args.dir or "pipelines")
    files = sorted(directory.glob("*.json"))
    if not files:
        print(f"no pipeline JSON files in {directory.resolve()}")
        return 0
    for path in files:
        try:
            pipeline = load_pipeline(path)
            problems = pipeline.validate()
            status = "ok" if not problems else f"{len(problems)} problem(s)"
            print(f"{path.name:<28}{len(pipeline.stages):>3} stages   {status:<16}{pipeline.description}")
        except SystemExit as exc:
            print(f"{path.name:<28}  ?  {exc}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description="Pluggable image-enhancement pipeline (restore / upscale / HDR / encode).")
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("stages", help="list registered stages")
    p.add_argument("filter", nargs="?", help="substring filter")
    p.set_defaults(func=_cmd_stages)

    p = sub.add_parser("describe", help="show one stage's parameters")
    p.add_argument("stage")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_describe)

    p = sub.add_parser("validate", help="check a pipeline's plane chain")
    p.add_argument("pipeline")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("run", help="execute a pipeline")
    p.add_argument("pipeline")
    p.add_argument("-i", "--input", help="input image (used by io.load when it has no path)")
    p.add_argument("-o", "--out", help="output directory")
    p.add_argument("-w", "--work", help="working directory for intermediates")
    p.add_argument("--set", action="append", metavar="STAGE.PARAM=VALUE", help="override a parameter (repeatable)")
    p.add_argument("--skip", action="append", metavar="STAGE", help="skip a stage (repeatable)")
    p.add_argument("--only", action="append", metavar="STAGE", help="run only these stages (repeatable)")
    p.add_argument("--until", metavar="STAGE", help="stop after this stage")
    p.add_argument("--config", help="tools JSON (see config/tools.example.json)")
    p.add_argument("--report", help="write the run report JSON here")
    p.add_argument("--cache", action="store_true", help="reuse cached stage outputs when unchanged")
    p.add_argument("--no-save-stages", action="store_true", help="do not keep per-stage intermediates")
    p.add_argument("--dry-run", action="store_true", help="validate and describe, then stop")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("add", help="insert a stage into a pipeline")
    p.add_argument("pipeline")
    p.add_argument("stage")
    p.add_argument("--after", help="insert after this stage/label")
    p.add_argument("--before", help="insert before this stage/label")
    p.add_argument("--index", type=int, help="insert at this position")
    p.add_argument("--params", help="JSON object of parameters")
    p.add_argument("--label", help="display name for this occurrence")
    p.add_argument("--note", help="comment stored in the pipeline file")
    p.add_argument("--disabled", action="store_true", help="insert disabled")
    p.add_argument("--force", action="store_true", help="write even if the chain becomes inconsistent")
    p.add_argument("--output", help="write to a different file")
    p.set_defaults(func=_cmd_add)

    p = sub.add_parser("remove", help="delete a stage from a pipeline")
    p.add_argument("pipeline")
    p.add_argument("target", help="stage name, label, or #index")
    p.add_argument("--output")
    p.set_defaults(func=_cmd_remove)

    p = sub.add_parser("move", help="reorder a stage")
    p.add_argument("pipeline")
    p.add_argument("target")
    p.add_argument("--to", type=int, required=True)
    p.add_argument("--output")
    p.set_defaults(func=_cmd_move)

    p = sub.add_parser("toggle", help="enable/disable a stage without deleting it")
    p.add_argument("pipeline")
    p.add_argument("target")
    p.add_argument("--output")
    p.set_defaults(func=_cmd_toggle)

    p = sub.add_parser("doctor", help="check installed tools and models")
    p.add_argument("--config")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_doctor)

    p = sub.add_parser("new", help="write a starter pipeline file")
    p.add_argument("path")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_new)

    p = sub.add_parser("pipelines", help="list pipeline files and validate them")
    p.add_argument("--dir")
    p.set_defaults(func=_cmd_pipelines)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_builtin_stages()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(f"error: {exc.code}", file=sys.stderr)
            return 2
        raise
    except Exception as exc:  # noqa: BLE001 - the CLI is the user-facing boundary
        name = type(exc).__name__
        if name == "StageError":
            print(f"stage error: {exc}", file=sys.stderr)
            return 1
        raise
