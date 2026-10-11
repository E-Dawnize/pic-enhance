"""Analysis stages: statistics, 1:1 crop comparisons and contact sheets.

These exist because "it looked good" is not evidence. The pipeline can emit the
numbers that decide whether a grade is acceptable (peak, how much of the frame
is above 400/500 nits, how large the affected area is) and the 1:1 crops that
make a regression visible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..colors import LUMA, linear_to_srgb
from ..registry import StageError, stage
from ..state import Frame

__all__ = ["stats", "crops", "sheet"]


def _font(size: int = 18):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


@stage(
    "analyze.stats",
    requires=("rgb",),
    produces=(),
    kind="sink",
    params={"filename": "{stem}.stats.json", "verbose": True},
    doc="Write a JSON report: SDR stats, HDR luminance distribution, pipeline passes.",
)
def stats(frame: Frame, ctx, filename: str = "{stem}.stats.json", verbose: bool = True) -> Frame:
    payload = dict(frame.meta.get("stats", {}))
    if frame.rgb is not None:
        y = frame.rgb @ LUMA
        payload["sdr"] = {
            "size": list(frame.size),
            "mean_luma": float(y.mean()),
            "p99_luma": float(np.percentile(y, 99)),
            "clipped_highlights_pct": float(np.mean(np.min(frame.rgb, axis=2) >= 0.999) * 100.0),
            "clipped_shadows_pct": float(np.mean(np.max(frame.rgb, axis=2) <= 0.001) * 100.0),
        }
    payload["passes"] = frame.meta.get("passes", [])
    payload["semantic_cache"] = frame.meta.get("semantic_cache")
    payload["notes"] = frame.notes
    name = filename.format(stem=frame.meta.get("name", "image"))
    target = ctx.out(name)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ctx.record(target)
    ctx.log(f"       wrote {target}")
    if verbose:
        for key in ("peak_nits", "p99_luma_nits", "affected_pct", "strong_pct", "over_500_pct", "suppressed_pct"):
            if key in payload:
                ctx.log(f"       {key:<20} {payload[key]}")
    return frame


_ANCHORS = {
    "center": (0.5, 0.5),
    "left": (0.18, 0.5),
    "right": (0.78, 0.5),
    "upper": (0.5, 0.2),
    "lower": (0.5, 0.8),
    "background": (0.85, 0.2),
}


@stage(
    "analyze.crops",
    requires=("rgb",),
    produces=(),
    kind="sink",
    params={
        "anchors": "center,right,background",
        "size": 512,
        "reference": "",
        "filename": "{stem}.crops.png",
        "subdir": "crops",
    },
    doc="Write 1:1 crop pairs (reference | result) for the given anchors, plus a combined sheet.",
)
def crops(
    frame: Frame,
    ctx,
    anchors: str = "center,right,background",
    size: int = 512,
    reference: str = "",
    filename: str = "{stem}.crops.png",
    subdir: str = "crops",
) -> Frame:
    if frame.rgb is None:
        raise StageError("analyze.crops needs an rgb plane")
    result = np.clip(frame.rgb, 0.0, 1.0)
    height, width = result.shape[:2]
    size = int(size)

    reference_rgb = None
    if reference:
        ref_path = Path(reference)
        if not ref_path.is_file():
            raise StageError(f"reference image not found: {ref_path}")
        with Image.open(ref_path) as handle:
            reference_rgb = np.asarray(handle.convert("RGB"), np.float32) / 255.0
    else:
        source = frame.meta.get("input_path")
        if source and Path(source).is_file():
            with Image.open(source) as handle:
                reference_rgb = np.asarray(handle.convert("RGB"), np.float32) / 255.0

    panels: list[tuple[str, np.ndarray]] = []
    directory = ctx.out(str(Path(subdir) if subdir else "."))
    directory.mkdir(parents=True, exist_ok=True)
    for anchor in [a.strip() for a in anchors.split(",") if a.strip()]:
        if anchor not in _ANCHORS:
            raise StageError(f"unknown anchor {anchor!r}; known: {', '.join(_ANCHORS)}")
        fx, fy = _ANCHORS[anchor]
        x = int(min(max(fx * width - size / 2, 0), max(0, width - size)))
        y = int(min(max(fy * height - size / 2, 0), max(0, height - size)))
        patch = result[y : y + size, x : x + size, :]
        if patch.shape[0] < size or patch.shape[1] < size:
            pad = np.zeros((size, size, 3), np.float32)
            pad[: patch.shape[0], : patch.shape[1]] = patch
            patch = pad
        rows = [patch]
        if reference_rgb is not None:
            rx = int(x * reference_rgb.shape[1] / width)
            ry = int(y * reference_rgb.shape[0] / height)
            rsize = max(1, int(size * reference_rgb.shape[1] / width))
            rpatch = reference_rgb[ry : ry + rsize, rx : rx + rsize, :]
            image = Image.fromarray((np.clip(rpatch, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB").resize(
                (size, size), Image.Resampling.LANCZOS
            )
            rows.insert(0, np.asarray(image, np.float32) / 255.0)
        pair = np.concatenate(rows, axis=0)
        panels.append((anchor, pair))
        single = directory / f"{frame.meta.get('name', 'image')}-{anchor}.png"
        Image.fromarray((np.clip(pair, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB").save(single, compress_level=4)
        ctx.record(single)

    if panels:
        gap = 12
        total_w = sum(p.shape[1] for _, p in panels) + gap * (len(panels) - 1)
        total_h = max(p.shape[0] for _, p in panels) + 34
        canvas = Image.new("RGB", (total_w, total_h), (18, 18, 20))
        draw = ImageDraw.Draw(canvas)
        x = 0
        for anchor, pair in panels:
            image = Image.fromarray((np.clip(pair, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
            canvas.paste(image, (x, 34))
            draw.text((x + 6, 9), anchor, fill=(230, 230, 230), font=_font(20))
            x += pair.shape[1] + gap
        name = filename.format(stem=frame.meta.get("name", "image"))
        target = ctx.out(name)
        canvas.save(target, compress_level=4)
        ctx.record(target)
        ctx.log(f"       wrote {target}")
        frame.note(f"crops written: {', '.join(a for a, _ in panels)}")
    return frame


@stage(
    "analyze.sheet",
    requires=(),
    produces=(),
    kind="sink",
    params={"images": [], "labels": [], "width": 900, "columns": 0, "filename": "sheet.png", "subdir": ""},
    doc="Build a labelled contact sheet from any list of image files (paths may contain {out}/{work}).",
)
def sheet(
    frame: Frame,
    ctx,
    images: list | None = None,
    labels: list | None = None,
    width: int = 900,
    columns: int = 0,
    filename: str = "sheet.png",
    subdir: str = "",
) -> Frame:
    if not images:
        raise StageError("analyze.sheet needs a list of image paths")
    resolved: list[Path] = []
    for item in images:
        path = Path(str(item).format(out=ctx.out_dir, work=ctx.work_dir))
        if not path.is_absolute():
            path = (ctx.out_dir / path) if not path.is_file() else path
        if not path.is_file():
            raise StageError(f"sheet image not found: {path}")
        resolved.append(path)

    tiles: list[tuple[str, Image.Image]] = []
    for index, path in enumerate(resolved):
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        scale = float(width) / image.width
        tiles.append(
            (
                (labels[index] if labels and index < len(labels) else path.stem),
                image.resize((int(width), max(1, int(image.height * scale))), Image.Resampling.LANCZOS),
            )
        )
    columns = int(columns) or min(len(tiles), 3)
    rows = (len(tiles) + columns - 1) // columns
    row_heights = [max(t[1].height for t in tiles[r * columns : (r + 1) * columns]) for r in range(rows)]
    gap = 14
    header = 30
    canvas_width = columns * width + gap * (columns - 1)
    canvas_height = sum(h + header for h in row_heights) + gap * (rows - 1)
    canvas = Image.new("RGB", (canvas_width, canvas_height), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    y = 0
    for row in range(rows):
        x = 0
        for label, image in tiles[row * columns : (row + 1) * columns]:
            draw.text((x + 8, y + 6), label, fill=(232, 232, 232), font=_font(20))
            canvas.paste(image, (x, y + header))
            x += width + gap
        y += row_heights[row] + header + gap
    name = filename.format(stem=frame.meta.get("name", "sheet"))
    target = ctx.out(str(Path(subdir) / name) if subdir else name)
    canvas.save(target, compress_level=4)
    ctx.record(target)
    ctx.log(f"       wrote {target} ({canvas_width}x{canvas_height})")
    return frame
