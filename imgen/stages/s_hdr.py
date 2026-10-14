"""HDR stages: semantic highlight estimation and the luminance mapping.

The design principle is **where, not how**:

* a semantic model (or a cache of its output) only answers *where light comes
  from* — a low-resolution, scene-linear highlight signal;
* an explicit, monotone, parameterised mapping answers *how many nits* each
  pixel gets, and it always keeps the original image's chroma and geometry.

That split is why the output can be re-tuned in seconds without re-running a
diffusion model, and why the picture never gets "repainted" by the model.

Mapping pipeline (``hdr.map``), all in numpy:

1. ``raw = 1 - exp(-max(semantic_luma - knee, 0) / rolloff)`` — soft knee.
2. Binary **opening** of ``raw >= support_threshold`` removes thin semantic
   responses (backlit character outlines). Only the *binary* mask is opened:
   greyscale morphology on a continuous signal creates visible terraces.
3. Isolated emitters (stars, specular points) are rescued separately.
4. The low-resolution signal is upscaled and then **guided-filtered against the
   source luma**, so highlight edges come from the original image, not the model.
5. A **diffuse-white gate** suppresses flat bright regions that are paint, not
   light. Two implementations are available:
   ``flatgate`` (saturation-agnostic; default, matches the tuned values in this
   project) and ``whitegate`` (bright + desaturated, the classic formulation).
6. ``nits = paper_white * Y`` for diffuse white; the highlight fraction lifts it
   towards ``peak_nits``; an exponential shoulder keeps channel peaks
   asymptotic instead of clipping.
7. ``scrgb = nits / 80``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .. import ops
from ..backends import external
from ..colors import LUMA, SCRGB_NITS, srgb_to_linear
from ..config import REPO_ROOT
from ..registry import StageError, stage
from ..state import Frame

__all__ = ["semantic_load", "semantic_identity", "semantic_external", "map_hdr", "preview", "resolve_semantic_paths"]

#: Where the semantic model lives when the repository is self-contained
#: (``scripts/fetch_models.py --lumipic`` downloads exactly here).
SEMANTIC_ROOT = REPO_ROOT / "tools" / "LumiPic" / "models"
BASE_MODEL_DIR = SEMANTIC_ROOT / "FLUX.2-klein-base-4B"
LORA_CANDIDATES = (
    SEMANTIC_ROOT / "LumiPic_lora" / "klein4b_alpha_step1750.safetensors",
    SEMANTIC_ROOT / "LumiPic_lora" / "klein4b_alpha_step1000.safetensors",
)


def resolve_semantic_paths(model: str | None, lora: str | None) -> tuple[Path | None, Path | None]:
    """Resolve the base model directory and LoRA file.

    Explicit parameters win; otherwise the repository's own ``tools/LumiPic/models``
    is used, which is what makes a fetched clone runnable without machine paths.
    """
    base = Path(model).expanduser() if model else None
    if base is None and (BASE_MODEL_DIR / "model_index.json").is_file():
        base = BASE_MODEL_DIR
    weights = Path(lora).expanduser() if lora else None
    if weights is None:
        weights = next((p for p in LORA_CANDIDATES if p.is_file()), None)
    return base, weights


# --------------------------------------------------------------------------- #
# semantic sources
# --------------------------------------------------------------------------- #
@stage(
    "hdr.semantic.load",
    requires=(),
    produces=("semantic",),
    params={"path": "", "gain": 1.0, "expects": "linear"},
    doc="Load a cached semantic highlight map (.npy, scene-linear float).",
)
def semantic_load(frame: Frame, ctx, path: str = "", gain: float = 1.0, expects: str = "linear") -> Frame:
    if not path:
        path = str(ctx.work_dir / "semantic.npy")
    file = Path(path)
    if not file.is_file():
        raise StageError(f"semantic cache not found: {file} (run hdr.semantic.external first?)")
    semantic = np.load(file).astype(np.float32)
    if semantic.ndim == 2:
        semantic = np.repeat(semantic[..., None], 3, axis=2)
    frame.semantic = np.maximum(semantic * float(gain), 0.0)
    frame.note(f"semantic loaded {frame.semantic.shape[1]}x{frame.semantic.shape[0]} from {file.name}")
    return frame


@stage(
    "hdr.semantic.identity",
    requires=("rgb",),
    produces=("semantic",),
    params={"gain": 3.2, "gamma": 1.0},
    doc="No model: derive a highlight signal analytically from the source luma (fallback / baseline).",
)
def semantic_identity(frame: Frame, ctx, gain: float = 3.2, gamma: float = 1.0) -> Frame:
    if frame.rgb is None:
        raise StageError("hdr.semantic.identity needs an rgb plane")
    linear = srgb_to_linear(frame.rgb)
    luma = np.maximum(linear @ LUMA, 0.0)
    semantic = (np.power(luma, float(gamma)) * float(gain))[..., None] * np.ones(3, np.float32)
    frame.semantic = semantic.astype(np.float32)
    frame.note("semantic: analytic fallback (no model) — expect a flatter, less semantic grade")
    ctx.log("       note: hdr.semantic.identity has no semantic knowledge; it is a baseline, not the good path")
    return frame


def _cache_key(path: Path, params: dict) -> str:
    stat = path.stat()
    payload = json.dumps({"path": str(path.resolve()), "size": stat.st_size, "mtime": int(stat.st_mtime), **params}, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


@stage(
    "hdr.semantic.external",
    requires=("rgb",),
    produces=("semantic",),
    params={
        "python": "",
        "script": "",
        "model": "",
        "lora": "",
        "steps": 25,
        "guidance": 3.0,
        "seed": 42,
        "long_edge": 1024,
        "align": 16,
        "prompt": "Convert this image to HDR",
        "cache": "",
        "reuse": True,
        "cpu_offload": True,
        "timeout": 3600,
        "extra_args": [],
    },
    doc="Run an external semantic-HDR model (LumiPic / FLUX-class LoRA) and cache its .npy output.",
)
def semantic_external(
    frame: Frame,
    ctx,
    python: str = "",
    script: str = "",
    model: str = "",
    lora: str = "",
    steps: int = 25,
    guidance: float = 3.0,
    seed: int = 42,
    long_edge: int = 1024,
    align: int = 16,
    prompt: str = "Convert this image to HDR",
    cache: str = "",
    reuse: bool = True,
    cpu_offload: bool = True,
    timeout: float = 3600,
    extra_args: list | None = None,
) -> Frame:
    if frame.rgb is None:
        raise StageError("hdr.semantic.external needs an rgb plane")

    interpreter = Path(python) if python else ctx.tools.python
    script_path = Path(script) if script else REPO_ROOT / "scripts" / "lumipic_infer.py"
    if not script_path.is_file():
        raise StageError(f"inference script not found: {script_path}")

    model_path, lora_path = resolve_semantic_paths(model, lora)
    if model_path is None or lora_path is None:
        missing = []
        if model_path is None:
            missing.append(f"base model ({BASE_MODEL_DIR})")
        if lora_path is None:
            missing.append(f"LoRA ({LORA_CANDIDATES[0]})")
        raise StageError(
            "hdr.semantic.external is missing "
            + " and ".join(missing)
            + "\n  fix: python scripts/fetch_models.py --lumipic   (or pass --set hdr.semantic.external.model=... )"
        )
    model, lora = str(model_path), str(lora_path)
    if not Path(str(interpreter)).is_file():
        raise StageError(
            f"semantic interpreter not found: {interpreter}\n"
            "  fix: python scripts/setup_env.py   (creates .venv next to the repository)"
        )

    # The model sees a 1024-long-edge PNG; the .npy cache is keyed on that file
    # plus the sampling parameters, so re-tuning the mapping never re-runs it.
    work = ctx.work_dir / "semantic"
    work.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    stem = Path(frame.meta.get("name", "image")).stem or "image"
    key_params = dict(steps=steps, guidance=guidance, seed=seed, model=model, lora=lora, prompt=prompt)
    key = _cache_key(Path(frame.meta.get("input_path", stem)), key_params) if frame.meta.get("input_path") else "nocache"
    target = Path(cache) if cache else work / f"{stem}-{key}.npy"
    preview_in = work / f"{stem}-{key}-input.png"

    if reuse and target.is_file():
        ctx.log(f"       reusing semantic cache {target.name}")
    else:
        height, width = frame.rgb.shape[:2]
        scale = float(long_edge) / max(width, height)
        step = max(1, int(align))
        new_w = max(step, int(round(width * scale / step)) * step)
        new_h = max(step, int(round(height * scale / step)) * step)
        Image.fromarray((np.clip(frame.rgb, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB").resize(
            (new_w, new_h), Image.Resampling.LANCZOS
        ).save(preview_in, compress_level=2)
        command = [
            str(interpreter),
            str(script_path),
            "--input",
            str(preview_in),
            "--output",
            str(target),
        ]
        if model:
            command += ["--model", str(model)]
        if lora:
            command += ["--lora", str(lora)]
        command += [
            "--steps",
            str(int(steps)),
            "--guidance",
            str(float(guidance)),
            "--seed",
            str(int(seed)),
            "--prompt",
            str(prompt),
        ]
        if not cpu_offload:
            command += ["--no-cpu-offload"]
        command += [str(a) for a in (extra_args or [])]
        if ctx.log:
            ctx.log("       $ " + " ".join(command))
            ctx.log("       (first run loads a multi-GB model; the result is cached for re-tuning)")
        external.run_checked(command, log=ctx.log, timeout=float(timeout))
    if not target.is_file():
        raise StageError(f"semantic inference produced no file at {target}")
    semantic = np.load(target).astype(np.float32)
    if semantic.ndim == 2:
        semantic = np.repeat(semantic[..., None], 3, axis=2)
    frame.semantic = np.maximum(semantic, 0.0)
    frame.meta["semantic_cache"] = str(target)
    frame.note(f"semantic: {target.name} {frame.semantic.shape[1]}x{frame.semantic.shape[0]}")
    return frame


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def _flatgate(source_y: np.ndarray, params: dict) -> np.ndarray:
    """Suppression weight for large, flat, bright regions (paint, not light).

    Saturation-agnostic on purpose: a bright cyan sky and a white dress are both
    diffuse, and the classic "bright and desaturated" test misses saturated
    flat regions entirely.
    """
    base = ops.box_mean(source_y, int(params["flat_blur"]))
    local = ops.box_mean(source_y, int(params["flat_local"]))
    peak = source_y - local
    flat = ops.smoothstep(params["flat_y_lo"], params["flat_y_hi"], base) * (
        1.0 - ops.smoothstep(params["flat_peak_lo"], params["flat_peak_hi"], peak)
    )
    binary = (flat > 0.5).astype(np.float32)
    binary = ops.morph_open(binary, int(params["flat_open"]))
    binary = ops.morph_dilate(binary, int(params["flat_dilate"]))
    return ops.gaussian_blur(binary, float(params["flat_sigma"]))


def _whitegate(source_y: np.ndarray, source: np.ndarray, params: dict) -> np.ndarray:
    """Classic gate: suppress bright *and* desaturated regions."""
    high = source.max(2)
    low = source.min(2)
    saturation = (high - low) / np.maximum(high, 1e-6)
    white = ops.smoothstep(params["white_y_lo"], params["white_y_hi"], source_y) * (
        1.0 - ops.smoothstep(params["white_sat_lo"], params["white_sat_hi"], saturation)
    )
    binary = (white > 0.5).astype(np.float32)
    binary = ops.morph_open(binary, int(params["white_open"]))
    binary = ops.morph_dilate(binary, int(params["white_dilate"]))
    return ops.gaussian_blur(binary, float(params["white_sigma"]))


_GATE_DEFAULTS = {
    "flatgate": {
        "flat_blur": 121,
        "flat_local": 31,
        "flat_y_lo": 0.55,
        "flat_y_hi": 0.80,
        "flat_peak_lo": 0.03,
        "flat_peak_hi": 0.12,
        "flat_open": 21,
        "flat_dilate": 61,
        "flat_sigma": 18.0,
    },
    "whitegate": {
        "white_y_lo": 0.60,
        "white_y_hi": 0.85,
        "white_sat_lo": 0.08,
        "white_sat_hi": 0.25,
        "white_open": 15,
        "white_dilate": 41,
        "white_sigma": 15.0,
    },
}

_MAP_PARAMS = {
    "knee": 2.6,
    "rolloff": 3.2,
    "paper_white_nits": 200.0,
    "peak_nits": 1000.0,
    "shoulder_width": 400.0,
    "support_kernel": 15,
    "support_threshold": 0.02,
    "broad_sigma": 3.0,
    "isolated_fraction": 0.55,
    "isolated_neighborhood": 0.15,
    "isolated_sigma": 1.25,
    "neighborhood_scale": 15,
    "guided_radius": 10,
    "guided_eps": 0.0002,
    "source_gate_low": 0.04,
    "source_gate_high": 0.32,
    "point_gate_low": 0.50,
    "point_gate_high": 0.90,
    "gate": "flatgate",
    "allow_glow": 0.0,
    "strength": 1.0,
    **_GATE_DEFAULTS["flatgate"],
    **_GATE_DEFAULTS["whitegate"],
}


@stage(
    "hdr.map",
    requires=("rgb", "semantic"),
    produces=("nits", "scrgb", "highlight"),
    params=dict(_MAP_PARAMS),
    doc="Map a semantic highlight signal to absolute luminance and scRGB (the core SDR->HDR step).",
)
def map_hdr(frame: Frame, ctx, **params) -> Frame:
    if frame.rgb is None or frame.semantic is None:
        raise StageError("hdr.map needs both rgb and semantic planes")

    peak = float(params["peak_nits"])
    paper = float(params["paper_white_nits"])
    if peak <= paper:
        raise StageError(f"peak_nits ({peak}) must exceed paper_white_nits ({paper})")

    source = srgb_to_linear(frame.rgb).astype(np.float32)
    height, width = source.shape[:2]
    source_y = source @ LUMA

    semantic = np.maximum(np.asarray(frame.semantic, np.float32), 0.0)
    semantic_y = semantic @ LUMA
    low_h, low_w = semantic_y.shape[:2]

    raw_low = 1.0 - np.exp(-np.maximum(semantic_y - float(params["knee"]), 0.0) / float(params["rolloff"]))
    eligible = (raw_low >= float(params["support_threshold"])).astype(np.float32)
    eligible = ops.morph_open(eligible, int(params["support_kernel"]))
    broad = ops.gaussian_blur(eligible, float(params["broad_sigma"]))

    source_y_low = ops.resize(source_y, (low_w, low_h), interp="area")
    neighborhood = ops.box_mean(source_y_low, int(params["neighborhood_scale"]))
    isolated = (
        (raw_low >= float(params["isolated_fraction"])) & (neighborhood <= float(params["isolated_neighborhood"]))
    ).astype(np.float32)
    isolated = np.maximum(ops.gaussian_blur(isolated, float(params["isolated_sigma"])), isolated)
    supported_low = raw_low * np.clip(np.maximum(broad, isolated), 0.0, 1.0)

    highlight = ops.resize(supported_low, (width, height), interp="linear")
    highlight = ops.guided_filter(source_y, highlight, int(params["guided_radius"]), float(params["guided_eps"]))
    highlight = highlight * ops.smoothstep(float(params["source_gate_low"]), float(params["source_gate_high"]), source_y)

    raw_hr = ops.resize(raw_low, (width, height), interp="linear")
    size = max(3, int(round(float(params["neighborhood_scale"]) * width / max(1, low_w))))
    if size % 2 == 0:
        size += 1
    dark = ops.box_mean(source_y, size) <= float(params["isolated_neighborhood"])
    highlight = np.maximum(
        highlight,
        raw_hr * ops.smoothstep(float(params["point_gate_low"]), float(params["point_gate_high"]), source_y) * dark,
    )

    gate_name = str(params["gate"] or "none").lower()
    if gate_name == "flatgate":
        suppression = _flatgate(source_y, params)
    elif gate_name == "whitegate":
        suppression = _whitegate(source_y, source, params)
    elif gate_name in ("none", ""):
        suppression = np.zeros_like(source_y)
    else:
        raise StageError(f"unknown gate {params['gate']!r}; use flatgate | whitegate | none")
    gate = np.clip(1.0 - suppression * (1.0 - float(params["allow_glow"])), 0.0, 1.0)

    highlight = np.clip(highlight * gate * float(params["strength"]), 0.0, 1.0)

    base_nits = paper * source_y
    target_nits = base_nits + (peak - base_nits) * highlight
    chroma = source / np.maximum(source_y[..., None], 1e-7)
    nits = chroma * target_nits[..., None]
    nits[source_y <= 1e-7] = 0.0

    channel_peak = nits.max(axis=2)
    excess = np.maximum(channel_peak - paper, 0.0)
    shoulder = float(params["shoulder_width"])
    mapped = paper + (peak - paper) * (1.0 - np.exp(-excess / shoulder))
    mapped = np.where(channel_peak <= paper, channel_peak, mapped)
    nits = nits * (mapped / np.maximum(channel_peak, 1e-7))[..., None]

    frame.nits = nits.astype(np.float32)
    frame.scrgb = (nits / SCRGB_NITS).astype(np.float32)
    frame.highlight = highlight.astype(np.float32)

    stats = {
        "peak_nits": float(nits.max()),
        "p99_luma_nits": float(np.percentile(nits @ LUMA, 99)),
        "affected_pct": float(np.mean(highlight > 1e-5) * 100.0),
        "strong_pct": float(np.mean(highlight >= 0.25) * 100.0),
        "over_400_pct": float(np.mean(channel_peak > 400) * 100.0),
        "over_500_pct": float(np.mean(channel_peak > 500) * 100.0),
        "over_600_pct": float(np.mean(channel_peak > 600) * 100.0),
        "suppressed_pct": float(np.mean(gate < 0.5) * 100.0),
        "gate": gate_name,
    }
    frame.meta.setdefault("stats", {}).update(stats)
    frame.note(
        f"hdr.map peak={stats['peak_nits']:.0f} nit  affected={stats['affected_pct']:.2f}%  "
        f">500nit={stats['over_500_pct']:.3f}%  gate={gate_name}"
    )
    return frame


@stage(
    "hdr.preview",
    requires=("scrgb",),
    produces=(),
    kind="sink",
    params={
        "filename": "{stem}.preview.png",
        "highlight_map": "{stem}.highlight-map.png",
        "paper_white_nits": 200.0,
        "mode": "reinhard",
    },
    doc="Write an SDR inspection preview (Reinhard on paper white) and the highlight map.",
)
def preview(
    frame: Frame,
    ctx,
    filename: str = "{stem}.preview.png",
    highlight_map: str = "{stem}.highlight-map.png",
    paper_white_nits: float = 200.0,
    mode: str = "reinhard",
) -> Frame:
    from PIL import Image

    from ..colors import linear_to_srgb

    if frame.nits is None:
        raise StageError("hdr.preview needs the nits plane (run hdr.map first)")
    nits = np.maximum(frame.nits, 0.0)
    if mode == "reinhard":
        mapped = nits / (nits + float(paper_white_nits)) * 2.0
    elif mode == "clip":
        mapped = nits / float(paper_white_nits)
    else:
        raise StageError(f"unknown preview mode {mode!r}; use reinhard | clip")
    display = np.clip(linear_to_srgb(np.clip(mapped, 0.0, 1.0)), 0.0, 1.0)
    name = filename.format(stem=frame.meta.get("name", "image"))
    target = ctx.out(name)
    Image.fromarray((display * 255.0 + 0.5).astype(np.uint8), "RGB").save(target, compress_level=4)
    ctx.record(target)
    ctx.log(f"       wrote {target}")
    if highlight_map and frame.highlight is not None:
        mask_name = highlight_map.format(stem=frame.meta.get("name", "image"))
        mask_target = ctx.out(mask_name)
        mask = np.clip(frame.highlight, 0.0, 1.0)
        Image.fromarray((mask * 255.0 + 0.5).astype(np.uint8), "L").save(mask_target, compress_level=4)
        ctx.record(mask_target)
        ctx.log(f"       wrote {mask_target}")
    frame.note(f"preview written ({mode})")
    return frame
