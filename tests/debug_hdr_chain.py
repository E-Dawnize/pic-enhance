"""Diagnostic: why does the ported mapping disagree with the reference grade?

Run it against a cached semantic map and it prints, for every intermediate of
hdr.map, the statistics that decide the peak: how much of the frame the model
calls emissive, how much of that the flat gate suppresses, and where the
brightest output pixels get their value from.

    python tests/debug_hdr_chain.py <source.png> <semantic.npy> [allow_glow]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imgen import ops  # noqa: E402
from imgen.colors import LUMA, SCRGB_NITS, srgb_to_linear  # noqa: E402


def describe(name: str, array: np.ndarray) -> None:
    flat = np.asarray(array, np.float32).ravel()
    qs = np.percentile(flat, [0, 50, 90, 99, 99.9, 100])
    print(f"  {name:<26} min={qs[0]:9.4f} p50={qs[1]:8.4f} p90={qs[2]:8.4f} p99={qs[3]:8.4f} p99.9={qs[4]:8.4f} max={qs[5]:8.4f}")


def main() -> int:
    source_path = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\Documents\pic-enhance\final\root-photo\root-photo-4K.png")
    semantic_path = Path(
        sys.argv[2] if len(sys.argv) > 2 else r"D:\Documents\pic-enhance\final\_hdr_jxr_work\root-photo-4K_cd2f42713f\lumipic.npy"
    )
    allow = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10

    params = dict(
        knee=2.6, rolloff=3.2, paper_white_nits=200.0, peak_nits=1000.0, shoulder_width=400.0,
        support_kernel=15, support_threshold=0.02, broad_sigma=3.0,
        isolated_fraction=0.55, isolated_neighborhood=0.15, isolated_sigma=1.25, neighborhood_scale=15,
        guided_radius=10, guided_eps=2e-4, source_gate_low=0.04, source_gate_high=0.32,
        point_gate_low=0.50, point_gate_high=0.90,
        flat_blur=121, flat_local=31, flat_y_lo=0.55, flat_y_hi=0.80, flat_peak_lo=0.03, flat_peak_hi=0.12,
        flat_open=21, flat_dilate=61, flat_sigma=18.0,
    )

    with Image.open(source_path) as handle:
        rgb = np.asarray(handle.convert("RGB"), np.float32) / 255.0
    semantic = np.load(semantic_path).astype(np.float32)
    print(f"source {rgb.shape[1]}x{rgb.shape[0]}   semantic {semantic.shape[1]}x{semantic.shape[0]}   allow_glow={allow}")

    source = srgb_to_linear(rgb).astype(np.float32)
    height, width = source.shape[:2]
    source_y = source @ LUMA
    sem_y = np.maximum(semantic @ LUMA, 0.0)
    low_h, low_w = sem_y.shape[:2]

    print("\n[1] inputs")
    describe("source_y (linear)", source_y)
    describe("semantic_y (model)", sem_y)

    raw_low = 1.0 - np.exp(-np.maximum(sem_y - params["knee"], 0.0) / params["rolloff"])
    describe("raw_low (highlight)", raw_low)

    eligible = (raw_low >= params["support_threshold"]).astype(np.float32)
    opened = ops.morph_open(eligible, params["support_kernel"])
    print(f"\n[2] support mask: eligible={eligible.mean()*100:.2f}% after opening={opened.mean()*100:.2f}%")
    broad = ops.gaussian_blur(opened, params["broad_sigma"])

    source_y_low = ops.resize(source_y, (low_w, low_h), interp="area")
    neighborhood = ops.box_mean(source_y_low, params["neighborhood_scale"])
    isolated = ((raw_low >= params["isolated_fraction"]) & (neighborhood <= params["isolated_neighborhood"])).astype(np.float32)
    isolated = np.maximum(ops.gaussian_blur(isolated, params["isolated_sigma"]), isolated)
    supported_low = raw_low * np.clip(np.maximum(broad, isolated), 0.0, 1.0)
    describe("supported_low", supported_low)

    highlight = ops.resize(supported_low, (width, height), interp="linear")
    highlight = ops.guided_filter(source_y, highlight, params["guided_radius"], params["guided_eps"])
    highlight = highlight * ops.smoothstep(params["source_gate_low"], params["source_gate_high"], source_y)

    raw_hr = ops.resize(raw_low, (width, height), interp="linear")
    size = max(3, int(round(params["neighborhood_scale"] * width / low_w)))
    size += 0 if size % 2 else 1
    dark = ops.box_mean(source_y, size) <= params["isolated_neighborhood"]
    point = raw_hr * ops.smoothstep(params["point_gate_low"], params["point_gate_high"], source_y) * dark
    print(f"[3] point-emitter term fires on {float(dark.mean())*100:.2f}% of pixels; max={float(point.max()):.4f}")
    highlight = np.maximum(highlight, point)
    describe("highlight (pre-gate)", highlight)

    base = ops.box_mean(source_y, params["flat_blur"])
    local = ops.box_mean(source_y, params["flat_local"])
    flat = ops.smoothstep(params["flat_y_lo"], params["flat_y_hi"], base) * (
        1.0 - ops.smoothstep(params["flat_peak_lo"], params["flat_peak_hi"], source_y - local)
    )
    binary = (flat > 0.5).astype(np.float32)
    print(f"\n[4] flat mask: raw={float((flat > 0.5).mean())*100:.2f}%  after open={float(ops.morph_open(binary, params['flat_open']).mean())*100:.2f}%")
    soft = ops.gaussian_blur(ops.morph_dilate(ops.morph_open(binary, params["flat_open"]), params["flat_dilate"]), params["flat_sigma"])
    describe("flat_soft", soft)
    gate = np.clip(1.0 - soft * (1.0 - allow), 0.0, 1.0)
    describe("gate", gate)

    highlight = np.clip(highlight * gate, 0.0, 1.0)
    describe("highlight (post-gate)", highlight)

    bright = source_y > 0.85
    if bright.any():
        print(
            f"\n[5] pixels with source_y>0.85 ({float(bright.mean())*100:.2f}% of frame): "
            f"gate mean={float(gate[bright].mean()):.3f}  highlight mean={float(highlight[bright].mean()):.4f} "
            f"max={float(highlight[bright].max()):.4f}"
        )

    base_nits = params["paper_white_nits"] * source_y
    target = base_nits + (params["peak_nits"] - base_nits) * highlight
    chroma = source / np.maximum(source_y[..., None], 1e-7)
    nits = chroma * target[..., None]
    nits[source_y <= 1e-7] = 0.0
    cp = nits.max(2)
    excess = np.maximum(cp - params["paper_white_nits"], 0.0)
    mapped = params["paper_white_nits"] + (params["peak_nits"] - params["paper_white_nits"]) * (
        1.0 - np.exp(-excess / params["shoulder_width"])
    )
    mapped = np.where(cp <= params["paper_white_nits"], cp, mapped)
    nits = nits * (mapped / np.maximum(cp, 1e-7))[..., None]

    print("\n[6] output")
    describe("nits (channel peak)", nits.max(2))
    for threshold in (250, 300, 400, 500, 600):
        print(f"      >{threshold} nit: {float((nits.max(2) > threshold).mean())*100:.4f}%")
    print(f"      peak = {float(nits.max()):.1f} nit   scrgb peak = {float(nits.max()/SCRGB_NITS):.4f}")

    # Where does the peak come from?
    peak_index = np.unravel_index(np.argmax(nits.max(2)), nits.shape[:2])
    y, x = int(peak_index[0]), int(peak_index[1])
    print(
        f"\n[7] brightest pixel at ({x},{y}): source_y={float(source_y[y,x]):.4f} "
        f"semantic_y={float(sem_y[min(low_h-1, int(y*low_h/height)), min(low_w-1, int(x*low_w/width))]):.3f} "
        f"flat_soft={float(soft[y,x]):.3f} gate={float(gate[y,x]):.3f} highlight={float(highlight[y,x]):.4f}"
    )
    top = np.argsort(nits.max(2).ravel())[-2000:]
    ys, xs = np.unravel_index(top, nits.shape[:2])
    print(f"      top-2000 pixels: mean gate={float(gate[ys,xs].mean()):.3f} mean flat_soft={float(soft[ys,xs].mean()):.3f} "
          f"mean highlight={float(highlight[ys,xs].mean()):.4f} mean source_y={float(source_y[ys,xs].mean()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
