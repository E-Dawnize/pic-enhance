"""Dump the intermediate maps as images so a human can see what went wrong.

    python tests/dump_maps.py <source.png> <semantic.npy> [outdir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imgen import ops  # noqa: E402
from imgen.colors import LUMA, srgb_to_linear  # noqa: E402

SOURCE = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\Documents\pic-enhance\final\root-photo\root-photo-4K.png")
SEMANTIC = Path(sys.argv[2] if len(sys.argv) > 2 else r"D:\Documents\pic-enhance\final\_hdr_jxr_work\root-photo-4K_cd2f42713f\lumipic.npy")
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else "_scratch/maps")
OUT.mkdir(parents=True, exist_ok=True)

with Image.open(SOURCE) as handle:
    rgb = np.asarray(handle.convert("RGB"), np.float32) / 255.0
semantic = np.load(SEMANTIC).astype(np.float32)

source = srgb_to_linear(rgb).astype(np.float32)
height, width = source.shape[:2]
source_y = source @ LUMA
sem_y = np.maximum(semantic @ LUMA, 0.0)
low_h, low_w = sem_y.shape[:2]

# --- the exact expressions from the reference scripts ---------------------- #
base = ops.box_mean(source_y, 121)
local = ops.box_mean(source_y, 31)
flat_cont = ops.smoothstep(0.55, 0.80, base) * (1.0 - ops.smoothstep(0.03, 0.12, source_y - local))
binary = (flat_cont > 0.5).astype(np.float32)
opened = ops.morph_open(binary, 21)
grown = ops.morph_dilate(opened, 61)
flat_soft = ops.gaussian_blur(grown, 18.0)

raw_low = 1.0 - np.exp(-np.maximum(sem_y - 2.6, 0.0) / 3.2)
eligible = ops.morph_open((raw_low >= 0.02).astype(np.float32), 15)
broad = ops.gaussian_blur(eligible, 3.0)
source_y_low = ops.resize(source_y, (low_w, low_h), interp="area")
neighborhood = ops.box_mean(source_y_low, 15)
isolated = np.maximum(
    ops.gaussian_blur(((raw_low >= 0.55) & (neighborhood <= 0.15)).astype(np.float32), 1.25),
    ((raw_low >= 0.55) & (neighborhood <= 0.15)).astype(np.float32),
)
supported_low = raw_low * np.clip(np.maximum(broad, isolated), 0.0, 1.0)
highlight = ops.resize(supported_low, (width, height), interp="linear")
highlight = ops.guided_filter(source_y, highlight, 10, 2e-4)
highlight = highlight * ops.smoothstep(0.04, 0.32, source_y)

print(f"image {width}x{height}, semantic {low_w}x{low_h}")
print(f"base(121)  p10={np.percentile(base,10):.3f} p50={np.percentile(base,50):.3f} p90={np.percentile(base,90):.3f}")
print(f"flat>0.5   {float(binary.mean())*100:.2f}%   after open {float(opened.mean())*100:.2f}%   after dilate {float(grown.mean())*100:.2f}%")
print(f"flat_soft  >=0.5 on {float((flat_soft>=0.5).mean())*100:.2f}% of the frame")
print(f"highlight  pre-gate >=0.5 on {float((highlight>=0.5).mean())*100:.3f}% of the frame")

# Where is the disagreement?  Bright AND emissive AND not gated.
bright_emissive = (source_y > 0.9) & (highlight > 0.5)
print(f"bright(source_y>0.9) & emissive(pre-gate>0.5): {float(bright_emissive.mean())*100:.3f}% of frame")
ys, xs = np.nonzero(bright_emissive)
if len(ys):
    print(f"  their flat_soft: mean={float(flat_soft[ys,xs].mean()):.3f} (ungated share {float((flat_soft[ys,xs]<0.5).mean())*100:.1f}%)")
    print(f"  sample coordinates: {list(zip(xs[:6].tolist(), ys[:6].tolist()))}")
    y0, x0 = int(ys[0]), int(xs[0])
    y1, y2 = max(0, y0 - 140), min(height, y0 + 140)
    x1, x2 = max(0, x0 - 140), min(width, x0 + 140)
    crop = rgb[y1:y2, x1:x2]
    Image.fromarray((np.clip(crop, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB").resize(
        ((x2 - x1) * 2, (y2 - y1) * 2), Image.Resampling.NEAREST
    ).save(OUT / "peak-neighbourhood.png")
    print(f"  crop around the first such pixel saved to {OUT/'peak-neighbourhood.png'}")
    print(f"    local: source_y={float(source_y[y0,x0]):.3f} base121={float(base[y0,x0]):.3f} "
          f"local31={float(local[y0,x0]):.3f} contrast={float(source_y[y0,x0]-local[y0,x0]):.4f} "
          f"flat_cont={float(flat_cont[y0,x0]):.3f} flat_soft={float(flat_soft[y0,x0]):.3f}")


def save_map(array: np.ndarray, name: str, scale: float = 1.0) -> None:
    data = np.clip(np.asarray(array, np.float32) * float(scale), 0.0, 1.0)
    image = Image.fromarray((data * 255 + 0.5).astype(np.uint8), "L")
    target_width = 1200
    image = image.resize((target_width, max(1, int(image.height * target_width / image.width))), Image.Resampling.LANCZOS)
    image.save(OUT / name)
    print(f"  {name}: mean={float(np.mean(array)):.3f} max={float(np.max(array)):.3f}")


print("\nmaps written to", OUT)
save_map(rgb @ LUMA, "01-source-luma.png")
save_map(base, "02-base121.png")
save_map(source_y - local, "03-local-contrast.png", scale=4.0)
save_map(flat_cont, "04-flat-continuous.png")
save_map(opened, "05-flat-binary-opened.png")
save_map(grown, "06-flat-dilated.png")
save_map(flat_soft, "07-flat-soft.png")
save_map(highlight, "08-highlight-pre-gate.png")
save_map(highlight * (1 - flat_soft * 0.9), "09-highlight-post-gate.png")
save_map(ops.resize(sem_y, (width, height), interp="linear") / 4.0, "10-semantic-luma-divided-4.png")
