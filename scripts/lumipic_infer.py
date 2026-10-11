"""Standalone semantic-HDR inference (LumiPic / FLUX-class LoRA) -> scene-linear .npy.

This is the **only** part of the toolkit that needs PyTorch and a multi-GB
model. It is deliberately a separate process, so the rest of the pipeline keeps
running on plain numpy: ``hdr.semantic.external`` shells out to this script, and
the resulting ``.npy`` is cached, which means re-tuning an HDR grade never pays
for the model again.

What it produces is *not* an HDR image. It is a **semantic highlight field**: a
scene-linear float32 array where the value answers "how much light does this
pixel appear to emit", decoded from the model's LogC3 output. The numeric
mapping to nits happens later, in ``imgen.stages.s_hdr.map_hdr``.

Requirements (a different interpreter from the main pipeline is fine):

    pip install "git+https://github.com/huggingface/diffusers.git"  # >= 0.37.0.dev
    pip install peft transformers accelerate safetensors pillow numpy

Usage::

    python scripts/lumipic_infer.py --input in_1024.png --output sem.npy \
        --model black-forest-labs/FLUX.2-klein-base-4B \
        --lora  oumoumad/LumiPic --weight-name klein4b_alpha_step1750.safetensors

A 12 GB card needs ``--cpu-offload`` (the default); ``--no-cpu-offload`` is
roughly 2x faster on 32 GB+.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ARRI LogC3 (EI 800): the model's output space. Kept inline so this script has
# no import dependency on the package it feeds.
_LC = dict(A=5.555556, B=0.052272, C=0.247190, D=0.385537, E=5.367655, F=0.092809, CUT=0.010591)


def logc3_to_linear(encoded: np.ndarray) -> np.ndarray:
    """LogC3 [0, 1] -> scene-linear float32 (up to ~55.1)."""
    value = np.clip(np.asarray(encoded, dtype=np.float32), 0.0, 1.0)
    cut_log = _LC["E"] * _LC["CUT"] + _LC["F"]
    from_log = (np.power(10.0, (value - _LC["D"]) / _LC["C"]) - _LC["B"]) / _LC["A"]
    from_lin = (value - _LC["F"]) / _LC["E"]
    return np.where(value >= cut_log, from_log, from_lin).astype(np.float32)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LumiPic semantic HDR inference -> .npy")
    parser.add_argument("--input", required=True, help="input PNG/JPEG (already resized to ~1024 long edge)")
    parser.add_argument("--output", required=True, help="output .npy (scene-linear float32 HxWx3)")
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-4B",
                        help="base model id or local directory")
    parser.add_argument("--lora", default="oumoumad/LumiPic", help="LoRA repo id or local .safetensors file")
    parser.add_argument("--weight-name", default="klein4b_alpha_step1750.safetensors",
                        help="LoRA filename inside the repo (step1000 = more aggressive, step1750 = faithful)")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Convert this image to HDR")
    parser.add_argument("--no-cpu-offload", action="store_true", help="32 GB+ GPUs only (~2x faster)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save-preview", action="store_true", help="also write a Reinhard preview PNG")
    parser.add_argument("--pad-to", type=int, default=16, help="pad the input so both sides are multiples of this")
    return parser.parse_args(argv)


def load_pipeline(model: str, lora: str, weight_name: str, cpu_offload: bool, dtype_name: str):
    try:
        import torch
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "this script needs torch + a recent diffusers:\n"
            '  pip install "git+https://github.com/huggingface/diffusers.git"\n'
            "  pip install peft transformers accelerate safetensors pillow numpy\n"
            f"({exc})"
        ) from exc

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    print(f"loading base model: {model}", flush=True)
    pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=dtype)
    if Path(lora).is_file():
        print(f"loading LoRA from disk: {lora}", flush=True)
        pipe.load_lora_weights(lora)
    else:
        print(f"loading LoRA from the hub: {lora} ({weight_name})", flush=True)
        pipe.load_lora_weights(lora, weight_name=weight_name)
    if cpu_offload:
        # enable_model_cpu_offload() can OOM at step 0 on 12 GB cards; the
        # sequential variant is the safe one (this is why it is the default).
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    return pipe, torch


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from PIL import Image

    source = Path(args.input)
    if not source.is_file():
        raise SystemExit(f"input not found: {source}")

    with Image.open(source) as handle:
        image = handle.convert("RGB")
    step = max(1, int(args.pad_to))
    width = max(step, round(image.width / step) * step)
    height = max(step, round(image.height / step) * step)
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    if width % step or height % step:
        raise SystemExit(f"input must be a multiple of {step} on both sides; got {width}x{height}")

    pipe, torch = load_pipeline(args.model, args.lora, args.weight_name, not args.no_cpu_offload, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(int(args.seed))

    print(f"inference: {width}x{height}, {args.steps} steps, guidance {args.guidance}, seed {args.seed}", flush=True)
    started = time.time()
    result = pipe(
        prompt=args.prompt,
        image=image,
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.guidance),
        generator=generator,
        output_type="pt",
    )
    pixels = result.images[0].float().clamp(0.0, 1.0)
    logc = pixels.permute(1, 2, 0).cpu().numpy()
    semantic = logc3_to_linear(logc)
    elapsed = time.time() - started

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, semantic.astype(np.float32))

    luminance = semantic @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    above_knee = float((luminance > 2.6).mean() * 100.0)
    print(
        f"saved {target}  {semantic.shape[1]}x{semantic.shape[0]}  "
        f"max={float(semantic.max()):.2f}  >knee(2.6)={above_knee:.2f}%  {elapsed:.1f}s",
        flush=True,
    )
    if args.save_preview:
        mapped = np.clip(luminance / (1.0 + luminance), 0.0, 1.0) ** (1 / 2.2)
        Image.fromarray((mapped * 255).astype(np.uint8), "L").save(target.with_suffix(".preview.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
