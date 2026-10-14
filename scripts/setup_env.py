"""Create the repository's Python environment (``.venv``) and install the semantic HDR stack.

The core toolkit (``imgen``) needs only numpy + Pillow, so it runs on whatever
Python you already have. The *semantic* stage (``hdr.semantic.external``) shells
out to a separate interpreter that can import torch/diffusers/transformers —
this script builds exactly that interpreter, inside the repository:

    python scripts/setup_env.py                # CUDA 12.8 build of torch (default)
    python scripts/setup_env.py --cpu          # CPU-only torch (slow, but it runs)
    python scripts/setup_env.py --torch-index https://mirrors.aliyun.com/pytorch-wheels/cu128
    python scripts/setup_env.py --pip-index https://pypi.tuna.tsinghua.edu.cn/simple
    python scripts/setup_env.py --check        # verify an existing .venv

Afterwards ``imgen doctor`` picks the venv up automatically (it looks for
``.venv`` next to the repository) and  ``hdr.semantic.external`` uses it without
any ``--python`` override. Torch wheels are multi-GiB; on a slow link point
``--torch-index`` at a mirror.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
PYTHON = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

#: The versions this project was validated against. diffusers is the first
#: release that ships ``Flux2KleinPipeline`` (what LumiPic needs).
TORCH_CUDA = "torch==2.11.0+cu128"
TORCH_CPU = "torch==2.11.0"
DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
PACKAGES = (
    "numpy==2.2.6",
    "Pillow>=10.0",
    "transformers==5.14.1",
    "accelerate==1.14.0",
    "peft==0.20.0",
    "safetensors>=0.4",
    "diffusers==0.40.0",
)
SMOKE_TEST = (
    "import torch, diffusers, transformers\n"
    "print('torch      ', torch.__version__, '| cuda', torch.cuda.is_available(), torch.version.cuda)\n"
    "print('diffusers  ', diffusers.__version__)\n"
    "print('transformers', transformers.__version__)\n"
    "from diffusers import Flux2KleinPipeline\n"
    "print('Flux2KleinPipeline import ok')\n"
)


def run(command: list[str], **kwargs) -> None:
    print("  $", " ".join(str(c) for c in command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the repository venv for semantic HDR inference")
    parser.add_argument("--python", default=sys.executable, help="interpreter used to create the venv")
    parser.add_argument("--cpu", action="store_true", help="install CPU-only torch")
    parser.add_argument("--torch-index", default=DEFAULT_TORCH_INDEX, help="index for the torch wheel")
    parser.add_argument("--pip-index", default="", help="index for everything else (mirror)")
    parser.add_argument("--force", action="store_true", help="recreate the venv from scratch")
    parser.add_argument("--check", action="store_true", help="only run the smoke test")
    args = parser.parse_args(argv)

    if args.check:
        if not PYTHON.is_file():
            print(f"no venv at {VENV} — run `python scripts/setup_env.py` first")
            return 1
        run([str(PYTHON), "-c", SMOKE_TEST])
        return 0

    if args.force and VENV.exists():
        import shutil

        print(f"removing {VENV}")
        shutil.rmtree(VENV)
    if not PYTHON.is_file():
        print(f"creating {VENV}")
        run([str(args.python), "-m", "venv", str(VENV)])

    pip = [str(PYTHON), "-m", "pip", "install"]
    extra = ["-i", args.pip_index] if args.pip_index else []

    run(pip + ["-U", "pip", "wheel"] + extra)
    run(pip + ([TORCH_CPU] if args.cpu else [TORCH_CUDA, "--index-url", args.torch_index]))
    run(pip + list(PACKAGES) + extra)
    print("\nverifying:")
    run([str(PYTHON), "-c", SMOKE_TEST])
    print("\ndone. `python -m imgen doctor` now reports the semantic interpreter as the repo venv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
