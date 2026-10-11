"""Download the external binaries and models the pipelines expect.

The repository deliberately ships **no** third-party binaries or weights — they
are large, separately licensed, and mostly replaceable. This script fetches the
ones the bundled pipelines reference, into ``tools/`` (which is git-ignored).

    python scripts/fetch_models.py --list         # show what would be downloaded
    python scripts/fetch_models.py --ncnn         # the three ncnn-vulkan tools
    python scripts/fetch_models.py --onnx         # the ONNX SR models
    python scripts/fetch_models.py --all --yes

Licences differ per artifact — read THIRD-PARTY.md before redistributing
anything you download with this script.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"

#: name -> (url, kind, licence note, destination relative to tools/)
ARTIFACTS = {
    "realesrgan": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip",
        "zip",
        "BSD-3-Clause (Real-ESRGAN); models have their own terms",
        ".",
    ),
    "realcugan": (
        "https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/20220728/realcugan-ncnn-vulkan-20220728-windows.zip",
        "zip",
        "MIT (tool); model weights: see upstream",
        ".",
    ),
    "waifu2x": (
        "https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20250915/waifu2x-ncnn-vulkan-20250915-windows.zip",
        "zip",
        "MIT (tool); model weights: see upstream",
        ".",
    ),
    # ONNX models are fetched from their publishers; put direct URLs here when
    # you have them, or drop the .onnx files into tools/ yourself.
    "anime-sharp-v4": (
        "https://huggingface.co/Kim2091/AnimeSharpV4/resolve/main/2x-AnimeSharpV4_RCAN_fp16_op17.onnx",
        "file",
        "check the model card before redistribution",
        ".",
    ),
}


def human(size: int) -> str:
    return f"{size / 1024 / 1024:.1f} MB"


def download(url: str, target: Path, assume_yes: bool) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        print(f"  already present: {target.name}")
        return target
    print(f"  {url}\n  -> {target}")
    if not assume_yes:
        answer = input("  download? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("  skipped")
            return target
    with urllib.request.urlopen(url) as response:  # noqa: S310 - explicit user request
        data = response.read()
    target.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    print(f"  {human(len(data))}  sha256={digest[:16]}…")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fetch the tools/models the pipelines need")
    parser.add_argument("--list", action="store_true", help="print the manifest and exit")
    parser.add_argument("--ncnn", action="store_true", help="fetch the ncnn-vulkan tools")
    parser.add_argument("--onnx", action="store_true", help="fetch the ONNX models")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args(argv)

    groups = {"ncnn": ("realesrgan", "realcugan", "waifu2x"), "onnx": ("anime-sharp-v4",)}
    selected: list[str] = []
    if args.all or not (args.ncnn or args.onnx):
        selected = list(ARTIFACTS)
    else:
        if args.ncnn:
            selected += list(groups["ncnn"])
        if args.onnx:
            selected += list(groups["onnx"])

    if args.list or not (args.ncnn or args.onnx or args.all):
        print("artifacts (destination: tools/, git-ignored):")
        for name, (url, kind, licence, dest) in ARTIFACTS.items():
            print(f"  {name:<16} {kind:<5} {licence}\n      {url}")
        print("\nre-run with --ncnn / --onnx / --all to download.")
        return 0

    TOOLS.mkdir(parents=True, exist_ok=True)
    for name in selected:
        url, kind, licence, dest = ARTIFACTS[name]
        print(f"[{name}] {licence}")
        target = TOOLS / dest / Path(url).name
        path = download(url, target, args.yes)
        if kind == "zip" and path.is_file():
            print(f"  extracting into {TOOLS}")
            with zipfile.ZipFile(path) as archive:
                archive.extractall(TOOLS)
            # keep the archive: it is the licence-carrying distribution
    print("\ndone. `imgen doctor` will now find the tools automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
