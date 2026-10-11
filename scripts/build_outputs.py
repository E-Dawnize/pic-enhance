"""Assemble ``outputs/`` — the clean, finished results — from the project's finals.

The repository keeps only *deliverables*: HDR wallpapers that Windows actually
renders, their SDR previews (downscaled to keep the repository small), and a
manifest with SHA-256 hashes, provenance and the exact pipeline that produced
each one.

    python scripts/build_outputs.py --source .. --out ../outputs

``--source`` is the parent working folder (the one holding ``final/``).
Everything this script writes is reproducible; nothing is hand-edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

#: (source relative to --source, destination relative to --out, caption, provenance, pixel size)
ITEMS: list[tuple[str, str, str, str, str]] = [
    (
        "final/root-photo/root-photo-4K-HDR-flatgate-balanced-460nit.jxr",
        "hdr/root-photo-4K-HDR-flatgate-balanced.jxr",
        "人像照片 4K HDR 壁纸（推荐档）",
        "Real-ESRGAN x4plus-anime 4x -> Lanczos 4K -> LumiPic 语义高光 -> flatgate 亮度映射 -> WIC JXR",
        "3840x2628",
    ),
    (
        "final/root-photo/root-photo-4K-HDR-flatgate-balanced-460nit.preview.png",
        "preview/root-photo-HDR-preview.jpg",
        "上图的 SDR 检视预览（Reinhard，200 nit 纸白）",
        "同一条管线的 hdr.preview 阶段输出，转 JPEG 以控制仓库体积",
        "1920x1314",
    ),
    (
        "final/anime6b-4x-deband16-HDR.jxr",
        "hdr/anime6b-4x-deband16-HDR.jxr",
        "动画壁纸 4K HDR（anime6b / deband16）",
        "Real-ESRGAN anime6b 4x -> deband -> LumiPic -> 亮度映射 -> JXR",
        "3840x2160",
    ),
    (
        "final/new_pic-HDR壁纸-自动v8.jxr",
        "hdr/new_pic-HDR-wallpaper-auto-v8.jxr",
        "插画壁纸 4K HDR（自动档 v8）",
        "同上；映射曲线为自动档",
        "3840x2560",
    ),
    (
        "final/anime6b-4x-deband16_hdr/anime6b-4x-deband16_preview.png",
        "preview/anime6b-HDR-preview.jpg",
        "动画 HDR 壁纸的 SDR 检视预览",
        "hdr.preview 输出，转 JPEG",
        "1920x1080",
    ),
    (
        "final/root-photo/对比_高光修复_1000nit_vs_flatgate-safe_vs_balanced.png",
        "compare/highlight-fix-1000nit-vs-flatgate.png",
        "高光过曝修复对比：1000nit 原版 / flatgate-safe / flatgate-balanced",
        "横向拼图，1:1 像素预览",
        "3300x1314",
    ),
]

MAX_PREVIEW_WIDTH = 1920
JPEG_QUALITY = 88


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_preview(source: Path, target: Path) -> None:
    with Image.open(source) as handle:
        image = handle.convert("RGB")
        if image.width > MAX_PREVIEW_WIDTH:
            height = round(image.height * MAX_PREVIEW_WIDTH / image.width)
            image = image.resize((MAX_PREVIEW_WIDTH, height), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, quality=JPEG_QUALITY, subsampling=0, optimize=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the clean outputs/ folder")
    parser.add_argument("--source", default=str(ROOT.parent), help="folder containing final/")
    parser.add_argument("--out", default=str(ROOT / "outputs"), help="destination folder")
    args = parser.parse_args(argv)

    source_root = Path(args.source).resolve()
    out_root = Path(args.out).resolve()
    manifest: list[dict] = []
    missing: list[str] = []

    for relative, destination, caption, provenance, size in ITEMS:
        source = source_root / relative
        if not source.is_file():
            missing.append(relative)
            continue
        target = out_root / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() in (".jpg", ".jpeg"):
            write_preview(source, target)
        else:
            shutil.copy2(source, target)
        entry = {
            "file": destination,
            "caption": caption,
            "provenance": provenance,
            "source": relative,
            "size": size,
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        }
        if target.suffix.lower() == ".jxr":
            entry["format"] = "JPEG XR, 64bppRGBAHalf, scRGB (1.0 = 80 nit)"
        manifest.append(entry)
        print(f"  {destination}  {target.stat().st_size/1024/1024:.1f} MB")

    (out_root / "manifest.json").write_text(
        json.dumps({"items": manifest, "missing": missing}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\n{len(manifest)} file(s) written to {out_root}")
    if missing:
        print("not found in the source tree (skipped):")
        for item in missing:
            print(f"  {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
