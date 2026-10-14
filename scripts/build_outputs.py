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
    # -- produced by this repository's own pipelines (paths are relative to ROOT) --
    (
        "work/source/eye-closeup-4k.png",
        "source/eye-closeup-4k.png",
        "源图：动漫眼部特写 3840x2160（容器是 2x 上采样，实测真实解析力 1920x1080）",
        "用户提供；LANCZOS 1/2 往返 PSNR 52.9 dB、中值高通噪声 sigma 1.4/255、无 8/16 像素块效应",
        "3840x2160",
    ),
    (
        "out/eye-closeup-4k.jxr",
        "hdr/eye-closeup-4K-HDR-rebuild.jxr",
        "眼部特写 4K HDR 壁纸（1080p 真实解析力重建 + 语义 HDR）",
        "pipelines/rebuild-4k-hdr.json：2x 均值降到 1080p -> Real-CUGAN models-pro 2x -> deband -> LumiPic 语义高光 -> flatgate knee 0.5 / rolloff 1.5 / 纸白 200 nit / 峰值上限 1000 nit -> WIC JXR；实测峰值 871 nit，受影响 3.19%，>500 nit 0.0005%",
        "3840x2160",
    ),
    (
        "out/eye-closeup-4k.preview.png",
        "preview/eye-closeup-HDR-preview.jpg",
        "上图的 SDR 检视预览（Reinhard，200 nit 纸白）",
        "hdr.preview 阶段输出，转 JPEG 以控制仓库体积",
        "3840x2160",
    ),
    (
        "out/eye-closeup-4k.sr-compare.png",
        "compare/eye-closeup-sr-rebuild-1to1.png",
        "超分对照：原图 4K vs 重建 4K（1:1 像素）",
        "左：源文件；右：2x 均值降采样 + Real-CUGAN models-pro 2x",
        "1760x590",
    ),
    # -- 2026-10-15 batch: five illustrations from the drop folder + the eye close-up --
    (
        "work/source/girls-flowers-gift-0206aeaf.jpeg",
        "source/girls-flowers-gift.jpeg",
        "源图：二人与花束（1216x1080 源）",
        "用户提供；同名的 HDR / SDR 成品都由它跑出（pipelines/upscale-4k-hdr-png.json）",
        "",
    ),
    (
        "work/source/girls-happy-new-year-51b65721.jpeg",
        "source/girls-happy-new-year.jpeg",
        "源图：HAPPY NEW YEAR 2025 贺卡（2048x1970 源）",
        "用户提供；同名的 HDR / SDR 成品都由它跑出（pipelines/upscale-4k-hdr-png.json）",
        "",
    ),
    (
        "work/source/girls-flowers-hair-64c7c71c.jpeg",
        "source/girls-flowers-hair.jpeg",
        "源图：二人与花（1647x1474 源）",
        "用户提供；同名的 HDR / SDR 成品都由它跑出（pipelines/upscale-4k-hdr-png.json）",
        "",
    ),
    (
        "work/source/girls-tree-73597ebf.jpeg",
        "source/girls-tree.jpeg",
        "源图：树上的二人（1961x2778 竖构图）",
        "用户提供；同名的 HDR / SDR 成品都由它跑出（pipelines/upscale-4k-hdr-png.json）",
        "",
    ),
    (
        "work/source/girls-sitting-a068d7ab.jpeg",
        "source/girls-sitting.jpeg",
        "源图：并坐的二人（1434x1651 竖构图）",
        "用户提供；同名的 HDR / SDR 成品都由它跑出（pipelines/upscale-4k-hdr-png.json）",
        "",
    ),
    (
        "out/batch/eye-closeup-4k-4K.png",
        "sdr4k/eye-closeup-4K.png",
        "4K SDR 成品（未做 HDR 映射），eye-closeup",
        "pipelines/rebuild-4k-hdr.json：2x 均值降到 1080p -> Real-CUGAN models-pro 2x -> deband -> 4K PNG",
        "3840x2160",
    ),
    (
        "out/batch/girls-flowers-gift-0206aeaf-4K.png",
        "sdr4k/girls-flowers-gift-4K.png",
        "4K SDR 成品（未做 HDR 映射），girls-flowers-gift",
        "pipelines/upscale-4k-hdr-png.json：Real-CUGAN models-se 4x -> 4K PNG",
        "3840x3411",
    ),
    (
        "out/batch/girls-happy-new-year-51b65721-4K.png",
        "sdr4k/girls-happy-new-year-4K.png",
        "4K SDR 成品（未做 HDR 映射），girls-happy-new-year",
        "pipelines/upscale-4k-hdr-png.json：Real-CUGAN models-pro 2x -> 4K PNG",
        "3840x3694",
    ),
    (
        "out/batch/girls-flowers-hair-64c7c71c-4K.png",
        "sdr4k/girls-flowers-hair-4K.png",
        "4K SDR 成品（未做 HDR 映射），girls-flowers-hair",
        "pipelines/upscale-4k-hdr-png.json：Real-CUGAN models-se 4x -> 4K PNG",
        "3840x3437",
    ),
    (
        "out/batch/girls-tree-73597ebf-4K.png",
        "sdr4k/girls-tree-4K.png",
        "4K SDR 成品（未做 HDR 映射），girls-tree",
        "pipelines/upscale-4k-hdr-png.json：Real-CUGAN models-pro 2x -> 4K PNG",
        "2711x3840",
    ),
    (
        "out/batch/girls-sitting-a068d7ab-4K.png",
        "sdr4k/girls-sitting-4K.png",
        "4K SDR 成品（未做 HDR 映射），girls-sitting",
        "pipelines/upscale-4k-hdr-png.json：Real-CUGAN models-se 4x -> 4K PNG",
        "3335x3840",
    ),
    (
        "out/batch/girls-flowers-gift-0206aeaf-4K.jxr",
        "hdr/girls-flowers-gift-HDR.jxr",
        "4K HDR 壁纸，girls-flowers-gift；语义映射 knee 1.4 / rolloff 1.7，纸白 200 nit，峰值上限 1000 nit",
        "LumiPic 语义高光 -> hdr.map(flatgate 0.45/0.65, allow_glow 0.1) -> WIC JXR；SDR 版本见 sdr4k/girls-flowers-gift-4K.png",
        "3840x3411",
    ),
    (
        "out/batch/girls-happy-new-year-51b65721-4K.jxr",
        "hdr/girls-happy-new-year-HDR.jxr",
        "4K HDR 壁纸，girls-happy-new-year；语义映射 knee 1.4 / rolloff 1.7，纸白 200 nit，峰值上限 1000 nit",
        "LumiPic 语义高光 -> hdr.map(flatgate 0.45/0.65, allow_glow 0.1) -> WIC JXR；SDR 版本见 sdr4k/girls-happy-new-year-4K.png",
        "3840x3694",
    ),
    (
        "out/batch/girls-flowers-hair-64c7c71c-4K.jxr",
        "hdr/girls-flowers-hair-HDR.jxr",
        "4K HDR 壁纸，girls-flowers-hair；语义映射 knee 1.0 / rolloff 1.2，纸白 200 nit，峰值上限 1000 nit",
        "LumiPic 语义高光 -> hdr.map(flatgate 0.45/0.65, allow_glow 0.1) -> WIC JXR；SDR 版本见 sdr4k/girls-flowers-hair-4K.png",
        "3840x3437",
    ),
    (
        "out/batch/girls-tree-73597ebf-4K.jxr",
        "hdr/girls-tree-HDR.jxr",
        "4K HDR 壁纸，girls-tree；语义映射 knee 1.4 / rolloff 1.7，纸白 200 nit，峰值上限 1000 nit",
        "LumiPic 语义高光 -> hdr.map(flatgate 0.45/0.65, allow_glow 0.1) -> WIC JXR；SDR 版本见 sdr4k/girls-tree-4K.png",
        "2711x3840",
    ),
    (
        "out/batch/girls-sitting-a068d7ab-4K.jxr",
        "hdr/girls-sitting-HDR.jxr",
        "4K HDR 壁纸，girls-sitting；语义映射 knee 1.4 / rolloff 1.7，纸白 200 nit，峰值上限 1000 nit",
        "LumiPic 语义高光 -> hdr.map(flatgate 0.45/0.65, allow_glow 0.1) -> WIC JXR；SDR 版本见 sdr4k/girls-sitting-4K.png",
        "3335x3840",
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
            # deliverables produced by this repository's own pipelines live under
            # ROOT (out/, work/), not under --source
            source = ROOT / relative
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
