"""Download the external binaries and models the pipelines expect.

The repository deliberately ships **no** third-party binaries or weights — they
are large, separately licensed, and mostly replaceable. This script fetches the
ones the bundled pipelines reference into ``tools/`` (which is git-ignored), so
that a fresh clone plus this script plus ``scripts/setup_env.py`` is a runnable
installation.

    python scripts/fetch_models.py --list            # show what would be downloaded
    python scripts/fetch_models.py --ncnn            # the three ncnn-vulkan tools
    python scripts/fetch_models.py --onnx            # the ONNX SR models
    python scripts/fetch_models.py --lumipic         # FLUX.2-klein-base-4B + LumiPic LoRA (~16 GiB)
    python scripts/fetch_models.py --ffmpeg          # ffmpeg (optional stages)
    python scripts/fetch_models.py --all --yes

Big model files are fetched in parallel byte ranges and are resumable: an
interrupted run leaves ``*.partN`` chunks next to the target and the next run
continues them. ``--endpoint`` switches to a mirror, which is worth trying when
huggingface.co is slow:

    python scripts/fetch_models.py --lumipic --endpoint https://hf-mirror.com --jobs 12

Licences differ per artifact — read THIRD-PARTY.md before redistributing
anything you download with this script.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"


class _Redirect(urllib.request.HTTPRedirectHandler):
    """Follow 308 as well — the stdlib on Python 3.10 only knows 301/302/303/307.

    Mirrors (hf-mirror.com and friends) answer with 308, so without this the
    whole script dies on the first request.
    """

    def http_error_308(self, req, fp, code, msg, headers):  # noqa: N802 - stdlib naming
        # pass 307 (not 308) down: redirect_request() only accepts 301/302/303/307
        return self.http_error_307(req, fp, 307, msg, headers)


_OPENER = urllib.request.build_opener(_Redirect)

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
    "ffmpeg": (
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
        "zip",
        "GPL build — only used by the optional denoise.ffmpeg / deband.ffmpeg stages",
        ".",
    ),
}

# --------------------------------------------------------------------------- #
# semantic HDR model (LumiPic): many files, a few of them multi-GiB
# --------------------------------------------------------------------------- #
LUMIPIC_ROOT = TOOLS / "LumiPic" / "models"
BASE_REPO = "black-forest-labs/FLUX.2-klein-base-4B"
LORA_REPO = "oumoumad/LumiPic"

#: relative path -> sha256 ("" = not pinned). Sizes come from the server.
BASE_FILES: dict[str, str] = {
    "model_index.json": "",
    "LICENSE.md": "",
    "README.md": "",
    "scheduler/scheduler_config.json": "",
    "transformer/config.json": "",
    "transformer/diffusion_pytorch_model.safetensors": "e109674697ffa1a3983126e32512f5428a9442bd8df59f9c95566ee90a473bb6",
    "text_encoder/config.json": "",
    "text_encoder/generation_config.json": "",
    "text_encoder/model.safetensors.index.json": "",
    "text_encoder/model-00001-of-00002.safetensors": "8c0506e7f4936fa7e26183a4fd8da4e2bdbc5990ba64ae441f965d51228f36ea",
    "text_encoder/model-00002-of-00002.safetensors": "82f2bd839378541b0557bfabaf37c7d3d637071fdcb73302dedd7cf61162ce07",
    "tokenizer/added_tokens.json": "",
    "tokenizer/chat_template.jinja": "",
    "tokenizer/merges.txt": "",
    "tokenizer/special_tokens_map.json": "",
    "tokenizer/tokenizer.json": "",
    "tokenizer/tokenizer_config.json": "",
    "tokenizer/vocab.json": "",
    "vae/config.json": "",
    "vae/diffusion_pytorch_model.safetensors": "",
}
LORA_FILES: dict[str, str] = {
    "klein4b_alpha_step1750.safetensors": "7f82ab02b9936c1db1ace597b7fb51d4d3f66bf312371d6dd470bd99d22f26d9",
}


def human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


# --------------------------------------------------------------------------- #
# plain downloads (small: zip archives and single files)
# --------------------------------------------------------------------------- #
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
    with _OPENER.open(url) as response:  # noqa: S310 - explicit user request
        data = response.read()
    target.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    print(f"  {human(len(data))}  sha256={digest[:16]}…")
    return target


# --------------------------------------------------------------------------- #
# parallel, resumable downloads (big: model weights)
# --------------------------------------------------------------------------- #
def remote_size(url: str, retries: int = 4) -> int:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, method="HEAD")
            with _OPENER.open(request, timeout=60) as response:  # noqa: S310
                return int(response.headers["Content-Length"])
        except Exception as exc:  # pragma: no cover - network dependent
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise SystemExit(f"cannot reach {url}: {last}")


def _fetch_range(url: str, start: int, end: int, part: Path, retries: int = 6) -> int:
    """Download ``bytes=start-end`` into ``part`` (skipped when already complete)."""
    want = end - start + 1
    if part.is_file() and part.stat().st_size == want:
        return want
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with _OPENER.open(request, timeout=180) as response, part.open("wb") as handle:  # noqa: S310
                shutil.copyfileobj(response, handle, 1 << 20)
            if part.stat().st_size == want:
                return want
            last = IOError(f"short read: {part.stat().st_size} != {want}")
        except Exception as exc:  # pragma: no cover - network dependent
            last = exc
        time.sleep(min(8.0, 1.5 * (attempt + 1)))
    raise SystemExit(f"giving up on {url} bytes {start}-{end}: {last}")


def fetch_parallel(
    url: str, dest: Path, *, jobs: int = 8, chunk_mb: int = 32, sha256: str = "", size: int | None = None
) -> str:
    """Download ``url`` to ``dest`` using ``jobs`` byte-range workers. Returns ``cached`` or ``downloaded``.

    ``size`` skips the probing HEAD request — huggingface.co answers HEAD in well
    over a minute, so the caller that already knows the size should pass it.
    """
    size = remote_size(url) if size is None else int(size)
    if dest.is_file() and dest.stat().st_size == size:
        if not sha256 or _sha256(dest) == sha256:
            return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    chunk = max(1 << 20, chunk_mb << 20)
    ranges = [(start, min(start + chunk - 1, size - 1)) for start in range(0, max(size, 1), chunk)]
    parts = [dest.with_name(f"{dest.name}.part{start}") for start, _ in ranges]
    started = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_fetch_range, url, start, end, part) for (start, end), part in zip(ranges, parts)]
        for future in concurrent.futures.as_completed(futures):
            done += future.result()
            pct = 100.0 * done / max(size, 1)
            rate = done / max(time.time() - started, 1e-6)
            print(f"\r    {pct:5.1f}%  {human(done)}/{human(size)}  {human(rate)}/s   ", end="", flush=True)
    print()
    tmp = dest.with_name(dest.name + ".assembling")
    with tmp.open("wb") as out:
        for part in parts:
            with part.open("rb") as handle:
                shutil.copyfileobj(handle, out, 1 << 20)
    if tmp.stat().st_size != size:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{dest.name}: assembled {tmp.stat().st_size} bytes, expected {size}")
    if sha256:
        digest = _sha256(tmp)
        if digest != sha256:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"{dest.name}: sha256 {digest} != expected {sha256}")
    os.replace(tmp, dest)
    for part in parts:
        part.unlink(missing_ok=True)
    return "downloaded"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_tree(endpoint: str, repo: str) -> dict[str, int]:
    """``path -> size`` for every file in a Hugging Face repo, in one request."""
    url = f"{endpoint.rstrip('/')}/api/models/{repo}/tree/main?recursive=true"
    with _OPENER.open(url, timeout=60) as response:  # noqa: S310
        entries = json.load(response)
    return {item["path"]: int(item.get("size", 0)) for item in entries if item.get("type") == "file"}


def fetch_lumipic(endpoint: str, jobs: int, assume_yes: bool, check: bool, chunk_mb: int = 8) -> int:
    """Download the semantic HDR base model and LoRA into tools/LumiPic/models/."""
    base_url = f"{endpoint.rstrip('/')}/{BASE_REPO}/resolve/main"
    lora_url = f"{endpoint.rstrip('/')}/{LORA_REPO}/resolve/main"
    base_dir = LUMIPIC_ROOT / "FLUX.2-klein-base-4B"
    lora_dir = LUMIPIC_ROOT / "LumiPic_lora"
    # Any ``*.part<N>`` left over from an interrupted run is reused when its size
    # already matches the range it stands for.
    small = [(name, digest) for name, digest in BASE_FILES.items() if not _is_big(name)]
    big = [(name, digest) for name, digest in BASE_FILES.items() if _is_big(name)]

    if check:
        problems = 0
        for name, digest in {**BASE_FILES}.items():
            path = base_dir / name
            if not path.is_file():
                print(f"  MISS  {name}")
                problems += 1
            elif digest and _sha256(path) != digest:
                print(f"  BAD   {name} (sha256 mismatch)")
                problems += 1
        for name, digest in LORA_FILES.items():
            path = lora_dir / name
            if not path.is_file():
                print(f"  MISS  {name}")
                problems += 1
            elif digest and _sha256(path) != digest:
                print(f"  BAD   {name} (sha256 mismatch)")
                problems += 1
        print("  all present and verified." if not problems else f"  {problems} file(s) missing or corrupt.")
        return 1 if problems else 0

    try:
        sizes = repo_tree(endpoint, BASE_REPO)
        lora_sizes = repo_tree(endpoint, LORA_REPO)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"  cannot read the file list from {endpoint} ({exc}); falling back to per-file HEAD")
        sizes = {name: remote_size(f"{base_url}/{name}") for name in BASE_FILES}
        lora_sizes = {name: remote_size(f"{lora_url}/{name}") for name in LORA_FILES}

    missing = []
    total = 0
    for name in BASE_FILES:
        size = sizes.get(name, 0)
        total += size
        target = base_dir / name
        if not (target.is_file() and target.stat().st_size == size):
            missing.append((f"{base_url}/{name}", target, name, size))
    for name in LORA_FILES:
        size = lora_sizes.get(name, 0)
        total += size
        target = lora_dir / name
        if not (target.is_file() and target.stat().st_size == size):
            missing.append((f"{lora_url}/{name}", target, name, size))

    print(f"[lumipic] {BASE_REPO} + {LORA_REPO}")
    print(f"  {len(BASE_FILES) + len(LORA_FILES)} files, {human(total)} -> {LUMIPIC_ROOT}")
    if not missing:
        print("  everything is already present.")
        return 0
    print(f"  {len(missing)} file(s) to download ({human(sum(item[3] for item in missing))})")
    if not assume_yes:
        answer = input("  download? [y/N]").strip().lower()
        if answer not in ("y", "yes"):
            print("  skipped")
            return 0

    for url, target, name, size in missing:
        print(f"  {name} ({human(size)})")
        state = fetch_parallel(
            url,
            target,
            jobs=jobs,
            chunk_mb=chunk_mb,
            sha256=BASE_FILES.get(name, LORA_FILES.get(name, "")),
            size=size,
        )
        print(f"    {state} -> {target}")
    return 0


def _is_big(name: str) -> bool:
    return name.endswith(".safetensors") and ("model-0000" in name or "pytorch_model" in name)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fetch the tools/models the pipelines need")
    parser.add_argument("--list", action="store_true", help="print the manifest and exit")
    parser.add_argument("--ncnn", action="store_true", help="fetch the three ncnn-vulkan tools")
    parser.add_argument("--onnx", action="store_true", help="fetch the ONNX models")
    parser.add_argument("--ffmpeg", action="store_true", help="fetch ffmpeg (optional stages)")
    parser.add_argument("--lumipic", action="store_true", help="fetch the semantic HDR model (~16 GiB)")
    parser.add_argument("--check", action="store_true", help="verify the fetched files and exit")
    parser.add_argument("--all", action="store_true", help="everything, including --lumipic")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    parser.add_argument("--jobs", type=int, default=8, help="parallel range workers for big files")
    parser.add_argument("--chunk-mb", type=int, default=8, help="bytes per range request when fetching big files")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                        help="model host, e.g. https://hf-mirror.com")
    args = parser.parse_args(argv)

    groups = {
        "ncnn": ("realesrgan", "realcugan", "waifu2x"),
        "onnx": ("anime-sharp-v4",),
        "ffmpeg": ("ffmpeg",),
    }
    selected: list[str] = []
    if args.all:
        selected = list(ARTIFACTS)
    else:
        for flag, names in groups.items():
            if getattr(args, flag):
                selected += list(names)

    if args.list or not (selected or args.lumipic or args.check):
        print("artifacts (destination: tools/, git-ignored):")
        for name, (url, kind, licence, dest) in ARTIFACTS.items():
            print(f"  {name:<16} {kind:<5} {licence}\n      {url}")
        print(f"\nsemantic HDR  LumiPic  {BASE_REPO} + {LORA_REPO}")
        print(f"      ~16 GiB -> {LUMIPIC_ROOT}  (--lumipic, resumable, --endpoint for a mirror)")
        print("\nrun with --ncnn / --onnx / --ffmpeg / --lumipic / --all to download, --check to verify.")
        return 0

    if args.check:
        return fetch_lumipic(args.endpoint, args.jobs, True, True, args.chunk_mb)

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
    if args.lumipic or args.all:
        status = fetch_lumipic(args.endpoint, args.jobs, args.yes, False, args.chunk_mb)
        if status:
            return status
    print("\ndone. `imgen doctor` will now find the tools automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
