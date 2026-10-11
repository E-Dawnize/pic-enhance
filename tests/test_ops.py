"""Unit tests for the pure-numpy layer.

These exist because the whole toolkit replaced OpenCV with hand-written numpy
code, and a silent error in ``box_mean`` or ``resize`` would corrupt every
result downstream without ever raising. Every test compares against a slow,
obviously-correct reference implementation.

Run:  python tests/test_ops.py       (no pytest required)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imgen import colors, ops  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "ok  " if condition else "FAIL"
    print(f"  {status} {name}{(' — ' + detail) if detail and not condition else ''}")
    if not condition:
        FAILURES.append(name)


def naive_box_mean(a: np.ndarray, radius: int) -> np.ndarray:
    """Explicit sliding average with edge clamping — the reference."""
    h, w = a.shape
    padded = np.pad(a, radius, mode="edge")
    total = np.zeros((h, w), np.float64)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            total += padded[dy : dy + h, dx : dx + w]
    return (total / (2 * radius + 1) ** 2).astype(np.float32)


def true_gaussian(a: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    padded = np.pad(a.astype(np.float64), radius, mode="edge")
    rows = np.zeros_like(a, dtype=np.float64)
    for i, k in enumerate(kernel):
        rows += k * padded[i : i + a.shape[0], radius : radius + a.shape[1]]
    padded = np.pad(rows, radius, mode="edge")
    out = np.zeros_like(a, dtype=np.float64)
    for i, k in enumerate(kernel):
        out += k * padded[radius : radius + a.shape[0], i : i + a.shape[1]]
    return out.astype(np.float32)


def test_box_mean() -> None:
    rng = np.random.default_rng(7)
    a = rng.random((37, 53), dtype=np.float32)
    for size in (3, 15, 31):
        mine = ops.box_mean(a, size)
        ref = naive_box_mean(a, size // 2)
        check(f"box_mean size={size} matches the sliding average", np.allclose(mine, ref, atol=1e-5), f"max diff {np.abs(mine - ref).max():.2e}")
    ones = np.ones((64, 64), np.float32)
    check("box_mean preserves a constant field", np.allclose(ops.box_mean(ones, 121), 1.0, atol=1e-5))
    check("box_mean handles an even size (rounds up to odd)", ops.box_mean(a, 16).shape == a.shape)


def test_gaussian() -> None:
    rng = np.random.default_rng(11)
    a = rng.random((48, 40), dtype=np.float32)
    for sigma in (1.0, 3.0, 8.0, 12.0, 16.0):
        mine = ops.gaussian_blur(a, sigma)
        ref = true_gaussian(a, sigma)
        diff = float(np.abs(mine - ref).max())
        check(f"gaussian_blur sigma={sigma} matches a true Gaussian (<1%)", diff < 0.01, f"max diff {diff:.4f}")
    for sigma in (18.0, 24.0, 40.0):
        mine = ops.gaussian_blur(a, sigma)
        ref = true_gaussian(a, sigma)
        diff = float(np.abs(mine - ref).max())
        check(f"gaussian_blur sigma={sigma} (box approximation) within 6%", diff < 0.06, f"max diff {diff:.4f}")
    check("gaussian_blur sigma=0 is a no-op", np.array_equal(ops.gaussian_blur(a, 0), a))
    check("gaussian_blur preserves the mean of a flat field", abs(float(ops.gaussian_blur(np.full((16, 16), 0.3, np.float32), 5).mean()) - 0.3) < 1e-4)


def test_resize() -> None:
    a = np.arange(64, dtype=np.float32).reshape(8, 8)
    down = ops.resize(a, (4, 4), interp="area")
    manual = a.reshape(4, 2, 4, 2).mean(axis=(1, 3))
    check("area resize == exact 2x block average", np.allclose(down, manual, atol=1e-5))
    up = ops.resize(a, (16, 16), interp="linear")
    check("linear resize produces the requested size", up.shape == (16, 16))
    check("linear resize keeps the corner values", abs(float(up[0, 0]) - float(a[0, 0])) < 1e-4)
    rng = np.random.default_rng(3)
    rgb = rng.random((20, 30, 3), dtype=np.float32)
    check("resize works on 3-channel arrays (auto = area when shrinking)", ops.resize(rgb, (10, 7)).shape == (7, 10, 3))
    check("resize is a no-op for the same size", ops.resize(rgb, (30, 20)) is not None and np.array_equal(ops.resize(rgb, (30, 20)), rgb))


def test_morphology() -> None:
    mask = np.zeros((60, 60), np.float32)
    mask[20:40, 20:40] = 1.0        # a solid block: survives an opening
    mask[5, 5:55] = 1.0             # a 1-px line: must disappear
    opened = ops.morph_open(mask, 15)
    check("morph_open keeps a solid block", float(opened[30, 30]) == 1.0)
    check("morph_open removes a 1-px line", float(opened[5, 30]) == 0.0)
    grown = ops.morph_dilate(opened, 21)
    check("morph_dilate grows the block by ~10 px", float(grown[30, 8]) == 1.0 and float(grown[30, 4]) == 0.0)
    pinhole = np.ones((40, 40), np.float32)
    pinhole[20, 20] = 0.0
    check("morph_close fills a pinhole", float(ops.morph_close(pinhole, 9)[20, 20]) == 1.0)
    check("morph_close keeps the background", float(ops.morph_close(pinhole, 9)[0, 0]) == 1.0)


def test_guided_filter() -> None:
    flat = np.full((32, 32), 0.4, np.float32)
    check("guided filter keeps a constant image", np.allclose(ops.guided_filter(flat, flat, 5, 1e-4), 0.4, atol=1e-3))
    rng = np.random.default_rng(5)
    guide = rng.random((24, 24), dtype=np.float32)
    noisy = guide + rng.normal(0, 0.05, guide.shape).astype(np.float32)
    filtered = ops.guided_filter(guide, noisy, 4, 1e-3)
    check("guided filter denoises towards the guide", np.abs(filtered - guide).mean() < np.abs(noisy - guide).mean())


def test_colors() -> None:
    values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    check("sRGB round trip is stable", np.allclose(colors.linear_to_srgb(colors.srgb_to_linear(values)), values, atol=1e-5))
    lin = colors.srgb_to_linear(np.array([0.5], np.float32))[0]
    check("sRGB 0.5 -> linear ~0.2140", abs(float(lin) - 0.2140) < 1e-3, f"got {float(lin):.4f}")
    check("LogC3 0.18 -> 0.391", abs(float(colors.LogC3.compress(np.array([0.18], np.float32))[0]) - 0.391) < 2e-3)
    check("LogC3 1.0 -> 0.5706", abs(float(colors.LogC3.compress(np.array([1.0], np.float32))[0]) - 0.5706) < 2e-3)
    check("LogC3 10.0 -> 0.8169", abs(float(colors.LogC3.compress(np.array([10.0], np.float32))[0]) - 0.8169) < 2e-3)
    check("LogC3 55.1 -> 1.0", abs(float(colors.LogC3.compress(np.array([55.1], np.float32))[0]) - 1.0) < 5e-3)
    probe = np.array([0.0, 0.01, 0.18, 1.0, 8.0, 55.0], np.float32)
    check("LogC3 round trip is stable", np.allclose(colors.LogC3.decompress(colors.LogC3.compress(probe)), probe, atol=2e-3))
    check("scRGB 1.0 == 80 nit", float(colors.nits_to_scrgb(np.array([80.0], np.float32))[0]) == 1.0)
    pq100 = float(colors.nits_to_pq(np.array([100.0], np.float32))[0])
    check("100 nit -> PQ ~0.5081", abs(pq100 - 0.5081) < 5e-3, f"got {pq100:.4f}")
    check("PQ round trip is stable", np.allclose(colors.pq_to_nits(colors.nits_to_pq(probe[:4] * 100)), probe[:4] * 100, rtol=1e-2))
    check("BT.2390 leaves values under the knee alone", float(colors.bt2390_tonemap(np.array([50.0], np.float32), 1000, 600)[0]) == 50.0)
    check("BT.2390 maps peak onto peak", abs(float(colors.bt2390_tonemap(np.array([1000.0], np.float32), 1000, 600)[0]) - 600.0) < 1.0)
    knee = 1.5 * 600 - 0.5 * 1000
    check(
        "BT.2390 is continuous at the knee",
        abs(float(colors.bt2390_tonemap(np.array([knee + 1e-3], np.float32), 1000, 600)[0]) - knee) < 0.5,
    )
    ramp = colors.bt2390_tonemap(np.linspace(0, 1000, 512, dtype=np.float32), 1000, 600)
    check("BT.2390 is monotone", bool(np.all(np.diff(ramp) >= -1e-3)))
    check("BT.2390 never exceeds the target peak", float(ramp.max()) <= 600.0 + 1e-3)


def test_smoothstep() -> None:
    check("smoothstep clamps below", float(ops.smoothstep(0.2, 0.8, np.array([0.1], np.float32))[0]) == 0.0)
    check("smoothstep clamps above", float(ops.smoothstep(0.2, 0.8, np.array([0.9], np.float32))[0]) == 1.0)
    check("smoothstep is 0.5 at the midpoint", abs(float(ops.smoothstep(0.2, 0.8, np.array([0.5], np.float32))[0]) - 0.5) < 1e-6)


def main() -> int:
    print("testing imgen.ops / imgen.colors")
    for name, func in (
        ("box_mean", test_box_mean),
        ("gaussian_blur", test_gaussian),
        ("resize", test_resize),
        ("morphology", test_morphology),
        ("guided_filter", test_guided_filter),
        ("colors", test_colors),
        ("smoothstep", test_smoothstep),
    ):
        print(f"\n[{name}]")
        func()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
