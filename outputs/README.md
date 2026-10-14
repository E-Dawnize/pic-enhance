# outputs/ — 干净成品

这里的文件是**可以直接用的结果**，不是中间产物。完整哈希、尺寸与来源见
[manifest.json](manifest.json)；生成方式见 `scripts/build_outputs.py`。

## HDR 壁纸（Windows 可直接使用）

| 文件 | 尺寸 | 说明 |
|---|---|---|
| `hdr/root-photo-4K-HDR-flatgate-balanced.jxr` | 3840×2628 | 人像照片；LumiPic 语义高光 + flatgate 抑制大面积漫反射白 |
| `hdr/anime6b-4x-deband16-HDR.jxr` | 3840×2160 | 动画壁纸（anime6b 4x + deband16） |
| `hdr/new_pic-HDR-wallpaper-auto-v8.jxr` | 3840×2560 | 插画壁纸（自动曲线档） |
| `hdr/eye-closeup-4K-HDR-rebuild.jxr` | 3840×2160 | 眼部特写：源文件虽为 4K，但实测真实解析力只有 1080p（2x 容器），先降到原生网格再 2x 重建，然后跑语义 HDR；峰值 871 nit，受影响 3.19% |
| `hdr/girls-*-HDR.jxr` | 3840 长边 | 2026-10-15 那一批五张动漫插画（含两张竖构图），每张的 SDR 版在 `sdr4k/` |

用法：显示器开启 HDR → 设置 → 个性化 → 背景 → 浏览图片 → 选择 `.jxr`。
格式为 **JPEG XR / 64bppRGBAHalf / scRGB（1.0 = 80 nit）**，即 Windows 合成器的原生 HDR 空间。

## 4K SDR 成品（`sdr4k/`）

同一批图的“只超分、不做 HDR”版本：长边 3840、保持原比例。竖构图不会被裁成十六比九
（`girls-tree` 是 2711×3840，`girls-sitting` 是 3335×3840）。
想用普通（SDR）壁纸就取这些 PNG，想用 HDR 就取 `hdr/` 里同名的 `.jxr`。

这一批的超分选择：源图长边 1216–2048，2x 后能盖住 3840 的就跑 2x（`models-pro`），
盖不住的就跑 4x（`models-se`，只有它带 `up4x-no-denoise`），然后统一按面积降到长边 3840。
五张源图都实测干净（无块效应，高频主要是画本身的笔触），所以全用 no-denoise；
超分收益的实例见 `compare/eye-closeup-sr-rebuild-1to1.png`。

## HDR 映射参数（`knee` / `rolloff`）

这批图的 knee 不是同一套模板抄的，而是按各自的语义分布选的：
眼部特写是夜景低调，knee 0.5；五张插画大多是高调白底，knee 1.0–1.4。
同一个 knee 用在两批图上，要么整片发白（knee 太低），要么毫无 HDR（knee 太高）。
判断标准不是 `affected_pct`（它在 flatgate + allow_glow 下会把被抑制区的残留也算上），
而是“峰值”和“超过 300 nit 的面积”：后者超过 ~5% 就开始发白。

## 检视预览

`preview/*.jpg` 是上面文件的 SDR 检视预览：Reinhard 映射、200 nit 纸白。
它**不是** HDR 屏幕的模拟，只用于在普通窗口里检查明暗分布是否合理。

## 对比图

`compare/highlight-fix-1000nit-vs-flatgate.png`：同一张图的 1000 nit 原始档、
flatgate-safe 档与 flatgate-balanced 档并排。

`compare/eye-closeup-sr-rebuild-1to1.png`：眼部特写的 1:1 像素对照，
左边是原图 4K（LANCZOS 2x 容器），右边是降到 1920×1080 后用 Real-CUGAN 2x 重建的结果。
发丝、睫毛、眼球里倒映人物的轮廓都能看到差异。

## 源图

`source/eye-closeup-4k.png` 是上面那张壁纸的输入，方便对照复现。
它是 3840×2160，但 1/2 往返 PSNR 52.9 dB（相当于 1920×1080 的信息量）。
`source/girls-*.jpeg` 是 2026-10-15 那五张的原始文件。

## 想自己重跑

```bash
# 用现有素材重建这个目录
python scripts/build_outputs.py --source ../.. --out .

# 或者从源图跑一遍完整管线（需要 ncnn 工具；语义阶段可选）
python -m imgen run pipelines/anime-4k-hdr.json -i <源图> -o ../../out
```

关于这些成品里的 HDR 参数为什么是这样，包括一次对照复现的完整结论
（哪一项复现了、哪一项复现不出来、为什么），见 [../docs/HDR-NOTES.md](../docs/HDR-NOTES.md)。
