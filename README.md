# anime-hdr-upscale

把「修复 → 超分 → 去色带 → 锐化 → 语义 HDR 映射 → JXR 交付」做成**一条可增删阶段的管线**：
每一段都是一个独立 stage，管线本身是一份 JSON，改一行就能插入、删除、换实现或改参数。

本仓库同时收录**干净的成品**（`outputs/`）与**最新方案调研**（`docs/RESEARCH-2026.md`）。

---

## 1. 为什么要有它

这个项目原来的工作方式是：每条思路写一个一次性脚本，参数硬编码在 py 里，产物和工具混在同一个目录。
后果很具体：

* 一次 LumiPic 语义推理要 132 秒，**改一下映射参数就要重跑模型**；
* 想比较 "先去噪再超分" 和 "先超分再去噪"，得复制脚本、改路径、重跑；
* 三个月后回看某个成品，**说不清它是哪套参数跑出来的**。

这套工具针对的就是这三点：**管线即数据**（可增删）、**阶段可缓存**（贵的那一步只跑一次）、
**每次运行都留下可复现的记录**（参数、耗时、统计、产物清单）。

---

## 2. 快速开始

```bash
# 0) 需要 Python 3.10+，核心只依赖 numpy + Pillow
python -m imgen doctor            # 这台机器能做什么、缺什么、怎么补
python -m imgen stages            # 列出全部 23 个 stage
python -m imgen pipelines         # 列出内置管线并校验

# 1) 最省事的路径：一次 2x 超分（几秒）
python -m imgen run pipelines/quick-2x.json -i input.jpg -o out

# 2) 完整 4K HDR 壁纸（需要语义模型，见第 6 节）
python -m imgen run pipelines/anime-4k-hdr.json -i source.png -o out

# 3) 已经 4K、只想调 HDR：复用语义缓存，几秒一轮
python -m imgen run pipelines/hdr-retune.json -i source-4K.png \
    --set hdr.semantic.load.path=cache/lumipic.npy \
    --set hdr.map.peak_nits=800
```

`imgen doctor` 在本机的输出（示例）会告诉你：Python/依赖是否齐全、三个 ncnn 工具在哪、
WIC JXR 编码器在哪、显卡是什么、models 目录在哪。

---

## 3. 管线即数据（可插入 / 删除 / 重排）

一份管线就是一个 JSON 列表，每个元素是 `{stage, params, label?, enabled?}`：

```json
{
  "name": "anime-4k-hdr",
  "stages": [
    { "stage": "io.load" },
    { "stage": "deband.gradient", "params": { "threshold": 0.02 } },
    { "stage": "upscale.ncnn", "label": "2x-1",
      "params": { "tool": "realesrgan", "model": "realesr-animevideov3", "scale": 2 } },
    { "stage": "upscale.ncnn", "label": "2x-2",
      "params": { "tool": "realesrgan", "model": "realesr-animevideov3", "scale": 2 } },
    { "stage": "hdr.semantic.external", "params": { "steps": 25, "seed": 42 } },
    { "stage": "hdr.map", "params": { "peak_nits": 1000, "gate": "flatgate" } },
    { "stage": "encode.jxr", "params": { "verify": true } }
  ]
}
```

不用手写 JSON 也可以：

```bash
imgen add    pipelines/anime-4k-hdr.json sharpen.unsharp --after 2x-2 --params '{"amount":0.3}'
imgen remove pipelines/anime-4k-hdr.json denoise.ffmpeg        # 删掉一个阶段
imgen move   pipelines/anime-4k-hdr.json sharpen.unsharp --to 2 # 换顺序
imgen toggle pipelines/anime-4k-hdr.json deband.gradient        # 临时停用而不删除
imgen validate pipelines/anime-4k-hdr.json                      # 只查管线是否自洽
```

**插入/删除是安全的**，因为每个 stage 都声明了自己 *需要* 和 *产出* 的「数据平面」
（`rgb → semantic → highlight → nits → scrgb`）。删掉 `hdr.map` 却留着 `encode.jxr`，
`validate` 会直接告诉你："`[5] encode.jxr` 需要 scrgb，但前面没有任何 stage 产出它"，
而不是等运行到一半报一堆 numpy 广播错误。

运行时还可以只跑一部分：

```bash
imgen run pipelines/anime-4k-hdr.json -i a.png --until hdr.preview   # 跑到预览为止
imgen run pipelines/anime-4k-hdr.json -i a.png --only upscale.ncnn  # 只跑超分
imgen run pipelines/anime-4k-hdr.json -i a.png --skip encode.jxr    # 跳过编码
imgen run pipelines/anime-4k-hdr.json -i a.png --set hdr.map.peak_nits=900 --report run.json
```

---

## 4. 阶段一览

| 分组 | stage | 说明 |
|---|---|---|
| IO | `io.load` / `io.crop` / `io.resize` / `io.save` | 读、裁、缩放、写 |
| 去噪 | `denoise.nlmeans` / `denoise.bilateral` | 纯 numpy，色相保持，放大前用 |
| 去噪 | `denoise.ffmpeg` | hqdn3d / nlmeans 预设（需要 ffmpeg） |
| 去色带 | `deband.gradient` | 纯 numpy 四方向梯度去色带 + 抖动 |
| 去色带 | `deband.ffmpeg` | ffmpeg `deband` / `gradfun` |
| 锐化 | `sharpen.unsharp` / `sharpen.highpass` | 只作用于亮度，避免彩边 |
| 超分 | `upscale.ncnn` | Real-ESRGAN / Real-CUGAN / waifu2x，可串联多趟 |
| 超分 | `upscale.onnx` | onnxruntime 分块推理 + 余弦羽化（无接缝） |
| HDR | `hdr.semantic.external` | 调外部模型（LumiPic）产出语义高光 `.npy`，**带缓存** |
| HDR | `hdr.semantic.load` / `hdr.semantic.identity` | 复用缓存 / 无模型时的解析基线 |
| HDR | `hdr.map` | 语义高光 → 绝对亮度（nits）→ scRGB，核心阶段 |
| HDR | `hdr.preview` | 生成 SDR 检视预览与高光图 |
| 交付 | `encode.jxr` / `encode.float` | WIC JPEG XR（Windows HDR 壁纸）/ 浮点交换格式 |
| 分析 | `analyze.stats` / `analyze.crops` / `analyze.sheet` | 数值报告、1:1 局部对比、拼图 |

全部参数见 **[docs/STAGES.md](docs/STAGES.md)**（由 `scripts/gen_stage_docs.py` 从代码注册表自动生成，
永远和实现一致）。

---

## 5. 成品（`outputs/`）

| 文件 | 是什么 |
|---|---|
| `hdr/root-photo-4K-HDR-flatgate-balanced.jxr` | 人像照片 4K HDR 壁纸，3840×2628，可直接设为 Windows 壁纸 |
| `hdr/anime6b-4x-deband16-HDR.jxr` | 动画壁纸 4K HDR |
| `hdr/new_pic-HDR-wallpaper-auto-v8.jxr` | 插画壁纸 4K HDR |
| `preview/*.jpg` | 对应的 SDR 检视预览（Reinhard 映射，非屏幕模拟） |
| `compare/highlight-fix-1000nit-vs-flatgate.png` | 高光过曝修复的横向对比 |

`outputs/manifest.json` 记录每个文件的 SHA-256、尺寸、来源与生成流程；
`outputs/README.md` 是索引。用 `python scripts/build_outputs.py --source <父目录>` 可重新生成。

> Windows 只在 **系统能直接渲染 HDR 壁纸** 这一条路上支持 JPEG XR；
> 显示器需开启 HDR，然后「设置 → 个性化 → 背景 → 浏览图片」选择 `.jxr`。

---

## 6. 语义 HDR 需要什么

`hdr.semantic.external` 会把活交给**另一个解释器**（因为主流程不装 torch）：

```bash
# 在那个环境里
pip install "git+https://github.com/huggingface/diffusers.git"
pip install torch peft transformers accelerate safetensors

# 然后告诉管线用哪个解释器（任选其一）
imgen run pipelines/anime-4k-hdr.json -i a.png \
    --set hdr.semantic.external.python=D:/venvs/hdr/python.exe \
    --set hdr.semantic.external.model=D:/models/FLUX.2-klein-base-4B \
    --set hdr.semantic.external.lora=D:/models/klein4b_alpha_step1750.safetensors
# 或写进 config/tools.json 的 "python" 字段
```

模型输出（LogC3 编码）会被解到**场景线性**并存成 `.npy`；之后改任何映射参数都**不再需要 GPU**——
这正是原来最痛的地方。缓存键 = 输入文件哈希 + 采样参数（步数/guidance/seed/模型），
所以换 seed 会自动重算，改 `peak_nits` 则直接命中缓存。

没有 GPU / 没有模型也能跑：把该阶段换成 `hdr.semantic.identity`，
它会用解析式（源亮度 × 增益）当高光信号。**这是基线，不是好结果**——工具会在日志里明说。

---

## 7. 本仓库的一条重要实测结论

移植过程中做了一次对照复现（详见 **[docs/HDR-NOTES.md](docs/HDR-NOTES.md)**）：

* 未加门控的原始档：峰值 **906.8 nit**（参考记录 924）、`>500nit` **10.12%**（参考 10.84%）、
  受影响面积 20.74%（参考 21.95%）——**算法链路复现成立**；
* 白/青大色块的抑制效果也对得上：safe 档白色区均值 **203**（参考 201）、青区 **202**（参考 202）、
  balanced 档青区 **230**（参考 228）；
* 但 flatgate 档的**全图峰值复现不出来**：参考标注 433/464 nit，本次为 **874 nit**。

差异被定位到 **0.049%（4901 个像素）**：左下角星形装饰、上方花环叶片与发丝的**高光点**。
它们的 `source_y ≈ 0.99`，但 121px 邻域均值只有 **0.617**，于是 `smoothstep(0.55, 0.80, 0.617) = 0.18 < 0.5`，
按公式**本来就不会进入平坦掩膜**——门控的判据是「大面积平坦亮区」，而这些点恰恰"不平坦"。

**结论与用法**：本仓库不声称复现历史文件里标注的峰值，而是给出可解释的等效档位——
把亮度 ramp 下移（`flat_y_lo=0.45, flat_y_hi=0.65`）即可让峰值落到 622–647 nit、
`>500 nit` 占比降到 0.000–0.002%，观感与 "完全不发光" 的诉求一致。

---

## 8. 目录结构

```
anime-hdr-upscale/
├── imgen/                  # 工具本体（纯 numpy 实现，无 OpenCV 依赖）
│   ├── cli.py              # imgen run / add / remove / move / toggle / doctor …
│   ├── registry.py         # @stage 装饰器：声明 requires / produces / params / needs
│   ├── runner.py           # 管线校验 + 执行 + 报告
│   ├── ops.py              # 积分图模糊、导向滤波、形态学、重采样（全部有单测对照）
│   ├── colors.py           # sRGB / LogC3 / scRGB / PQ / BT.2390
│   ├── backends/           # ncnn、onnxruntime、WIC JXR、外部进程
│   └── stages/             # 23 个阶段实现
├── pipelines/              # 5 条可直接运行的管线
├── outputs/                # 干净成品 + manifest.json
├── docs/                   # 阶段参考、管线说明、HDR 笔记、调研报告
├── scripts/                # 语义推理、模型下载、成品构建、文档生成
├── tests/                  # 数值单测（对照暴力实现）+ 诊断脚本
└── config/tools.example.json
```

---

## 9. 环境与依赖

| 依赖 | 必需 | 用途 |
|---|---|---|
| Python 3.10+ | ✅ | |
| numpy、Pillow | ✅ | 全部图像运算 |
| onnxruntime(-gpu) | 可选 | `upscale.onnx` |
| Real-ESRGAN / Real-CUGAN / waifu2x ncnn-vulkan | 可选 | `upscale.ncnn`（`scripts/fetch_models.py --ncnn`） |
| ffmpeg | 可选 | `denoise.ffmpeg` / `deband.ffmpeg` |
| jxr-encode-wic + jxr-inspect + verify-jxr | 可选 | `.jxr` 交付（没有它仍可输出 scRGB 浮点） |
| torch + diffusers | 可选 | 语义 HDR 推理（独立解释器） |

工具路径的解析顺序：`--config` → 环境变量（`IMGEN_*`）→ `config/tools.json` → 自动发现
（仓库 `tools/`、父目录 `../tools`、`~/Documents` 等）。运行时不会硬编码任何机器路径。

---

## 10. 测试

```bash
python tests/test_ops.py           # 数值层对照暴力实现的单测（模糊/重采样/形态学/色彩）
python tests/debug_hdr_chain.py    # 打印 hdr.map 每一步的分布，定位峰值来源
python tests/dump_maps.py          # 把中间图 dump 成 PNG，肉眼确认掩膜
python scripts/gen_stage_docs.py --check   # 文档是否与代码同步
```

---

## 11. 许可

本仓库代码：MIT（见 `LICENSE`）。
第三方工具与模型**不随仓库分发**，各自的许可与注意事项见 **[THIRD-PARTY.md](THIRD-PARTY.md)**。
`outputs/` 里的图片素材版权归原作者/使用者，请自行确认再分发范围。
