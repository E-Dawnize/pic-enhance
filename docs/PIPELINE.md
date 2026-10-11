# 管线内部机制与扩展指南

面向"要改这个工具"的人。看完这篇你应该能：加一个 stage、换一个超分后端、
读懂运行报告，并知道为什么删掉一个阶段不会让整条管线悄悄跑错。

---

## 1. 数据平面（plane）与校验

`Frame` 是一个装着若干「平面」的容器，且**有严格顺序**：

```
rgb  →  semantic  →  highlight  →  nits  →  scrgb
```

| 平面 | 形状 / 类型 | 含义 |
|---|---|---|
| `rgb` | `(H,W,3) float32`，sRGB 编码，[0,1] | 工作图像（SDR、显示参考） |
| `semantic` | `(H,W,3) float32`，场景线性，可低分辨率 | 模型认为"光从哪里来" |
| `highlight` | `(H,W) float32`，[0,1] | 解析出的高光比例（决定多少 nit） |
| `nits` | `(H,W,3) float32` | 绝对亮度 cd/m² |
| `scrgb` | `(H,W,3) float32` | Windows scRGB（1.0 = 80 cd/m²） |

每个 stage 用装饰器声明自己 `requires` / `produces` 哪些平面，并声明 `kind`：

* `source`：只产出（如 `io.load`），整条管线最多一个；
* `process`：既读又写；
* `sink`：只写文件，不产出平面（如 `encode.jxr`）。

`Pipeline.validate()` 按顺序推演"当前有哪些平面"，任何一步缺料都会指名道姓地报出来：

```
$ imgen validate pipelines/broken.json
[ 5] encode.jxr: needs plane(s) scrgb which no earlier stage produces
     (available: rgb, semantic, highlight)
```

**修改 `rgb` 的 stage（去噪/超分/去色带/锐化/缩放/裁剪）会自动清空后面所有派生平面**——
因为 `semantic`、`highlight`、`nits` 都是针对旧像素算出来的。
这条规则写在 `Frame.invalidate_after()` 里，是"不会算错"的关键：
把超分插到 `hdr.map` 后面，不会得到"一半新一半旧"的结果，而是直接报缺平面。

---

## 2. 一次 run 的生命周期

```
load_pipeline(JSON)
   → apply_overrides(--set)
   → validate()                     # 平面链 + 参数名 + 外部依赖
   → 逐个 stage：require(needs) → func(frame, ctx, **params) → 计时 → 记录
   → report（JSON）+ 产物清单
```

`RunContext` 是 stage 唯一能碰到的外部世界：

| 成员 | 用途 |
|---|---|
| `ctx.tools` | 已解析的工具路径（`Tools` 数据类） |
| `ctx.log(msg)` | 统一的日志出口（CLI 里就是 print） |
| `ctx.out(name)` / `ctx.record(path)` | 成品路径 / 登记产物（进报告） |
| `ctx.temp(name)` | 中间文件（`work/tmp/`） |
| `ctx.index` / `ctx.label` | 当前阶段在管线中的序号与显示名 |
| `ctx.env["input"]` | `--input` 传入的原始路径 |

---

## 3. 写一个新 stage

```python
# imgen/stages/s_mystage.py
from .. import ops
from ..registry import StageError, stage
from ..state import Frame


@stage(
    "mystage.denoise",                       # 名字：分组.动作
    requires=("rgb",),                       # 需要哪些平面
    produces=("rgb",),                       # 产出哪些平面
    params={"strength": 0.5, "passes": 2},   # 参数与默认值
    doc="一句话说明（会出现在 imgen stages 与 docs/STAGES.md）",
)
def denoise(frame: Frame, ctx, strength: float = 0.5, passes: int = 2) -> Frame:
    if frame.rgb is None:
        raise StageError("mystage.denoise needs an rgb plane")
    out = frame.rgb.astype("float32", copy=True)
    for _ in range(int(passes)):
        out = out - (out - ops.gaussian_blur(out, 2.0)) * float(strength)
    frame.rgb = out
    frame.invalidate_after("rgb")   # 改了 rgb，就让派生平面失效
    frame.note(f"mystage strength={strength}")   # 进报告
    return frame
```

然后在 `imgen/stages/__init__.py` 里 import 这个模块（注册发生在 import 时），
跑一次 `python scripts/gen_stage_docs.py` 更新文档即可。

约定：

* 参数必须有默认值，且**名字与函数签名一致**（`with_params` 会校验多余/未知参数）；
* 失败要 `raise StageError("...\n  fix: ...")`，把"怎么修"写进消息；
* 需要外部程序就声明 `needs=("ffmpeg",)` 等，runner 会在运行前调用 `tools.require()` 给出可执行提示；
* 涉及浮点结果时**不要**顺手写 8-bit PNG 中间文件（会引入量化损失），
  除非该阶段本身以文件为接口（如 ffmpeg）。

---

## 4. 换/加一个超分引擎

`upscale.ncnn` 的参数 `tool` 决定走哪个可执行文件；新增一个同族引擎只需：

1. 在 `imgen/config.py` 的 `_CANDIDATES` / `_ENV_VARS` 里加上**候选相对路径**与环境变量名；
2. 在 `imgen/backends/ncnn.py` 的 `build_command()` 里加上该工具的参数语义
   （这三个工具对 `-n` / `-m` 的解释互不相同，这是必须显式分支的原因）。

ONNX 模型不需要改代码：把 `.onnx` 放进 `tools/` 或 `models/`，然后

```bash
imgen run pipelines/quick-2x.json -i a.png \
  --set upscale.ncnn.tool=onnx --set upscale.ncnn.model=2x-AnimeSharpV4_RCAN_fp16_op17
```

（换成 `upscale.onnx` 阶段时用 `--set upscale.onnx.model=...`；
分块大小 `tile`、重叠 `overlap`、`fp16`、`provider` 都可调，
重叠区用余弦羽化加权，避免 ncnn 那类接缝问题。）

---

## 5. 参数覆盖、跳过与缓存

```bash
# 覆盖：stage.param=value（JSON 解析优先，其次是字符串）
--set hdr.map.peak_nits=900 --set hdr.map.gate=none
--set hdr.semantic.load.path=D:/cache/lumipic.npy

# 也可以按序号：#3 表示第 4 个阶段（imgen validate 会打印序号）
--set '#3.amount=0.5'

# 部分运行
--skip decode.x --only upscale.ncnn --until hdr.preview
```

**语义缓存的语义**（这是本项目最省钱的一条）：`hdr.semantic.external` 的缓存键是
`输入文件哈希 + (steps, guidance, seed, model, lora, prompt)`。
所以：

* 改映射参数（`peak_nits`、`flat_y_*`、`allow_glow`…）→ **命中缓存，秒级**；
* 改 seed / 步数 / 换模型 → 自动重算，不会拿旧缓存骗你。

---

## 6. 运行报告

`--report run.json` 写出的结构：

```json
{
  "pipeline": "anime-4k-hdr",
  "input": "...", "out_dir": "...", "work_dir": "...",
  "stages": [
    {"index": 8, "stage": "hdr.map", "status": "ok", "seconds": 2.14,
     "params": {"peak_nits": 1000.0, "...": "..."},
     "size": [3840, 2628], "wrote": []}
  ],
  "stats": {"peak_nits": 873.6, "affected_pct": 20.13, "over_500_pct": 0.036, "gate": "flatgate"},
  "notes": ["hdr.map peak=874 nit  affected=20.13%  >500nit=0.036%  gate=flatgate"],
  "outputs": [".../root-photo-4K.jxr"]
}
```

`stats` 由各阶段写入 `frame.meta["stats"]`，`analyze.stats` 再把它们连同 SDR 统计落成
`<name>.stats.json`。**"这次跑出来多少 nit、多少面积受影响"应该是可查的**，而不是靠印象。

---

## 7. 常见报错与处理

| 报错 | 原因 / 处理 |
|---|---|
| `needs plane(s) scrgb which no earlier stage produces` | 管线被删/移乱了；用 `imgen validate` 看序号，把 `hdr.map` 放回编码前 |
| `unknown parameter(s) ...` | 参数名写错；`imgen describe <stage>` 查可用参数 |
| `the <tool> ncnn-vulkan binary was not found` | 装工具或设 `IMGEN_*` 环境变量；`imgen doctor` 会显示搜索过哪些位置 |
| `jxr-encode-wic.exe was not found` | 没装 JXR 编码器：可以先用 `encode.float` 出 scRGB 浮点，或补上工具 |
| `refusing to run NL-means with search_stride=1 on 21.0 MP` | 保护性拒绝：4K 上纯 numpy NL-means 太慢，换 `denoise.ffmpeg` 或放大 `search_stride` |
| `semantic cache not found` | `hdr.semantic.load` 的 `path` 未设置或还没跑过 `hdr.semantic.external` |
| `pipeline file is not valid JSON` | JSON 语法（多余逗号最常见） |
