# 图片超分与 HDR 方案调研（2026-10）

> 本文是五份分领域调研（`docs/research/A–E`）的综合版，面向**本地离线、Windows + 12 GB RTX 4070 SUPER**
> 这个具体约束，回答"现在有哪些思路、哪些能落地、哪些只是论文"。

---

## 0. 方法与可信度（请先读）

**怎么做的**：五个并行的检索代理分别负责
(A) 扩散/一步生成超分、(B) 经典与部署侧超分、(C) 单图 HDR/逆色调映射、
(D) HDR 交付与显示工程、(E) 真实世界工具链与实践。原始报告全文在 `docs/research/`。

**可信度分层**（重要）：

| 标记 | 含义 |
|---|---|
| ✅ | 本项目**实测**过（见 `docs/HDR-NOTES.md` 与 `tests/`） |
| `[检索]` | 来自搜索结果标题/摘要，**未打开原文** |
| `[待核]` | 许可、显存占用、具体数值等，需要打开原始页面确认 |

**已知的方法缺陷**：本轮检索环境下 `web_fetch` 对绝大多数域名被 DNS 拦截
（arxiv.org、github.com 等解析到非公网地址），因此**没有打开过任何论文正文、README 或榜单**；
所有条目来自搜索索引的标题与摘要。唯一可用的全文通道是 GitHub 镜像
`raw.gitcode.com/gh_mirrors/...`，报告 E 用它读到了 chaiNNer / Upscayl / Real-ESRGAN /
Anime4K / FFmpeg 滤镜源码等一手材料。

**因此**：下面的**方法名、年份、会议**可信度较高；**许可、显存、精确指标数值请自行复核**。
报告末尾列出了优先核验清单。

---

## 1. 超分辨率

### 1.1 一句话结论

**生成式超分已经从"多步扩散"坍缩到"一步"。** 2024 年的 SUPIR / DiffBIR / StableSR 那套
SDXL 多步管线，在交互式场景里已经被 OSEDiff（NeurIPS 2024）、S3Diff 这类一步模型取代——
单张 1024² 约 **1.3–1.4 秒** `[检索]`。同时，"保真 ↔ 真实感"的**可调旋钮**成为 2025–2026 的主线设计。

### 1.2 扩散/生成式超分谱系

| 代际 | 代表 | 年份/会议 `[检索]` | 要点 |
|---|---|---|---|
| 多步扩散 | StableSR、SeeSR、SUPIR、CoSeR、SinSR | CVPR 2024 | 质量高、慢；**SUPIR 在 12 GB 上会 OOM**（ComfyUI-SUPIR issue #12）`[检索]` |
| 多步（改） | PASD、DiffBIR | ECCV 2024 | 同上 |
| 残差位移 | ResShift | NeurIPS 2023 | 少步化的早期路线 |
| 一步蒸馏 | OSEDiff、DreamClear | NeurIPS 2024 | 一步出图，进入可用区间 |
| 一步/少步 | TSD-SR、PiSA-SR（Dual-LoRA）、AdcSR、InvSR | CVPR 2025 | **PiSA-SR 的双 LoRA 就是"保真度旋钮"** |
| 结构迁移 | DiT4SR、FlowSR | ICCV 2025 | UNet → DiT → 流匹配 |
| 2026 前沿 | OFTSR (ICLR 2026)、TEASR (ECCV 2026)、StrSR (ECCV 2026)、DriftSR (arXiv 2610.04819)、Fill2SR (2609.33582)、TOLA (2609.29240)、CLIMB-flow (2609.33834) | 2026 | 一步流匹配 + 可调保真度；文本/UI 专用分支出现 |
| 视频（创新引擎） | SeedVR2（3B/7B，**Apache-2.0**）、FlashVSR（宣称 12 GB 流式）、DUO-VSR (CVPR 2026) | 2025–2026 | 一步化 + 显存优化最激进的地方 |

其他值得记的名字：4KAgent（NeurIPS 2025，agent 式任意图 → 4K）、SRSR（NeurIPS 2025，语义准确性）、
NVIDIA **PiD**（NeurIPS 2026，`nv-tlabs/PiD`，用像素空间解码器**替掉 SDXL VAE**——
这是 4K 拼接/VAE 伪影最实在的一个解法）`[检索]`。
SD3 基座的开源权重出现了（HF `dipta007/OracleZoom`），**FLUX 基座的大规模 SR 仍未成为主流** `[检索]`。

### 1.3 经典 / GAN / Transformer / Mamba 侧

* **保真型 Transformer**（SwinIR 2021、HAT / DAT / SRFormer 2023、ATD 2024）**不适合**作为
  4K+ 动画的默认选择：慢、吃显存、结果偏平滑。它们是参照物，不是工具。
* **Mamba/SSM 是 2024–2026 最大的架构动向**：MambaIR（ECCV 2024）→ MambaIRv2（CVPR 2025）→
  2026 的轻量 SSM 一波（EDMT、State-prior Mamba、MambaRaw ECCV 2026、ProMamba）。
  但**没有任何一个发布了可用于动画超分的权重/可执行文件**——是 2027 的赌注，不是今天的选择。
* **真正在用的仍是 GAN/紧凑 CNN**：`realesr-animevideov3`、`RealESRGAN_x4plus_anime_6B`（均为
  BSD-3-Clause ✅ 许可层面已核对）、`2x-AnimeSharpV4`（RCAN，许可 `[待核]`）、Real-CUGAN（`[待核]`）。
* **动画/漫画专用**的新东西：**Caelum**（`yumenana/Caelum`，4×/2× 动画插画超分，明确针对
  "Pixiv/X/Facebook 多次上传再压缩"的真实退化链）`[检索]`、**TheAnimeScripter**（超分+RIFE+修复一条龙）、
  Anime4KCPP v3.2.0、ACNetGLSL。
* **1× 修复模型**（`1x StarSample V2.0 NS`、`2x Adore`）是 2026 一个被低估的类别：
  **只去伪影、不改倍率**。对"忠实存档"路线意义很大。
* ⚠️ **命名冲突**：`Anime4K` 现在指两个东西——bloc97 的 GLSL 着色器链（MIT，实时播放）和
  一篇论文《Anime4K: A Hybrid CNN–Transformer Network for Anime SR》（IEEE 11244719）。引用时必须区分。

### 1.4 部署现实

| 事实 | 影响 |
|---|---|
| ncnn-vulkan 的**分块接缝**是长期未修的 bug（issues #73/#197、Upscayl #600） | 大图优先用大 `tile` 或改用 ONNX + 羽化（本仓库 `upscale.onnx` 已用余弦羽化） |
| ONNX Runtime + CUDA/TensorRT 在 4070 SUPER 上最快；DirectML 通用但 fp16 覆盖不全 | `--set upscale.onnx.provider=cuda` |
| **fp16 几乎免费，int8 会真的伤画质**（PTQ-SR CVPR 2023；CVPRW 2026 仍在研究部署友好 INT8） | 不要为了省显存上 int8 |
| 显卡驱动级 **RTX VSR 是播放特性**（2023 年驱动 531.18 起，后补 RTX 20 系），**4K 源无收益**，且**不能吃静图** | 离线批量必须自己跑模型 |
| **RTX HDR 同样不能吃静图**；社区绕法：包成视频走 `NVEncC --vpp-ngx-truehdr`，或 MPC-BE 截图成 JXR | 想"一键 HDR 静图"目前没有官方路径 |
| **Windows Auto SR**（Copilot+ NPU，2026-09 扩到 Intel Core Ultra 3）是微软对 RTX VSR 的回应 | 同样是播放/游戏向，不是离线批处理 |
| NTIRE 2026 的主题是**效率与移动端真实退化**（第 11 届 ESR，arXiv 2604.03198；新增首届 Mobile Real-World SR，2604.17306） | 榜单不再奖励"更大的自然图像 ×4 模型" |

### 1.5 评测：没有共识，别只看 PSNR

* 全参考：PSNR/SSIM + LPIPS/DISTS/FID；无参考：NIQE/MUSIQ/MANIQA/CLIPIQA/TOPIQ/Q-Align。
* **NIQE 在生成式输出上基本失效**；2026 年仍**没有**被广泛接受的替代指标。
* NTIRE 2026 自己的工作就警告：**拿 IQA 指标当训练目标会过拟合**；CVPR 2026 有补充材料显示
  PI、NIQE、CLIP-IQA、MANIQA 在细微结构上会判错 `[检索]`。
* 有人直接主张方法论转向：《Evaluation of Visual Processing Should Be Human-Centered, Not Metric-Centered》(arXiv 2603.00643) `[检索]`。
* **动画评测有了新基准**：**Re-Anime600**（CVPRW 2026 AIGENS）——动画视频质量评估与超分评测数据集，
  比拿 DIV2K/Urban100 评动画靠谱得多。注意经典榜单几乎都是**双三次退化的自然图像**，
  跟"赛璐璐/高调插画 + 网络压缩"完全不是一回事。

---

## 2. HDR

### 2.1 一句话结论

**"生成模型负责定位、显式曲线负责定亮度"的混合架构胜出**——这与本项目已经采用的
`hdr.semantic.* + hdr.map` 思路一致 ✅。2026 年的新变化是：**gain map 从"质量故事"变成了"效率故事"**。

### 2.2 逆色调映射（ITM）谱系

| 类别 | 代表 | 年份/会议 `[检索]` |
|---|---|---|
| 传统深度 ITM（已平台期） | HDRTVNet (ICCV 2021) → HDRTVNet++ (TMM) → RealRep (AAAI-40, 2026) | 全局映射 + 局部细节 + 重新设计的监督 |
| 生成式端到端 | LEDiff (CVPR 2025)、Gain-Map Decomposed Diffusion (ICCV 2025)、Diffusion-Regularization ITM (ICCVW 2025 / AIM 2025) | 扩散直接产出 HDR |
| 效率化（2026 头号变化） | **FastGaMer** "Efficient GainMap Learning for Practical ITM" (CVPR 2026)、**GMODiff** 一步 gain map 精修 (arXiv 2512.16357) | "便宜预测器 + 一步扩散精修" |
| DiT 规模化 | LumaFlux (arXiv 2604.02787)、LumaGuide（免训练，2607.26237） | 质量上限，但**本地 12 GB 不舒适** |
| 语义混合 | 语义高光掩膜 + 物理映射（**本项目路线**）、MLLM-ITM (IJCAI 2026) | 生成模型只回答"哪里亮" |
| 文本驱动 | "Text-driven ITM for compressed SDR videos" (Pattern Recognition 2026) | 语义来自文本而非学习 |
| ⚠️ 许可 | **ExpandDiff** (arXiv 2609.39624) 摘要显示 **CC BY-NC-SA 4.0（非商用）** | 商用交付前必须回避或确认 |

**没有任何 2025–2026 生成式 ITM 方法公布了官方仓库**（`[待核]`：FastGaMer/GMODiff/LEDiff/LumaFlux 均未找到 repo），
权重可用性整体未确认。能跑的传统实现只有 HDRTVNet / HDRTVNet++ / ExpandNet。

### 2.3 失败模式与公认缓解手段

**主导失败类别是"平坦的赛璐璐色块 / UI / 文字"**——信号本身没有结构，任何 ITM 都在凭空造信息。

业界收敛到的缓解手段（与本项目 `hdr.map` 的第 ②③④⑤ 步一致 ✅）：

1. 高光资格/支撑掩膜；
2. **只在二值掩膜上**做形态学开/闭运算（对连续信号做灰度形态学会造台阶）；
3. 导向滤波把结果锁到原图边缘；
4. **漫反射白门控**：大面积平坦亮区把增益压到 1.0 以下；
5. 图形/UI 类内容用 100 nit 的低参考白。

⚠️ **专利先例**：US8050512（形态学开闭 + 漫反射白上限）、JP2023516184A（按区域判定"正漫反射白" `YposDW`）。
工程实现独立，但商用前应做 FTO 评估。

### 2.4 参考白与色彩约定

* **203 nit = 75% PQ** 是 ITU-R BT.2408 的广播参考白（现行版 **BT.2408-9，2026-03**）`[检索]`；
  100 nit 是低白替代；**200 nit 是消费/显示端习惯，不是标准值**。
* 本项目用 200 nit 纸白 ✅ —— 属于消费习惯档，与标准差 1.5%，可接受；若要对齐广播交付应改 203。

### 2.5 评测（HDR）

* **PU21（PU-PSNR / PU-SSIM）是事实上的基准指标**；HDR-VDP-3 是参考感知模型；色差用 ICtCp 下的 **δE-ITP**；
  视频可用 HDR-VQM（较老）。
* **AIM 2025 的 ITM 挑战赛报告仍是最近一届**（未找到 AIM 2026 ITM）。
* SIGGRAPH 2026《Adapting Quality Metrics to Tone Mapping》指出：**标准指标假设输入是显示编码的，
  会误判色调映射/HDR 对**——所以现在"只报 PU-PSNR"已经落后于前沿。
* 2026 新数据集：广播向的 **SDR 参考 ITM HDR 视频质量数据集（ACM 2026）**、**xDR**（Image Communication 2026，原生 HDR/SDR 成对）。

### 2.6 交付格式与显示链路（决定"能不能真的看到 HDR"）

| 事项 | 结论 | 状态 |
|---|---|---|
| Windows 合成空间 | **scRGB + FP16 + sRGB/BT.709 原色 + 线性 gamma，1.0 = 80 nit** | ✅ 与本仓库 `encode.jxr` 一致 |
| 内置 HDR 壁纸 | **只支持 JPEG XR（.jxr）**；Insider 2023-08 / Win11 24H2 GA | ✅ 本仓库主交付格式 |
| 合成器行为 | 对超出面板峰值的部分是**裁剪**，不是色调映射 → 映射必须在编码前做对 | ✅ 解释了为什么要 flatgate |
| Gain map（分享向） | 8-bit SDR 底 + 8-bit 对数比图；**ISO 21496-1:2025** 是总标准；Adobe XMP `hdrgm`；Apple 称 Adaptive HDR；Google Ultra HDR v1.1；Android 15 同时写两套元数据 | `[检索]` |
| 2026 新进展 | **libultrahdr v2.0.0（2026-08-07）支持 HEIF/HEIC 与 AVIF gain map**；libavif 1.2+ 的 gain map API 转正 | `[检索]` |
| JPEG XL | 2026 年进入主流（Chromium 合并、Chrome 155 GA） | `[待核]` 版本号 |
| ICC 层面 | **ADGC / adaptiveGainCurveType** 已进 ICC.1（2025-04-17），是与 ISO 21496-1 对齐的规范 | `[检索]` |
| PNG 第三版 | 新增 `cICP`、`mDCV`、`cLLI`、**`hRWL`（HDR 参考白 = 203 cd/m²）**；`cICP` 优先级高于 `iCCP` | `[检索]` |
| Python 现实 | **Pillow 不支持 JXR / gain map / JXL / HEIC**；写 JXR 要么用本仓库这种外部 CLI，要么走 WIC(COM) | ✅ 与 doctor 输出一致 |

### 2.7 本地能用的现成货

* **RTX Video HDR / TrueHDR**：实时、驱动级，但**算法固定**——不能设参考白，也不能做平坦白门控；
  静图需绕道 NVEncC。
* **LTX SDR-to-HDR**（开源模型 + ComfyUI 节点 + 社区 LoRA）、**Beeble SwitchHDR**（云，可当质量上限参照）。
* **`wallpaper-auto-hdr`**：专门做"桌面壁纸 SDR→HDR"的小工具，和本项目目标重叠，值得对标。

---

## 3. 对本仓库的具体建议

按"性价比"排序，都能挂在现有管线上：

1. **评测补 PU21**：新增 `analyze.pu21` 阶段（PU-PSNR/PU-SSIM），并把现有的 nits 统计一起写进报告。
   理由：HDR 侧唯一被广泛接受的指标，成本极低。
2. **加 1× 修复阶段**：`upscale.onnx` 现在只按倍率走；把它当成 1× 去伪影（`scale=1`）用于网络压缩图，
   比"先放大再补救"更符合真实退化链。可直接挂 `2x Adore`/StarSample 类模型。
3. **gain map 输出**：给 `.jxr` 之外补一条"可分享"的交付（Ultra HDR JPEG / AVIF gain map）。
   需要引入 libultrahdr / libavif 二进制（Pillow 做不到），属于新后端但接口简单。
4. **一步生成式 SR 的接入位**：架构上已经就绪（独立解释器 + `.npy` 缓存 + 可插拔 stage）。
   建议先接 **OSEDiff / S3Diff** 这类一步模型（12 GB 可跑 `[检索]`），并保留"保真度旋钮"参数；
   **不要**先碰 SUPIR（12 GB OOM `[检索]`）。
5. **评测数据集换成 Re-Anime600**（动画向）而不是 DIV2K/Urban100。
6. **阈值参数按尺度归一化**：`flat_blur=121` 在 4K 上≈3% 画面宽；把它改成"画面宽的百分比"参数，
   换分辨率时行为才一致（见 `docs/HDR-NOTES.md` §5）。
7. **避免的坑**：int8 量化、ncnn 小 tile、`gradfun` 处理 10-bit、以及任何 GPL/AGPL 组件的闭源分发
   （APISR = GPL-3.0，Video2X = AGPLv3，Waifu2x-Extension-GUI 仅个人免费）。

---

## 4. 优先核验清单（建议用能正常联网的环境逐条打开）

1. FastGaMer / GMODiff 的**是否有开源权重与许可**（决定第 3 条能否落地）；
2. ExpandDiff 的 **CC BY-NC-SA 4.0** 是否为最终许可（商用红线）；
3. NVIDIA **PiD**（NeurIPS 2026）仓库与显存需求（4K VAE/拼接伪影的正解）；
4. **OracleZoom**（SD3 基座）与 **StrSR**（ECCV 2026）的权重与许可；
5. NTIRE 2026 ESR 与 Mobile Real-World SR 的**获胜方法与数值**；
6. JPEG XL 的 Chrome/Firefox GA 版本号；
7. 2x-AnimeSharpV4 / Real-CUGAN / Caelum 的**模型许可**（决定能否打包分发）。

> 五份原始报告（含完整来源列表与逐条不确定性标注）：
> **A** 扩散/一步生成超分 · **B** 经典与部署 · **C** HDR/ITM · **D** HDR 交付与显示 · **E** 真实世界工具链。
> 都在 [`docs/research/`](research/) 下。
