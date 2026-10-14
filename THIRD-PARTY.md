# 第三方组件与许可 / Third-party notices

本仓库**不包含第三方二进制和模型权重**：十几个 GB 的东西由 `scripts/fetch_models.py`
按清单下载进 `tools/`（该目录不进版本库）。唯一的例外是三个WIC JXR 小工具
（`jxr-encode-wic.exe` / `jxr-inspect.exe` / `verify-jxr.exe`，共约 380 KB）：
它们没有上游发布页可拉，不随库里就不能完成最后一步编码，所以随仓库分发。
以下是本管线会调用到的东西，以及已知的许可要点。**再做分发（尤其是商用）前请自行复核最新条款。**

## 1. 超分工具与模型

| 组件 | 用途 | 许可 / 注意 |
|---|---|---|
| Real-ESRGAN ncnn-vulkan | `upscale.ncnn` (tool=realesrgan) | 工具 BSD-3-Clause；模型（realesrgan-x4plus / anime6b / realesr-animevideov3）随发布分发，商用前确认 |
| Real-CUGAN ncnn-vulkan | `upscale.ncnn` (tool=realcugan) | 工具 MIT；模型权重条款见上游仓库 |
| waifu2x ncnn-vulkan | `upscale.ncnn` (tool=waifu2x) | 工具 MIT；模型权重条款见上游仓库 |
| 2x-AnimeSharpV4 (RCAN) | `upscale.onnx` | 模型卡片条款需自行确认（社区模型） |
| Adore / Fallin-strong 等 ONNX | 参考用途 | 各自的模型卡片 |

参考：APISR 为 **GPL-3.0**（传染性），本项目**未**包含也未调用；
Waifu2x-Extension-GUI 为**个人使用免费**；Video2X 为 **AGPLv3**——若要商用请先读许可。

## 2. HDR / 编码

| 组件 | 用途 | 注意 |
|---|---|---|
| `jxr-encode-wic.exe` | scRGB 浮点 → JPEG XR（WIC） | 随仓库分发（仓库内构建的小工具，无外部依赖）。商用前请自行确认你所在司法辖区的可分发性 |
| `jxr-inspect.exe` / `verify-jxr.exe` | 校验 JXR 解码 | 同上 |
| Windows Imaging Component (WIC) | JXR 编解码底座 | 操作系统组件 |
| LumiPic / FLUX.2-klein-base-4B + LoRA | `hdr.semantic.external` | 基座模型与 LoRA 各有许可（HF 模型卡片）；**本仓库不分发权重**，`scripts/fetch_models.py --lumipic` 会从 Hugging Face 拉取（约 15 GiB，支持镜像与断点续传） |

## 3. 可选依赖

| 组件 | 用途 | 许可 |
|---|---|---|
| numpy | 全部数值运算 | BSD-3-Clause |
| Pillow | 图像读写 | HPND (PIL Software License) |
| onnxruntime | `upscale.onnx` | MIT |
| ffmpeg | `denoise.ffmpeg` / `deband.ffmpeg` | LGPL/GPL（取决于构建；分发 ffmpeg 二进制需注意） |
| PyTorch / diffusers | 语义 HDR 推理 | BSD-3-Clause / Apache-2.0 |

## 4. 图片素材

`outputs/` 中的 HDR 壁纸、预览、对照图与 `outputs/source/` 里的源图都由本管线在**本地素材**上生成。
素材版权归原作者或使用者所有；本仓库为**私有**仓库，请勿在未确认权利前再分发这些图片。

## 5. 专利提示

HDR 逆色调映射中的"形态学开/闭运算 + 漫反射白上限"以及"按区域判定漫反射白"
在专利文献中有先例（例如 US8050512、JP2023516184A）。
本仓库的实现是独立编写的工程实现，**不构成法律意见**；
若计划商业化，请自行做 FTO（自由实施）评估。
