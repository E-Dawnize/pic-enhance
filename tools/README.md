# tools/ — 本地依赖

这个目录**不进版本库**（`.gitignore` 里 `tools/*`，只放行三个 JXR 小工具），
但整条管线默认就到这里找东西：`imgen doctor` 会把每一项解析结果打出来。

## 目录布局

```text
tools/
├── jxr-encode-wic.exe         # committed：scRGB 浮点 -> JPEG XR（WIC）
├── jxr-inspect.exe            # committed：读像素格式
├── verify-jxr.exe             # committed：WIC 解码自检
├── realesrgan-ncnn-vulkan.exe # scripts/fetch_models.py --ncnn
├── realcugan-ncnn-vulkan-20220728-windows/
├── waifu2x-ncnn-vulkan-20250915-windows/
├── models/                    # ncnn 超分权重（realesrgan / realcugan / waifu2x）
└── LumiPic/models/            # scripts/fetch_models.py --lumipic（约 15 GiB）
    ├── FLUX.2-klein-base-4B/  # transformer + text_encoder + vae + tokenizer
    └── LumiPic_lora/klein4b_alpha_step1750.safetensors
```

## 一次装齐

```bash
python scripts/fetch_models.py --all --yes        # ncnn + onnx + ffmpeg + LumiPic
python scripts/setup_env.py                       # .venv：torch cu128 + diffusers + transformers
python -m imgen doctor                            # 每项都应该是 ok
```

模型权重都是几十 GB 级的第三方文件，**不随仓库分发**：`fetch_models.py` 按
Hugging Face 仓库清单并行分片下载，支持断点续传（`*.partN` 会保留到拼装成功）。
国内网络慢时加 `--endpoint https://hf-mirror.com`，或者 `--jobs 16 --chunk-mb 4`。

许可与再分发注意事项见仓库根目录的 `THIRD-PARTY.md`。
