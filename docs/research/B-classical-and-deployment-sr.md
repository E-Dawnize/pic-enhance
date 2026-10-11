# Image Super-Resolution, Part B — Classical / Non-Diffusion Methods and Their Deployment

*Compiled **2026-10-11**. Priority: 2025–2026 work; anything from 2026 is treated as the current frontier. A few older-but-still-shipped methods are included and explicitly age-flagged. Target scenario assumed throughout: **Windows 11, Python 3.10, onnxruntime + ncnn, RTX 4070 SUPER (12 GB), offline 4K+ upscaling of anime/illustration**.*

> **Verification caveat (read first).** In this session `web_fetch` was non-functional — every hostname (including arxiv.org, github.com, en.wikipedia.org) was rejected with *"resolves to a non-public IP address"*; subagent delegation was also disabled. **All findings come from the `web_search` index (titles + snippets) only; no primary page was opened.** Claims are marked `[V]` (seen directly in a snippet) or `[?]` (not verified this session). Do not republish a license or a benchmark number from this file without opening the source.

## TL;DR

- **The 2026 frontier in classical SR is *hybrid Mamba–Transformer* plus *degradation-aware efficiency*.** MambaIR (ECCV 2024) → MambaIRv2 (CVPR 2025) → 2026 brings EDMT (hybrid Mamba-Transformer, dual-path selective scan), State-prior Mamba SR, MambaRaw (ECCV 2026, 4K RAW) and ProMamba (all-in-one). `[V]`
- **The catch: almost none of the 2026 SSM line ships usable weights.** For a shipping tool in Oct 2026 the practical stack is still 2021–2024 GAN/compact-CNN plus 2026 *efficient CNN* work (PLKSR-Rep, IAFMNet, FANSR, DPAMixerSR). `[V]`
- **CVPR 2026 is the efficiency + real-degradation year.** IAFMNet (information-aware feature modulation, CVPR 2026) and the NTIRE 2026 efficiency/mobile tracks confirm the pivot away from bicubic PSNR. `[V]`
- **⚠️ Name collision you must not get wrong: "Anime4K" now means two different things.** bloc97/Anime4K (2019) is a **GLSL shader chain** for mpv. *"Anime4K: A Hybrid CNN–Transformer Network for Anime Super-Resolution"* (IEEE 11244719) is a separate **learned anime SR paper**. Cite them distinctly. `[V]`
- **Anime SR finally got a proper benchmark: Re-Anime600** (CVPR 2026 Workshop AIGENS) — a curated dataset for anime video quality assessment *and* SR evaluation. The single most useful 2026 addition for anime-specific work. `[V]`
- **Windows Auto SR (Automatic Super Resolution) is Microsoft's answer to RTX VSR**, running on the Copilot+ NPU; by Sept 2026 it had expanded to a third platform (Intel Core Ultra Series 3), after Qualcomm and AMD. It targets **video playback and games**, not offline image batches. `[V]`
- **APISR is confirmed GPL-3.0** (viral — blocks closed bundling); Real-ESRGAN models are the only widely-used anime upscalers whose permissive **BSD-3-Clause** license I verified. 2x-AnimeSharpV4, Real-CUGAN, NAFNet, Restormer, AutoDIR, InstructIR and R2R licenses remain unverified here. `[V]`
- **ncnn-vulkan is still the lowest-friction deployment** (single exe, no CUDA/Python) but tiling remains the #1 defect source: Real-ESRGAN-ncnn issues #73/#197 and Upscayl discussion #600 are long-standing, and 2026 hardware papers are still specifically about **tiling artifact reduction**. `[V]`
- **ONNX Runtime is at ~1.24.1 release / 1.26.0 (2026)**; Windows ML 2 now documents an EP matrix including NPU vendors. fp16 is ~free; **int8 still needs SR-specific treatment** — the paper addressing it is CVPRW 2026, i.e. this is an open problem, not a solved one. `[V]`
- **NPU SR moved from demo to product**: Qualcomm ships an optimized `Real-ESRGAN-x4plus` for Snapdragon, Intel documents AI Boost NPU upscaling, and Windows Auto SR is NPU-resident across three vendor platforms. All are fixed-model, single-image or playback scope — none replaces a CUDA batch pipeline. `[V]`
- **Composition beats model size, and blind A/B beats PSNR.** Deband → denoise → 2× → 2× (f3kdb/vs-deband → SCUNet/NAFNet → GAN SR) fixes banding/blocking that a GAN upscaler otherwise sharpens into halos. Judge on line integrity, screentone moiré, halos, banding and colour drift — CVPR 2026 work documents that NR-IQA metrics "can fail on cases where subtle structure" is the point, and challenge papers warn that optimizing IQA objectives overfits them. `[V]`

---

## 1. Fidelity backbones: CNN / Transformer

| Name | Year | One-line idea | Repo / weights | License | Deploy | CPU vs GPU | Fit for 4K+ anime on 12 GB 4070S | Weaknesses |
|---|---|---|---|---|---|---|---|---|
| **SwinIR** | 2021 (old) | Swin-window attention + residual Swin blocks; canonical SR baseline | `JingyunLiang/SwinIR`; ONNX mirror `Heliosoph/swinir-onnx` `[V]` | Apache-2.0 `[V]` | PyTorch, ONNX | CPU minutes/4K; GPU heavy activations | Poor default; needs tiling; too smooth for line art | Over-smoothing, washed texture, large VRAM |
| **HAT** | 2023 | Hybrid attention: channel + window + **overlapping cross-attention** | `XPixelGroup/HAT`; HF `Acly/hat` `[V]` | Apache-2.0 `[?]` | PyTorch only | GPU-only in practice, ~20M+ params | **Not recommended** — best PSNR, worst ergonomics | Very slow, memory-hungry, over-smooth, no ONNX path |
| **DAT** | 2023 | Dual aggregation: alternating spatial + channel attention | `zhengchen1999/DAT`; HF conversions `[V]` | Apache-2.0 `[?]` | PyTorch, community ONNX | GPU; DAT-S ~14M | Usable tiled; sharper than SwinIR/HAT | Slow; ringing on hard edges |
| **SRFormer** | 2023 | Permuted self-attention: large window, small channel budget | `Zhou/SRFormer` `[V]` | Apache-2.0 `[?]` | PyTorch | GPU; lighter than HAT | Moderate; still Transformer-slow at 4K | Mild over-smoothing; few ONNX/ncnn ports |
| **Omni-SR** | 2023 | Omni aggregation for **lightweight** SR (~0.8M params) | `Francis0625/Omni-SR` `[V]` | Apache-2.0 `[?]` | PyTorch | Fast on GPU | Good speed tier; weaker than GANs on real anime degradation | Blurrier than GAN line; name collides with "OmniSR: Shadow Removal…" (arXiv 2410.01719) `[V]` |
| **ATD** | 2024 | Adaptive **token dictionary** to break the local-window limit | arXiv 2401.08209; `LabShu/ATD` `[?]` | `[?]` | PyTorch | GPU | Better PSNR/param than HAT, still fidelity-oriented | Not perceptually sharp; no lightweight deploy path |
| **IAFMNet** | **2026 CVPR** | Information-aware feature modulation for *efficient* SR | CVPR 2026 (Xu et al.) `[V]` | `[?]` | PyTorch → ONNX | Efficient tier | **Current-frontier efficient candidate** | Repo/weights unverified; efficiency-track evaluation only |

*Older ancestors as reference points only:* SRCNN (2014), EDSR (2017), RCAN (2018 — the architecture 2x-AnimeSharpV4 builds on `[V]`), ESRGAN (2018).

## 2. Mamba / state-space and RWKV SR — the 2024→2026 line

| Name | Year | One-line idea | Repo / weights | License | Deploy | CPU vs GPU | 4K-anime fit | Weaknesses |
|---|---|---|---|---|---|---|---|---|
| **MambaIR** | 2024 ECCV | Residual state-space blocks + local enhancement; linear-complexity global receptive field | `csguoh/mambair` `[V]` | Apache-2.0 `[V]` | PyTorch + CUDA selective-scan | GPU-only realistically | Not perceptual; tiling needed | Custom CUDA scan ops — poor ONNX story; over-smooth |
| **MambaIRv2** | 2025 CVPR | "Attentive state-space": non-causal scanning so tokens attend outside scan order | same repo `[V]` | Apache-2.0 `[V]` | PyTorch | GPU | Research-grade | Deploy friction unchanged |
| **VmambaIR** | 2024 | Visual state-space (VMamba) blocks for restoration | `AlphacatPlus/VmambaIR` `[V]` | `[?]` | PyTorch | GPU | Research-grade | No production tooling |
| **EDMT** | **2026** | Efficient **hybrid Mamba-Transformer** with dual-path selective scan for *lightweight* SR | IEEE 11633820 `[V]` | `[?]` | Paper | Edge/lightweight tier | **Most deployment-relevant SSM entry of 2026** | No weights surfaced |
| **State-prior Mamba SR** | **2026** | Injects a learned state prior into the SSM for SR | Computers & Graphics 2026 `[V]` | `[?]` | Paper | — | Conceptually relevant | No repo |
| **MambaRaw** | **2026 ECCV** | Selective state-space modelling for efficient **4K RAW** reconstruction | `Peizeli1/MambaRaw` `[V]` — **has a public repo** | `[?]` | PyTorch | Efficient 4K | RAW domain, but the 4K-efficiency claim transfers | Not anime; RAW-only |
| **ProMamba** | **2026** | Prompt-guided Mamba for **all-in-one** restoration | Signal Processing vol. 248 `[V]` | `[?]` | Paper | — | Unified restore stage | Not an upscaler |
| **PropMambaSR / CAMSR / SP-MoMamba** (2025–26 lightweight-SSM trio) | 2025/26 | Propagation SSM (IEEE 11417188), content-aware SSM (IEEE 11677129), superpixel **mixture of state-space experts** (arXiv 2605.25892) — all *lightweight* SR | IEEE 11417188, IEEE 11677129, arXiv 2605.25892 `[V]` | `[?]` | Paper only | Edge/efficient tier | Good fit in principle | **No weights for any of them** — not deployable today |
| **RWKV-style linear-attention SR** | 2025 | RWKV linear attention replaces softmax attention for SISR | arXiv 2502.00404 `[V]` | `[?]` | PyTorch | Cheaper than quadratic | Plausible efficiency play | Immature ecosystem |
| **MambaSR** | — | Arbitrary-scale Mamba SR | **Could not verify any canonical repo/paper** — treat the name as ambiguous | — | — | — | — | **Do not cite** |

**Honest read (Oct 2026):** the SSM line dominates *papers* but has produced **no widely-used, licensed, exe/ncnn/ONNX anime upscaler**. MambaRaw is the only 2026 SSM-adjacent SR entry with a public repo I found. For shipping, SSM is a 2027 bet. Surveys: "Mamba-based image restoration: a comprehensive survey" (2026) `[V]`, "Mamba in Vision" (arXiv 2410.03105) `[V]`.

## 3. Efficient / lightweight SR (the NTIRE-ESR lineage → 2026)

| Name | Year | Idea | Deploy | Notes |
|---|---|---|---|---|
| **SPAN** | 2024 | Swift parameter-free attention; strong PSNR/param at low latency | ONNX-friendly | IEEE 10678610 `[V]` |
| **PLKSR / PLKSR-Rep** | 2024 / **2026** | Partial large-kernel CNN; "Rep" reparameterised variant for mobile | ONNX/NPU-friendly | NTIRE 2026 mobile real-world SR entry `[V]` |
| **FANSR** | **2026** | Frequency-adaptive network for efficient SR | ONNX-friendly | IEEE 11461774 `[V]` |
| **DPAMixerSR** | **2026** | Degradation-pattern-aware efficient SR | Efficient tier | arXiv 2609.32705 (Sept 2026), explicitly targets real degradation `[V]` |
| **SMFANet, ELAN, SAFMN, CFAT, SRVGGNet** | 2022–2024 | Sub-1M-param efficient family | ONNX/ncnn | SRVGGNetCompact is the architecture behind Real-ESRGAN's compact models; PureScale 2.0 builds on ESRGAN/SAFMN/RealPLKSR `[V]` |
| **NAWQ-SR** | 2022 (old) | Hybrid-precision **NPU** engine for on-device SR (Meta) | Custom NPU | `FP32W8 32.04/0.8921` `[V]` — a rare published NPU-quantized SR number |

## 4. NTIRE / AIM / LoViF challenges 2024–2026 (what actually won)

| Challenge | Report | Notable teams / methods | Takeaway |
|---|---|---|---|
| **NTIRE 2024 / 2025 ESR (9th, 10th)** | arXiv 2404.10343, arXiv 2504.10686 `[V]` | Attention-guidance **distillation** nets (CVPRW 2024) `[V]`; a results row for **"HannahSR"** shows top PSNR at ~1.5M params `[V]` (*winner status unconfirmed*) | Reparameterisation + distillation + Transformer→CNN; sub-2M-param models now match full-size Transformers |
| **NTIRE 2026 ESR (11th)** | **arXiv 2604.03198**; CVPRW 2026 PDF `[V]` | Confirmed published | Efficiency track continues |
| **NTIRE 2026 — 1st Mobile Real-World SR** | arXiv 2604.17306; CVPRW 2026 `[V]` | PLKSR-Rep (compact large-kernel CNN); "Speed-Efficient Restoration with Compact High-Frequency" `[V]` | **The 2026 pivot: mobile + real degradation, not bicubic** |
| **NTIRE 2026 overall results** | CVPRW 2026 + press `[V]` | A Wuhan-University group (Chen Zhenzhong / Sun Wanjie) took **4 championships + 1 runner-up**; **Xiaomi took 3 awards incl. the portrait-restoration track championship** `[V]` | Industrial and university labs now dominate the restoration tracks |
| **LoViF 2026 — 1st & 2nd Real-World All-in-One Restoration** | arXiv 2604.19445 and **arXiv 2607.21118** `[V]` | Two challenge rounds in one year | All-in-one restoration now has its own benchmark series |
| **AIM 2025 Efficient Perceptual ISR** | arXiv 2510.12765 `[V]` | States perceptual quality *is* achievable under strict efficiency budgets `[V]` | Closest match to an anime/illustration tool's goals |

## 5. GAN-based practical SR (general photo)

| Name | Year | Idea | Repo / weights | License | Deploy | 4K-anime fit | Weaknesses |
|---|---|---|---|---|---|---|---|
| **Real-ESRGAN** | 2021 (old) | High-order degradation synthesis + U-Net discriminator w/ spectral norm | `xinntao/Real-ESRGAN`; `RealESRGAN_x4plus`, `x2plus`, `x4plus_anime_6B`, `realesr-animevideov3` `[V]` | **BSD-3-Clause** `[V]` | PyTorch, **ncnn-vulkan exe**, ONNX | **Best all-round default** | Over-sharpening halos; "plastic" texture; invents detail on illustration |
| **BSRGAN** | 2021 (old) | Randomised blur/resize/noise/JPEG degradation + GAN; ancestor ESRGAN/BasicSR (2018) added RRDB + relativistic avg GAN | `cszn/BSRGAN`, `XPixelGroup/BasicSR` `[?]` | `[?]` / Apache-2.0 `[?]` | PyTorch | Superseded; BasicSR still a training framework | Weaker real-world results; ESRGAN alone has severe artifacts on real degradation |
| **DASGAN** | 2025 | Degradation *generation* + adaptive GAN training for real-world SR | IEEE 11392932 `[V]` | `[?]` | Paper | Research | Unverified details |
| **Adversarial Diffusion Compression (AdcSR)** | 2025 CVPR | Distils diffusion SR into a compact adversarial net | CVPR 2025 `[V]` | `[?]` | Research | **Bridge from diffusion quality to non-diffusion speed** | Fresh; distillation-only |

## 6. Anime / illustration / manga / line-art SR

| Name | Year | Idea | Repo / weights | License | Deploy | 4K-anime fit | Weaknesses |
|---|---|---|---|---|---|---|---|
| **Anime4K** (shaders) | 2019 (old) | **GLSL shader** real-time chain, v4.x modes A/A+/B/B+/C. *Not a learned model* | `bloc97/Anime4K` `[V]` | **MIT** `[V]` | mpv/VLC shaders, Anime4KCPP | Best for live playback; zero weights | Line darkening/thickening, halos, kills grain, no true detail recovery |
| **"Anime4K: A Hybrid CNN–Transformer Network for Anime SR"** | 2025/26 `[?]` | A **separate learned anime SR paper** that happens to share the name | IEEE 11244719 `[V]` | `[?]` | Paper | Potentially the newest anime-specific architecture | **Name collision — do not conflate with the shaders**; weights unverified |
| **Re-Anime600** | **2026 CVPRW** | Curated benchmark dataset for **anime video quality assessment and SR evaluation** | CVPRW 2026 AIGENS (Fargetta et al.) `[V]` | `[?]` | Dataset/benchmark | **Use this to judge anime SR instead of guessing** | Benchmark, not a model; VQA-oriented |
| **APISR** | 2024 | Anime *production*-inspired degradation for real-world anime SR | `HikariDawn/APISR` `[V]` | **GPL-3.0** `[V]` | PyTorch → ONNX | Best 2024+ research-grade anime SR (2x/4x) | **GPL-3.0 is viral** — blocks closed bundling; over-sharpening |
| **realesr-animevideov3** | 2021/22 | Compact SRVGG net trained for **anime video** | Real-ESRGAN releases `[V]` | BSD-3-Clause `[V]` | ncnn-vulkan, ONNX, PyTorch | **Top pick** for 4K→8K anime batches | Softer than x4plus-anime; per-frame, not temporally stable |
| **RealESRGAN_x4plus_anime_6B** | 2021 | 6-block RRDB variant tuned on anime | Real-ESRGAN releases `[V]` | BSD-3-Clause `[V]` | ncnn-vulkan, ONNX, PyTorch | Best line-art crispness of the family | Halos; thickens/aliases thin lines; slower |
| **Real-CUGAN** | 2021 (old) | "Real Cascade U-Nets for Anime Image SR" (bilibili); conservative / denoise / pro variants | `bilibili/Real-CUGAN`; ModelScope `[V]` | `[?]` — unverified (a 2026 third-party note about an MIT change referred to *that downstream project*, not upstream) | ncnn-vulkan, PyTorch; integrated by Video2X `[V]` | Strong on compressed anime; community-favourite denoise levels | Colour/washed-out shifts at high denoise; over-smooths grain; `pro` vs `se` variant confusion |
| **waifu2x** | **2015 (old)** | Original anime CNN upscaler; `upconv_7`/`cunet`, denoise 0–3 | `nagadomi/waifu2x`, `waifu2x-ncnn-vulkan` `[V]` | MIT `[V]` | ncnn-vulkan exe, caffe | Still sane 2x for DVD-era sources | Ringing on hard edges; weak on real degradation; kills screentones at high denoise |
| **AnimeSR** | 2022 (old) | Real-world anime video SR; AIS dataset + temporal loss | `TencentARC/AnimeSR` `[V]` | repo `[?]`; **AVC dataset on request** `[V]` | PyTorch | Historically important (temporal) | Gated dataset; thin tooling; superseded |
| **Anime4KCPP / ACNetGLSL** | 2019–26 | C++ port of the Anime4K shader chains + own CNN models; ACNetGLSL re-implements those models in **GLSL** for real-time anime upscaling | `TianZerL/Anime4KCPP` v3.2.0, `TianZerL/ACNetGLSL` `[V]` | `[?]` | CLI/GUI, Vulkan, CPU, GLSL shader | Practical batch or real-time shader upscaling | Same shader artifacts; CNN models weaker than Real-ESRGAN; GLSL precision limits |
| **TheAnimeScripter** | 2024–26 | Free open-source anime video toolkit: AI upscale + RIFE interpolation + restore in one pass | `NevermindNilas/TheAnimeScripter` `[V]` | `[?]` | CLI, standalone, After Effects | **Best single harness for anime video pipelines** | Video-first; plugin licence unverified |
| **PureScale 2.0** | **2026** | Self-trained ESRGAN/SAFMN/RealPLKSR family on 11,750+ curated HD images | `limitlesslab/AI-upscaling-models` (release `PureScale2`) `[V]` | `[?]` | `.pth` / chaiNNer | Photorealistic-oriented; light on anime line art | Trained for textures (skin, fabric, grit), not anime; license unverified |
| **openmodeldb** | 2022– | Community index of upscaling models with architecture/scale metadata | openmodeldb.info `[V]` | index only | — | **Where to find new anime models (e.g. 2x Adore)** | Metadata quality varies; license is per-model |
| **GlyphSR / StyleSRN** | 2025 | Glyph-prior and text-style-embedding SR for text images | AAAI 2025; ICCV 2025 `[V]` | `[?]` | Research | Relevant to **text/line preservation** | Text domains, not anime |

**Manga / screentone note.** No 2025–2026 flagship *manga* SR model surfaced; the domain is served by anime models plus de-halftone preprocessing. Screentone **moiré** is the recurring failure — GAN upscalers turn halftone dots into banding or grey mush; de-halftone/denoise *before* upscaling, never after. Commercial manga/webtoon upscalers exist (e.g. LlamaGen's public manga upscaler `[V]`) but publish no model details.

## 7. Degradation-aware restoration and all-in-one models (compose with SR)

Practitioner consensus on order: **denoise → deband/deblock → upscale → (optional) light re-sharpen**.

| Name | Year | One-line idea | Repo / weights | License | Deploy | CPU vs GPU | Role with an upscaler | Weaknesses |
|---|---|---|---|---|---|---|---|---|
| **SCUNet** | 2022 CVPR | Swin-Conv-UNet for *blind real* denoising incl. JPEG | `cszn/SCUNet` `[?]`; ONNX `Heliosoph/scunet-onnx` `[V]` | **Apache-2.0** `[V]` (ONNX card) | ONNX, PyTorch, chaiNNer | GPU-preferred; ONNX CPU OK at low res | **Best pre-upscale denoiser**; chaiNNer tiles it `[V]` | Plasticises grain; over-smooths at high noise; tile seams if overlap wrong |
| **NAFNet** | 2022 ECCV | Nonlinear-activation-free blocks: cheap, strong deblur/denoise | `megvii-research/NAFNet` `[?]`; ONNX/OpenCV ports `[V]` | `[?]` | ONNX, PyTorch, OpenCV sample | Fast; ONNX-friendly | General deblur/denoise pre-stage | Less blind-robust than SCUNet on unknown JPEG |
| **Restormer** | 2021 (old) | Channel-wise transformer restoration baseline | `swz30/Restormer` `[?]` | `[?]` — **verify before commercial use** | PyTorch | GPU | Baseline only | Slow; license ambiguity |
| **FBCNN** | 2021 (old) | Flexible **blind JPEG** artifact removal with predicted quality factor | ICCV 2021 `[V]` | `[?]` | PyTorch | GPU | Targeted deblock for JPEG-crushed anime | JPEG only; can smooth |
| **InstructIR** | 2024 ECCV | Restoration conditioned on a **human text instruction** | `mv-lab/InstructIR` `[V]` | `[?]` | PyTorch, HF Space | GPU | One model, many degradations | Prompting awkward to automate; hallucination risk |
| **AutoDIR** | 2024 ECCV | Automatic all-in-one blind restoration (latent diffusion) | `jiangyitong/AutoDIR` `[V]` | `[?]` | PyTorch | GPU | Fully automatic pre-stage | Diffusion → nondeterministic, slow, can invent detail |
| **R2R (Retrieve-to-Restore)** | **2026 CVPR** | Retrieval-based degradation bank for efficient all-in-one restoration | `cscxwang/R2R` `[V]` | `[?]` | PyTorch | Efficient tier | **Newest credible unified pre-stage** | Not an upscaler; no anime validation |
| **ProMamba** | **2026** | Prompt-guided Mamba all-in-one restoration | Signal Processing 248 `[V]` | `[?]` | Paper | — | Unified restoration | No repo found |
| **LoViF 2026 challenge entries** | **2026** | Real-world all-in-one restoration, two challenge rounds | arXiv 2604.19445, 2607.21118 `[V]` | varies | Research | — | Best current read on all-in-one SOTA | Challenge-scoped; no anime domain |
| **OmniSSR** | — | **Could not verify this name at all.** The only "OmniSR" found is *shadow removal* (arXiv 2410.01719) `[V]` | — | — | — | — | — | Likely a confusion with **Omni-SR** (§1) — **do not cite** |

### 7b. Non-ML classical primitives (still the right tool for banding/blocking)

- **Deband:** `f3kdb` / `neo_f3kdb`, `vs-deband`; **Deblock:** `deblockpp7` / `vsdeblock` (VapourSynth). CPU-only, free, and better than any neural net at flat-gradient banding in anime skies — and it matters because GAN upscalers convert banding into contour ridges and blocking into stair-step halos. Fix at source, not after.

## 8. How quality is judged in practice

- **Re-Anime600 (CVPRW 2026) is the new anime-specific reference** for both VQA and SR evaluation `[V]` — prefer it over generic DIV2K/Urban100 when claiming anime quality.
- **Blind A/B remains the community standard:** stills at 100–400 % on line edges, screentones, flat gradients and dark scenes, plus a short motion clip for temporal consistency.
- **PSNR/SSIM reward blur**; they belong to the fidelity tier (§1–4) and are misleading for GAN/SSM output. **Perceptual/learned metrics:** LPIPS, DISTS, FID/KID; NR-IQA: NIQE, BRISQUE, CLIP-IQA, MANIQA, TOPIQ (a 2026 reference work on SR quality assessment, metrics and validation exists `[V]`).
- **Documented metric failure:** CVPR 2026 supplemental notes NR-IQA metrics (PI, NIQE, CLIP-IQA, MANIQA) "can fail on cases where subtle structure" is precisely what matters `[V]`; NTIRE 2026 works likewise warn that "directly optimizing IQA-related objectives may overfit" `[V]`.
- **Video:** VMAF is the practical judge; per-frame metrics miss temporal flicker — the actual complaint about per-frame anime models. One-step distilled VSR is an active 2026 direction (DUO-VSR, CVPR 2026) `[V]`.
- **Defect checklist:** over-smoothing (lost grain/screentone), ringing (Gibbs at hard edges), seam lines (tiling), colour shift (chroma handling, esp. Real-CUGAN/Anime4K), over-sharpening halos (Real-ESRGAN family), line thickening/aliasing, hallucinated detail.

## 9. Deployment matrix (engine × platform × speed × quality)

| Engine / route | Platform | Speed (RTX 4070S class) | Setup friction | Quality ceiling | Notes |
|---|---|---|---|---|---|
| **ncnn-vulkan** (Real-ESRGAN/waifu2x/Real-CUGAN exes) | Win/Linux/mac, Vulkan GPU; CPU fallback | **Very fast**, fp16, no Python | **Lowest** — exe + model dir | Good (GAN models) | Seam risk; `-t` tile size, `-x` TTA (slower, marginal gain) `[V]` |
| **ONNX Runtime + CUDA EP** | Win/Linux, NVIDIA | Fast; fp16 supported | Low (pip) | Full model quality | Best dev-loop default on the 4070S. ORT at **1.24.1 release / 1.26.0 (2026)** `[V]` |
| **ONNX Runtime + TensorRT EP** | Win/Linux, NVIDIA | **Fastest**; fp16/int8 | High — engine build per shape | fp16 ≈ FP32; int8 needs calibration | Worth it for fixed 4K→8K batches; prebuilt engines circulate (e.g. CCSR-TensorRT-Engine) `[V]` |
| **ONNX Runtime + DirectML EP** | Windows any GPU | Moderate; below CUDA | Low | fp32 solid; **fp16 op coverage partial** `[V]` | Vendor-neutral fallback; Microsoft ships a DirectML SR sample `[V]` |
| **ONNX Runtime + OpenVINO EP** | Intel CPU/iGPU/**NPU** | Good on Intel; NPU modest | Medium | fp16/int8 | Intel documents NPU-accelerated upscaling `[V]` |
| **ONNX Runtime + QNN EP** | Snapdragon X / Hexagon NPU | Adequate for single 1080p→4K | Medium | int8-centric | Qualcomm publishes an optimized **`Real-ESRGAN-x4plus`** (v0.45.0) plus a `super_resolution_windows_cpp` app `[V]` |
| **Windows ML 2 / Windows AI APIs** | Copilot+ PCs; auto EP selection incl. NPU vendors | NPU-resident **Image Super-Resolution API** | Low (platform API) | Fixed, OS-provided models | Documented supported-EP matrix `[V]`; great for app integration but **no custom anime model, no batch 8K** `[V]` |
| **Windows Auto SR (Automatic Super Resolution)** | Copilot+ NPU — Qualcomm, then AMD, then **Intel Core Ultra Series 3 (Sept 2026)** `[V]` | Real-time | Zero (OS toggle) | Tuned for **video playback + games** | Microsoft's RTX-VSR analogue; **not an offline image batch tool** `[V]` |
| **NVIDIA RTX Video Super Resolution** | RTX 20/30/40/50, **video playback only** | Real-time 4K | Zero | Tuned for compression artifacts | v1.5 added RTX 20-series (2024) `[V]`; browser/player integration `[V]` |
| **NVIDIA Maxine / Video Effects SDK (VSR effect)** | RTX, Win/Linux, C/C++ + `nvidia-vfx` wheel | Real-time, up to 4x `[V]` | Medium; NVIDIA SDK licence | Broadcast-grade | Offline-usable via SDK/`nvidia-vfx` `[V]`; ComfyUI wrapper exists `[V]`; **proprietary, redistribution-restricted** |
| **PyTorch + CUDA** / **WebGPU (onnxruntime-web)** | Win/Linux; browser for WebGPU | Fastest for research models, slowest per-image overhead; WebGPU slow and memory-limited | Medium (env) / Low | Highest (any model) / limited model set | PyTorch is the only route for ATD/HAT/MambaIR/APISR/R2R; WebGPU is experimental, not for 8K batches `[V]` |

### 9b. Tiling, seams, and quantisation — the two things that actually break outputs

**Tiling.**
- Real-ESRGAN ncnn pads each tile with a **16-px margin and blends** overlaps; tile-boundary structure is still reported — issues **#73 "Output image subdivided in zoomed tiles"**, **#197 "Tile-Size Averaging"** `[V]`, and Upscayl **discussion #600** `[V]`. In 2026 this is still an active research topic: IEEE papers on **SR processors with tiling artifact reduction** (11516490, 11377884) were published this year `[V]`.
- Residual seams come from **padding mode**, not overlap size — safe recipe: **overlap ≥ 2× the effective receptive field, reflection padding, linear/cosine feather blend, per-tile mean/variance matching**. Harnesses that do this for you: `ComfyUI-Advanced-Tile-Processing` ("intelligent tiling and seamless merging") `[V]` and chaiNNer (tiles large images for SCUNet `[V]`). **12 GB rule of thumb:** at x4, input tiles ≤ 512–768 px for Transformer/SSM, ≤ 1024–1536 px for compact CNN/GAN, always with overlap.

**Quantisation.**
- **fp16 is effectively free** on CUDA/TensorRT: ~2× memory saving, ~1.5–2× speed, negligible quality change. Prefer it everywhere.
- **int8 for SR is still unsolved.** SR is a regression task with wide dynamic range in the residual; naive per-tensor PTQ visibly degrades edges. Evidence it remains open: *Toward Accurate PTQ for Image SR* (CVPR 2023) `[V]`, **INT8 SISR via deployment-aware quantization + teacher-guided training (CVPRW 2026, MAI)** `[V]`, and **TAQ: static-deployable temporal-aware quantization for real-world VSR (2026)** `[V]`. Hybrid precision helps — NAWQ-SR `FP32W8` reported `32.04/0.8921` `[V]`.
- **Don't** apply ORT dynamic int8 quantisation to a GAN SR generator and expect matching output — discriminator-tuned high-frequency detail is exactly what quantisation noise destroys.

## 10. Practical recommendations — Windows + Python 3.10 + onnxruntime/ncnn

1. **Baseline batch pipeline (fastest to ship):** `realesr-animevideov3` via **ncnn-vulkan exe**, tile size tuned per source resolution, light Lanczos downscale. Zero Python.
2. **Python-native pipeline:** carry models as **ONNX**, run with **onnxruntime-gpu** `CUDAExecutionProvider` (ORT ~1.24/1.26, 2026) for the 4070 SUPER, `DmlExecutionProvider` as portable fallback. Pin `onnxruntime-gpu` and CUDA/cuDNN together — EP/provider mismatch is the top install failure.
3. **Model shortlist for 4K+ anime on 12 GB:** crisp line art → `RealESRGAN_x4plus_anime_6B` (BSD-3-Clause, bundleable) · conservative 2x → `2x-AnimeSharpV4` (verify license) · compressed DVD/BD → **Real-CUGAN** `denoise3x` pre-upscale (verify license) · best research quality, GPL OK → **APISR** 2x/4x · pre-stage denoise → **SCUNet** (Apache-2.0 ONNX = safest to redistribute).
4. **Composition recipe (safe order):** `f3kdb`/`vs-deband` → SCUNet-light or NAFNet denoise → Real-CUGAN/animevideov3 at 2x → final 2x with x4plus-anime_6B → optional 5 % unsharp + grain re-synthesis. Never upscale first and denoise after.
5. **Tiling defaults:** overlap 64–128 px at x4 with feather blending; enable tiling *below* the VRAM cliff rather than retrying after an OOM; disable TTA (`-x`) for batches. Throughput expectation: compact CNN/ncnn models do 4K→8K stills in seconds and thousands of 1080p→4K frames/hour on a 4070 SUPER-class GPU; full RRDB/Transformer models drop to seconds-to-minutes per 4K image. *(Order-of-magnitude only — no published 4070S benchmark verified this session.)*
6. **Anime video: don't hand-roll it.** Use `TheAnimeScripter` (upscale + RIFE + restore; CLI/standalone/AE) or `Video2X`; add `Anime4KCPP`/`ACNetGLSL` shaders when you need real-time rather than archival quality.
7. **NPU path:** relevant for app integration (Windows AI APIs, Windows ML 2 auto-EP, Qualcomm's optimized Real-ESRGAN). **Playback-only alternatives:** Windows Auto SR (three NPU platforms as of Sept 2026) and RTX Video Super Resolution. Keep CUDA as the quality reference for everything.
8. **Quality gate:** build a 20-crop regression set (line edge, screentone, flat sky gradient, dark scene, text overlay, motion blur) plus a blind A/B harness — and for anime *video*, evaluate against **Re-Anime600**. Never accept a model change on PSNR alone; watch for seam lines, banding ridges, chroma drift and halos.
9. **Licensing triage before shipping:** BSD-3-Clause (Real-ESRGAN) and Apache-2.0 (SwinIR, MambaIR, SCUNet ONNX card) are safe; **GPL-3.0 (APISR) is viral**; 2x-AnimeSharpV4, Real-CUGAN, AutoDIR, InstructIR, Restormer and R2R were **not license-verified here** — open the repo LICENSE before bundling.

## Sources

1. [MambaIR: A Simple Baseline for Image Restoration with State-Space Model — ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02740.pdf) (2024)
2. [MambaIRv2: Attentive State Space Restoration — arXiv 2411.15269](https://ar5iv.labs.arxiv.org/html/2411.15269) (2024/2025)
3. [csguoh/mambair — official MambaIR + MambaIRv2 repo](https://github.com/csguoh/mambair) (2025)
4. [cguoh/MambaIR model card — license: apache-2.0](https://huggingface.co/cguoh/MambaIR/blob/main/README.md?code=true) (2025)
5. [EDMT: Efficient Hybrid Mamba-Transformer with Dual-Path Selective Scan for Lightweight SR — IEEE](https://ieeexplore.ieee.org/abstract/document/11633820) (2026)
6. [State prior Mamba for image super-resolution — Computers & Graphics](https://www.sciencedirect.com/science/article/abs/pii/S0097849326001482) (2026)
7. [MambaRaw: Selective State Space Modeling for Efficient 4K RAW Image Reconstruction — ECCV 2026](https://github.com/Peizeli1/MambaRaw) (2026)
8. [ProMamba: Prompts guided Mamba for all-in-one image restoration — Signal Processing 248](https://www.sciencedirect.com/science/article/abs/pii/S0165168426002161) (2026)
9. [Exploring Linear Attention Alternative for Single Image Super-Resolution (RWKV-style) — arXiv 2502.00404](https://arxiv-org.ezproxy.obspm.fr/html/2502.00404v2) (2025)
10. [Mamba-based image restoration: a comprehensive survey — J. Image and Graphics](http://www.cjig.cn/en/article/doi/10.11834/jig.260111/) (2026)
11. [Omni Aggregation Networks for Lightweight Image Super-Resolution — CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Omni_Aggregation_Networks_for_Lightweight_Image_Super-Resolution_CVPR_2023_paper.pdf) (2023)
12. [SRFormer: Permuted Self-Attention for SISR — ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_SRFormer_Permuted_Self-Attention_for_Single_Image_Super-Resolution_ICCV_2023_paper.html) (2023)
13. [HAT: Activating More Pixels in Image Super-Resolution Transformer — CVPR 2023](https://www.openaccess.thecvf.com/content/CVPR2023/html/Chen_Activating_More_Pixels_in_Image_Super-Resolution_Transformer_CVPR_2023_paper.html) (2023)
14. [DAT: Dual Aggregation Transformer for Image Super-Resolution — ICCV 2023](https://www.openaccess.thecvf.com/content/ICCV2023/html/Chen_Dual_Aggregation_Transformer_for_Image_Super-Resolution_ICCV_2023_paper.html) (2023)
15. [ATD: Adaptive Token Dictionary for Image Restoration — paper page](https://huggingface.co/papers/2401.08209) (2024)
16. [Heliosoph/swinir-onnx — license: apache-2.0](https://huggingface.co/Heliosoph/swinir-onnx/blob/d2500929641e9bb1e03f7ab83793ba59c307fb06/README.md?code=true) (2023/2024)
17. [IAFMNet: Information-Aware Feature Modulation for Efficient Super-Resolution — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_IAFMNet_Information-Aware_Feature_Modulation_for_Efficient_Super-Resolution_CVPR_2026_paper.html) (2026)
18. [FANSR: Frequency Adaptive Network for Efficient Image Super-Resolution — IEEE](https://xplorestaging.ieee.org/document/11461774) (2026)
19. [DPAMixerSR: An Efficient Degradation-Pattern-Aware Model for Image Super-Resolution — arXiv 2609.32705](https://arxiv-org.ezproxy.obspm.fr/abs/2609.32705) (2026)
20. [Swift Parameter-free Attention Network for Efficient SR (SPAN) — IEEE](https://ieeexplore.ieee.org/document/10678610) (2024)
21. [Partial Large Kernel CNNs for Efficient Super-Resolution (PLKSR) — IEEE](https://ieeexplore.ieee.org/ielx8/6287639/11323511/11493906.pdf) (2024/2025)
22. [NAWQ-SR: A Hybrid-Precision NPU Engine for Efficient On-Device Super-Resolution — arXiv 2212.09501](https://ar5iv.labs.arxiv.org/html/2212.09501) (2022)
23. [The Tenth NTIRE 2025 Efficient Super-Resolution Challenge Report — arXiv 2504.10686](https://ar5iv.labs.arxiv.org/html/2504.10686) (2025)
24. [The Eleventh NTIRE 2026 Efficient Super-Resolution Challenge Report — arXiv 2604.03198](https://ar5iv.labs.arxiv.org/html/2604.03198) (2026)
25. [The First Challenge on Mobile Real-World Image Super-Resolution at NTIRE 2026 — arXiv 2604.17306](https://www.semanticscholar.org/paper/The-First-Challenge-on-Mobile-Real-World-Image-at-Li-Chen/d6ae5bc0f2c2a6a445b4f69784327d9489156005) (2026)
26. [PLKSR-Rep: A Compact Large-Kernel CNN for Mobile Real-World Image SR — NTIRE 2026](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Deng_PLKSR-Rep_A_Compact_Large-Kernel_CNN_for_Mobile_Real-World_Image_Super-Resolution_CVPRW_2026_paper.pdf) (2026)
27. [Speed-Efficient Restoration with Compact High-Frequency for On-Device SR — NTIRE 2026](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Im_Speed-Efficient_Restoration_with_Compact_High-Frequency_for_On-Device_Super-Resolution_CVPRW_2026_paper.pdf) (2026)
28. [Chen Zhenzhong / Sun Wanjie group wins 4 championships + 1 runner-up at CVPR 2026 NTIRE](https://mp.weixin.qq.com/s/yEyZ75_YG59bexaSZI1__A) (2026)
29. [Xiaomi wins three CVPR 2026 NTIRE awards incl. portrait-restoration champion](https://www.ithome.com/0/952/050.htm) (2026)
30. [The Second LoViF 2026 Challenge on Real-World All-in-One Image Restoration — arXiv 2607.21118](https://ar5iv.labs.arxiv.org/html/2607.21118) (2026)
31. [LoViF 2026 Challenge on Real-World All-in-One Image Restoration — arXiv 2604.19445](https://arxiv.org/pdf/2604.19445) (2026)
32. [Efficient Perceptual Image Super Resolution: AIM 2025 Study and Benchmark — arXiv 2510.12765](https://ar5iv.labs.arxiv.org/html/2510.12765) (2025)
33. [xinntao/Real-ESRGAN — LICENSE (BSD-3-Clause)](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) (2021)
34. [Real-ESRGAN ncnn executable guide](https://mintlify.wiki/xinntao/Real-ESRGAN/guides/ncnn-executable) (2022)
35. [Real-ESRGAN-ncnn-vulkan issue #73 — "Output image subdivided in zoomed tiles"](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/issues/73) (2021)
36. [Real-ESRGAN issue #197 — "Feature request: Tile-Size Averaging"](https://github.com/xinntao/Real-ESRGAN/issues/197) (2021)
37. [Upscayl discussion #600 — "Images are rendering with tiles"](https://github.com/orgs/upscayl/discussions/600) (2024)
38. [chaiNNer image processing docs — tiling behaviour](https://deepwiki.com/chaiNNer-org/chaiNNer/5.4-image-processing) (2024)
39. [Re-Anime600: A Curated Benchmark Dataset for Anime Video Quality Assessment and SR Evaluation — CVPRW 2026 (AIGENS)](https://openaccess.thecvf.com/content/CVPR2026W/AIGENS/papers/Fargetta_Re-Anime600_A_Curated_Benchmark_Dataset_for_Anime_Video_Quality_Assessment_CVPRW_2026_paper.pdf) (2026)
40. [Anime4K: A Hybrid CNN–Transformer Network for Anime Super-Resolution — IEEE](https://ieeexplore.ieee.org/abstract/document/11244719) (2025/2026)
41. [Anime4K (bloc97) — LICENSE (MIT)](https://github.com/bloc97/Anime4K/blob/master/LICENSE) (2019)
42. [Anime4K GLSL / MPV advanced instructions (v4.x modes)](https://raw.githubusercontent.com/bloc97/Anime4K/master/md/GLSL_Instructions_Advanced.md) (2023)
43. [Anime4KCPP v3.2.0 release](https://newreleases.io/project/github/TianZerL/Anime4KCPP/release/v3.2.0) (2026)
44. [ACNetGLSL — Anime4KCPP deep-learning models re-implemented in GLSL](https://github.com/TianZerL/ACNetGLSL) (2024/2026)
45. [TheAnimeScripter — open-source anime video toolkit (upscale + RIFE + restore)](https://github.com/NevermindNilas/TheAnimeScripter) (2026)
46. [APISR: Anime Production Inspired Real-World Anime Super-Resolution — arXiv 2403.01598](https://ar5iv.labs.arxiv.org/html/2403.01598) (2024)
47. [HikariDawn/APISR model card — license: gpl-3.0](https://huggingface.co/HikariDawn/APISR/blob/main/README.md?code=true) (2024)
48. [AnimeSR: Learning Real-World Super-Resolution Models for Animation Videos — NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/file/48cca987b3af66e1a607abd4820b330d-Paper-Conference.pdf) (2022)
49. [Kim2091/2x-AnimeSharpV4 model card](https://huggingface.co/Kim2091/2x-AnimeSharpV4/blob/main/README.md?code=true) (2023)
50. [Real Cascade U-Nets for Anime Image Super Resolution — Real-CUGAN (ModelScope)](https://www.modelscope.cn/models/bilibili/cv_bilibili_image-super-resolution/resolve/master/README_CH.md) (2021)
51. [waifu2x-ncnn-vulkan package (Arch Linux)](https://archlinux.org/packages/extra/x86_64/waifu2x-ncnn-vulkan/) (2025)
52. [k4yt3x/video2x — anime video upscaling pipeline](https://github.com/k4yt3x/video2x) (2024/2025)
53. [PureScale 2.0 release — limitlesslab/AI-upscaling-models](https://github.com/limitlesslab/AI-upscaling-models/releases/tag/PureScale2) (2026)
54. [openmodeldb — community upscaling model index](https://openmodeldb.info/models/2x-Adore) (2022/2026)
55. [GlyphSR — AAAI 2025 (text-image SR with glyph prior)](https://dlnext.acm.org/doi/10.1609/aaai.v39i8.32893) (2025)
56. [SCUNet ONNX model card — license: apache-2.0](https://huggingface.co/api/resolve-cache/models/Heliosoph/scunet-onnx/56e26da0a8c15970eb4b9d803cb43d20783e51a3/README.md) (2024)
57. [FBCNN: Towards Flexible Blind JPEG Artifacts Removal — ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Towards_Flexible_Blind_JPEG_Artifacts_Removal_ICCV_2021_paper.pdf) (2021)
58. [AutoDIR: Automatic All-in-One Image Restoration with Latent Diffusion — ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05684.pdf) (2024)
59. [mv-lab/InstructIR — ECCV 2024 human-instruction restoration](https://github.com/mv-lab/instructir) (2024)
60. [Retrieve-to-Restore: Efficient All-in-One Image Restoration with a Retrieval-Based Degradation Bank — CVPR 2026](https://www.openaccess.thecvf.com/content/CVPR2026/html/Wang_Retrieve-to-Restore_Efficient_All-in-One_Image_Restoration_with_a_Retrieval-Based_Degradation_Bank_CVPR_2026_paper.html) (2026)
61. [Degradation-Aware Adaptive Context Gating for Unified Image Restoration — Semantic Scholar](https://www.semanticscholar.org/paper/Degradation-Aware-Adaptive-Context-Gating-for-Image-He-Chu/733ea13e27ba7b7d16d2b6cc4d14c33e78e3d40f) (2026)
62. [DASGAN: Degradation Generation and Adaptive GAN Training for Real-World SR — IEEE](https://ieeexplore.ieee.org/document/11392932) (2025)
63. [Adversarial Diffusion Compression for Real-World Image SR — CVPR 2025](https://openaccess.thecvf.com//content/CVPR2025/html/Chen_Adversarial_Diffusion_Compression_for_Real-World_Image_Super-Resolution_CVPR_2025_paper.html) (2025)
64. [Toward Accurate Post-Training Quantization for Image Super Resolution — CVPR 2023 supplemental](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Tu_Toward_Accurate_Post-Training_CVPR_2023_supplemental.pdf) (2023)
65. [Efficient INT8 SISR via Deployment-Aware Quantization and Teacher-Guided Training — CVPRW 2026 (MAI)](https://openaccess.thecvf.com/content/CVPR2026W/MAI/html/Nguyen_Efficient_INT8_Single-Image_Super-Resolution_via_Deployment-Aware_Quantization_and_Teacher-Guided_Training_CVPRW_2026_paper.html) (2026)
66. [TAQ: Static-Deployable Temporal-Aware Quantization for Real-World Video Super-Resolution](https://eccv2024.ecva.net/virtual/2026/poster/5539) (2026)
67. [An Energy-Efficient Block-Convolution-Based Super-Resolution Processor With Tiling Artifact Reduction — IEEE](https://ieeexplore.ieee.org/document/11516490) (2026)
68. [Choosing Execution Providers for Performance — ONNX Runtime docs](https://pkreg101.github.io/onnxruntime/docs/performance/choosing-execution-providers.html) (2026)
69. [onnxruntime 1.26.0 on PyPI](https://newreleases.io/project/pypi/onnxruntime/release/1.26.0) (2026)
70. [Windows ML execution providers — Microsoft Learn](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers?tabs=winml2) (2026)
71. [What are Windows AI APIs? — Microsoft Learn (Image Super Resolution API)](https://learn.microsoft.com/en-us/windows/ai/apis/) (2025/2026)
72. [Automatic Super Resolution (Auto SR) — Microsoft Support](https://support.microsoft.com/en-us/windows/ai/ai-features/automatic-super-resolution) (2026)
73. [Auto SR expands to Intel Core Ultra Series 3 — third NPU platform](https://www.ixbt.com/news/2026/09/18/436724-apskeiler-autosr-vstroennyi-priamo-v-windows-11-teper-podderzivaet-processory-intel-core-ultra-series-3.html) (2026)
74. [DirectML Super-Resolution sample — microsoft/DirectML](https://raw.githubusercontent.com/microsoft/DirectML/master/Samples/DirectMLSuperResolution/Readme.docx) (2021/2022)
75. [Enhance AI Upscaling with Intel AI Boost NPU — Intel developer article](https://www.intel.com/content/www/us/en/developer/articles/technical/enhance-ai-upscaling-with-intel-ai-boost-npu.html) (2025)
76. [qualcomm/Real-ESRGAN-x4plus — Snapdragon-optimized model card](https://huggingface.co/qualcomm/Real-ESRGAN-x4plus/tree/v0.45.0) (2026)
77. [ussoewwin/CCSR-TensorRT-Engine](https://huggingface.co/ussoewwin/CCSR-TensorRT-Engine/blob/main/README.md) (2026)
78. [nvidia-vfx Python wheel — NVIDIA Video Effects / Maxine VSR bindings](https://pypi.org/project/nvidia-vfx/0.2.0.0/) (2025)
79. [NVIDIA Maxine Video Effects SDK user guide (Super Resolution effect)](https://docs.nvidia.com/maxine/vfx/1.3.0/nvidia-vfx-sdk-user-guide.pdf) (2024)
80. [NVIDIA AI upscaling in browsers (RTX VSR launch, Chrome/Edge, 30/40-series) — The Verge](https://www.theverge.com/2023/1/4/23538584/nvidia-ai-upscaling-browser-chrome-edge-30-40-series-gpu-graphics-cards-4k-1080p) (2023)
81. [NVIDIA VSR 1.5 arrives on RTX 20-series](https://www.tomshw.it/hardware/nvidia-vsr-arriva-sulla-rtx-serie-20-con-la-versione-15) (2024)
82. [NR-IQA metrics "can fail on cases where subtle structure…" — CVPR 2026 supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Su_Bridging_the_Perception_CVPR_2026_supplemental.pdf) (2026)
83. [Super-Resolution Image Quality: Assessment, Metrics, and Validation — 2026 reference work](https://www.sciencedirect.com/science/chapter/referencework/abs/pii/B9780443342035000346) (2026)
84. [DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution — CVPR 2026](https://github.com/zhaoyang97/Paper-Notes/blob/main/docs/CVPR2026/image_generation/duo-vsr_dual-stream_distillation_for_one-step_video_super-resolution.md) (2026)
