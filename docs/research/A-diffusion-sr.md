# Generative SR: Diffusion, Flow-Matching and One-Step Super-Resolution — State of the Art (October 2026)

**Scope.** Multi-step diffusion SR, one-step/distilled SR, DiT & flow-matching SR, video SR where it informs image SR, degradation modelling, GAN-vs-diffusion trade-offs, faithfulness control, tiling/VAE artifacts, benchmarks and metrics. **As-of date: 2026-10-11**; anything dated 2026 is treated as the current frontier, and §3 leads with Sept–Oct 2026 work.
**Method note.** `web_fetch` was DNS-blocked in this sandbox (every host resolved to a non-public IP) and direct TLS from the shell was blocked (`SEC_E_NO_CREDENTIALS`). All findings come from **web search result titles/snippets** only; primary-source HTML/PDF could not be read. Items I could not confirm from a primary source are marked **[unverified]**. Treat numbers as directional.

---

## TL;DR

- **The field has collapsed to one step.** By 2026 the default for real-world SR is a *single* network evaluation of an SD/SDXL/DiT backbone with LoRA or a distilled student, at ~1.3–1.5 s per 1024² image (OSEDiff 1.3 s, S3Diff 1.4 s, reported in a CVPR 2026 comparison). Multi-step SAM/SDXL pipelines (SUPIR, DiffBIR, StableSR) are now legacy for interactive use.
- **Quality is bought with fidelity, explicitly and controllably.** 2025–2026 work is dominated by *fidelity–realism dials* (PiSA-SR Dual-LoRA CVPR 2025, OFTSR ICLR 2026 "tunable trade-offs", IFCSR CVPR 2026 "inference-free control", arXiv 2512.14061). One-step SR sits at **PSNR ~22–25 dB on DIV2K-val** — far below regression SR, by design.
- **Backbone migration: UNet → DiT → flow matching is now complete at the frontier.** DiT4SR (ICCV 2025) started it; FlowSR (ICCV 2025, consistency rectified flow), OFTSR (ICLR 2026), **StrSR and Adversarial Score Distillation (both ECCV 2026)**, Fill2SR and TOLA (arXiv Sep 2026), TEASR (ECCV 2026), PixelIR / MeanSR / DriftSR (arXiv Aug–Oct 2026) continue it. **An SD3-Medium-based SR model with open weights now exists (`dipta007/OracleZoom`), plus a flow-based `TensorForger/FlowUpscaler`** — the earlier "no SD3/FLUX-SR checkpoint" gap is closing. Large-scale FLUX-based SR is still not mainstream **[unverified]**.
- **Sept–Oct 2026 is the live edge.** New since the Aug-2026 snapshot: **Super-Resolution in the Right Latent Space** (arXiv 2610.04781, frozen vision-foundation substrate), **DriftSR** (arXiv 2610.04819, one-step distribution drifting), **DLC** (arXiv 2610.11433, metric-guided dynamic loss controller for multi-objective restoration), **Fill2SR** (arXiv 2609.33582, repurposed inpainting DiTs), **TOLA** (arXiv 2609.29240, text-image SR), **CLIMB-flow** (arXiv 2609.33834, coupled linear-inverse posterior flow sampling), and **ECCV 2026** (Oct 2026) papers StrSR and Adversarial Score Distillation. **NVIDIA PiD is confirmed NeurIPS 2026** with an official repo (`nv-tlabs/PiD`).
- **Open weights are near-universal; permissive licensing is not.** SeedVR2 is the clean case (**Apache-2.0**, HF `ByteDance-Seed/SeedVR2-3B` / `-7B`). StableSR is `license: other`; SUPIR inherits SDXL OpenRAIL++ and LLaVA terms; several SR repos are research-only.
- **12 GB (RTX 4070 SUPER) is genuinely workable — for one-step / 1024²-class models only, and CPU is a dead end.** SUPIR at native settings is documented to **OOM on 12 GB** (kijai/ComfyUI-SUPIR issue #12); OSEDiff/S3Diff/InvSR/ResShift-class run in fp16 with tiling; SeedVR2 3B and FlashVSR advertise 12 GB-class operation. One fp32 UNet step at 1024² is tens of seconds to minutes on a desktop CPU, so reserve CPU for the non-diffusion ONNX fallback (Real-ESRGAN/SwinIR-class).
- **ONNX/ncnn covers the *non*-diffusion half of the toolkit.** Real-ESRGAN (official conversion tooling + ncnn-Vulkan), SwinIR (community `swinir-onnx`), Real-CUGAN/waifu2x (ncnn-Vulkan), PiD ONNX/TensorRT export. Diffusion SR has **no maintained official ONNX path**; runtime-exported graphs are fragile (control flow, CFG, tiling, custom samplers).
- **GAN SR is not dead.** For 4K+ throughput, batch processing, and anime/clean-input upscaling it still wins on speed-per-quality by orders of magnitude; the diffusion family wins on degraded/real-world inputs where a generative prior is actually needed.
- **Degradation modelling is the real bottleneck.** Real-ESRGAN's high-order pipeline (2021) is still the default synthetic training recipe; 2025–2026 work attacks it via *learned/continuous* degradation (AAAI 2026 latent flow-matching degradation; arXiv 2508.07214 rectified-flow degradation modelling) — i.e. the field now treats the pipeline itself as the thing to learn.
- **Metrics are the weakest link and the field knows it.** NTIRE 2026's own PLKSR-Rep paper warns that "directly optimizing IQA-related objectives may overfit"; a CVPR 2026 supplement shows PI, NIQE, CLIP-IQA and MANIQA "can fail" on subtle-structure cases. CLIPIQA/MUSIQ/MANIQA reward the AI-sharp look; NIQE is near-useless on generative output.
- **Video SR is now the innovation engine for image SR.** Adversarial post-training to one step (SeedVR2), block/streaming attention and cache routing (ReCaVSR, FlashVSR) and dual-stream distillation (DUO-VSR CVPR 2026) are all more aggressive than the image-SR literature.

---

## 1. Multi-step diffusion SR (2023–2024; now legacy)

| Method | Year + venue | Core idea (one line) | Backbone | Weights / repo | License | Compute | 12 GB local? | Known weaknesses |
|---|---|---|---|---|---|---|---|---|
| LDM-SR / SD×4 upscaler | 2023 | First "Stable Diffusion as an SR prior"; text-prompted upscaling | SD 1.5/2.1 UNet | HF `stabilityai/stable-diffusion-x4-upscaler` | OpenRAIL-M | ~10 steps, ~8–10 GB | Yes | Low fidelity, invented texture, weak on real degradations |
| **ResShift** | **NeurIPS 2023** | Residual shifting: shrink the diffusion trajectory between LR and HR to ~15 steps | Custom UNet | `zsyOAOA/ResShift` | [unverified] | 15 NFE, ~4–6 GB | Yes | Still 15 NFE; limited realism vs SD-prior methods |
| **StableSR** | **CVPR 2024** | Time-aware encoder + CFW; keeps SD prior, avoids retraining the UNet | SD 2.1 | HF `Iceclear/StableSR`; webui `sd-webui-stablesr` | `license: other` | 50–100 steps, 10–14 GB | Marginal (tiling/offload) | Slow; color/brightness shift; prompt-sensitive |
| **SeeSR** | **CVPR 2024** | Semantics-aware: a tagger produces object-level prompts; degradation-aware prompt | SD 2 | `cswry/SeeSR` | Apache-2.0 [unverified] | ~50 steps, ~10 GB | Marginal | Extra tagger cost; hallucinated semantics from wrong tags |
| **SUPIR** | **CVPR 2024** | SDXL + a ~4 B multimodal LLM for restoration-aware prompts + restoration-guided sampler | SDXL | `Fanghua-Yu/SUPIR`; ComfyUI-SUPIR (kijai) | SDXL OpenRAIL++ + LLaVA; research-oriented | 24 GB+ typical | **No — OOM on 12 GB documented** | Very slow, huge VRAM, strong hallucination, color drift |
| **PASD** | **ECCV 2024** | Pixel-aware: pixel-level + semantic guidance to preserve structure in SD | SD | `yangxy/PASD` | [unverified] | ~20–50 steps | Marginal | Needs face/segmentation helpers; slower than one-step |
| **CoSeR** | **CVPR 2024** | "Cognitive" SR: LR → generated caption + concept embedding to steer SD | SD 2 | `VINHYU/CoSeR` | [unverified] | ~50 steps | Marginal | Caption errors → wrong content; slow |
| **DiffBIR** | **ECCV 2024** (arXiv 2308.15070) | Two-stage: SwinIR restoration stage, then SD generative refinement | SD 2.1 | `XPixelGroup/DiffBIR`; HF space `MultiMatrix/DiffBIR` | Repo Apache-2.0-class; SD base OpenRAIL [partly unverified] | 50 steps, ~12–16 GB | Marginal | Slow; stage-2 invents detail; stage coupling tuning |
| **DreamClear** | **NeurIPS 2024** | High-capacity DiT restoration + privacy-safe dataset curation (DIV2K-P) | DiT | `shallowdream204/DreamClear` | [unverified] | Multi-step DiT, ~14 GB+ | Marginal | DiT inference cost; dataset-centric gains don't transfer freely |

**Newer than the list:** `SRSR` (NeurIPS 2025, spatially re-focused text conditioning for *semantic* accuracy — attacks the "right-looking, wrong-content" failure), `VARSR` (arXiv 2509.23774, texture VQ + reconstruction-aware prediction), `4KAgent` (NeurIPS 2025, *agentic* any-image→4K pipeline rather than a single model), `Dual Prompting Image Restoration with Diffusion Transformers` (CVPR 2025).

---

## 2. One-step / few-step and distilled SR (the current mainstream)

| Method | Year + venue | Core idea | Backbone | Weights / repo | License | Compute | 12 GB local? | Known weaknesses |
|---|---|---|---|---|---|---|---|---|
| **SinSR** | **CVPR 2024** | First single-step distillation of a multi-step SR diffusion (ResShift) | ResShift UNet | `wangyinz/SinSR` **[repo unverified]** | [unverified] | 1 NFE, ~4 GB | Yes, easily | Lowest ceiling; little gain on heavy degradations |
| **OSEDiff** | **NeurIPS 2024** | One-step *variational score distillation* on SD + LoRA; no reward model, no extra discriminator net | SD 2.1 + LoRA | `cswry/OSEDiff` | Apache-2.0 [unverified] | **1.3 s @1024²**, ~8–10 GB fp16 | **Yes** | Slight hallucination; low PSNR by design; LoRA must match domain |
| **S3Diff** | 2025 (arXiv 2409.17058) | Degradation-guided one-step: a degradation-aware adapter steers a LoRA-tuned SD | SD 2.1 | `ArcticHare105/S3Diff` | [unverified] | **1.4 s @1024²**, ~9–11 GB | **Yes** | Adapter adds a small overhead; still SD-VAE color drift |
| **TSD-SR** | **CVPR 2025** | Target Score Distillation: distil a multi-step teacher's *score* into one step | SD | [unverified] | [unverified] | 1 NFE | Yes | Teacher-student gap; limited diversity |
| **PiSA-SR** (Dual-LoRA) | **CVPR 2025** (Sun et al.) | Dual LoRA: pixel-level LoRA + semantic-level LoRA → **adjustable** fidelity/realism | SD | `csslc/PiSA-SR` **[unverified]** | [unverified] | 1 NFE + 2 LoRA | Yes | Two-LoRA inference overhead; tuning is per-image |
| **AdcSR** | **CVPR 2025** (arXiv 2411.13383) | Adversarial *diffusion compression*: prune the multi-step SR net, then distil into a compact one-step model | Compressed SD-derived | [unverified] | [unverified] | 1 NFE, smaller than OSEDiff | Yes, comfortable | Compression ceiling; robustness to unseen degradations |
| **AddSR** | **Pattern Recognition 175 (2025)**, doi 10.1016/j.patcog.2025.113012 | Adversarial diffusion distillation for blind SR acceleration | SD-derived | [unverified] | [unverified] | 1–few NFE | Yes | Similar ceiling to AdcSR |
| **RSD** | 2025 (arXiv 2503.13358) | One-step **residual shifting** distillation with a *hyperparameter-free* distillation loss | ResShift-derived | [unverified] | [unverified] | 1 NFE, small | Yes, comfortable | Residual-shift realism ceiling; less "generative" look |
| **OFTSR** | **ICLR 2026** (arXiv 2412.09465) | **One-step flow** matching SR with *tunable* fidelity–realism trade-off at inference | Flow-matching net | [unverified] | [unverified] | 1 NFE (flow) | Likely yes | Newest; ecosystem/tooling immature |
| **InvSR** | **CVPR 2025** | Diffusion **inversion**: estimate a good starting noise from the LR, then run *arbitrary* steps | SD | [unverified] | [unverified] | 1–N steps, user-chosen | Yes at 1–4 steps | More steps ≠ monotonically better; inversion cost |
| Time-Aware / Realism-Control / Controllable one-step (arXiv 2508.16557, 2509.10122, 2512.14061) | 2025 | Three minor 2025 variants: timestep-aware conditioning; an explicit realism knob; controllable fidelity–reality | SD-derived | [unverified] | [unverified] | 1 NFE | Yes | Incremental; FR+NR metric mix has comparability caveats |
| OSDFace (sibling) | 2024 (arXiv 2411.17163) | Same one-step recipe applied to **face** restoration | SD | [unverified] | [unverified] | 1 NFE | Yes | Domain-specific |

**"UFO (one-step)"**: the only match found in this space is **UFOGen** (CVPR 2024, *You Forward Once* — one-step text-to-image/video diffusion via adversarial training), which is **not an SR model**. No SR-specific "UFO" surfaced in 2024–2026 indexes. **[unverified / likely naming mismatch]**

---

## 3. DiT, flow-matching and 2026 frontier SR

| Method | Year + venue | Core idea | Backbone | Repo/weights | License | 12 GB local? | Notes |
|---|---|---|---|---|---|---|---|
| **DiT4SR** | **ICCV 2025** | First strong "tame a Diffusion Transformer for real-world SR"; DiT + SR-specific adaptation | DiT | [unverified] | [unverified] | Marginal | Includes a user study — rare in this literature |
| **FlowSR** | **ICCV 2025** | Fast SR via **consistency rectified flow** | Flow matching | HF `chunjie-spring/FlowSR` | [unverified] | Likely yes | One of the first flow-matching SR releases |
| **NVIDIA PiD** (Pixel Diffusion) | **NeurIPS 2026** | Pixel-space diffusion **decoder replacing the SDXL VAE**, 4-step; targeted at 2K→4K | Pixel DiT | official `nv-tlabs/PiD`; HF `Glebka/PiD-res2kto4k-ONNX`, `guillekenzo/PiD-bucket` | [unverified] | Yes (has ONNX/TensorRT export) | Directly relevant to VAE-artifact pain — see §6 |
| **StrSR** | **ECCV 2026** (Oct 2026) | **Spectral + trajectory regularization** for DiT-based SR | DiT | official `jkwang28/StrSR` | [unverified] | Likely yes | Targets DiT spectral bias / trajectory drift |
| **Adversarial Score Distillation for Stable One-Step Diffusion in Real-World SR** | **ECCV 2026** (Oct 2026) | Adversarial score distillation to *stabilise* one-step training | SD-derived | [unverified] | [unverified] | Yes | Attacks one-step training instability directly |
| **Fill2SR** | 2026 (arXiv 2609.33582, Sep 2026) | **Repurpose inpainting diffusion transformers** for real-world SR | Inpainting DiT | [unverified] | [unverified] | Likely yes | Free prior from an inpainting DiT; new as of Sep 2026 |
| **TOLA** | 2026 (arXiv 2609.29240, Sep 2026) | Text-aware **one-step latent adaptation** for *text-image* SR | SD-derived | [unverified] | [unverified] | Yes | Domain-specific (text/OCR legibility); reports a text-SR test set |
| **Super-Resolution in the Right Latent Space** | 2026 (arXiv 2610.04781, Oct 2026) | SR in a **frozen vision-foundation** latent substrate instead of the SD VAE latent | Frozen VF encoder | [unverified] | [unverified] | Likely yes | Attacks VAE-latent mismatch — pairs with the PiD/VAE thread |
| **CLIMB-flow** | 2026 (arXiv 2609.33834, Sep 2026) | Coupled linear-inverse **posterior sampling** via multiscale flow | Flow matching | [unverified] | CC BY 4.0 | Marginal | Inverse-problem framing; SR is one instance |
| **DLC** | 2026 (arXiv 2610.11433, Oct 2026) | **Metric-guided dynamic loss controller** for multi-objective restoration training | n/a (training) | [unverified] | [unverified] | n/a | Directly addresses the metric-gaming problem in §6 |
| **OracleZoom** | 2026 | SR built on **Stable Diffusion 3 Medium** (`stabilityai/stable-diffusion-3-medium-diffusers`) | **SD3 / MMDiT** | HF `dipta007/OracleZoom` | Inherits SD3 Community Licence [unverified] | Likely yes (~SD3-M class) | The clearest SD3-based SR with open weights found |
| **FlowUpscaler** | 2026 | Flow-based upscaler | Flow matching | HF `TensorForger/FlowUpscaler` | [unverified] | Likely yes | Very early; unverified quality claims |
| **TEASR** | **ECCV 2026** (arXiv 2606.16188) | Training-efficient **any-step** DiT for real-world SR | DiT | [unverified] | [unverified] | Likely yes | "Any-step" = one model, many NFE budgets |
| **One-Step Diffusion Transformer for Controllable Real-World SR** | **CVPR 2026** (arXiv 2511.17138) | Controllability moved into a one-step **DiT** | DiT | [unverified] | [unverified] | Likely yes | Flagship 2026 CVPR entry |
| **IFCSR** | **CVPR 2026** | *Inference-free* fidelity–realism control for one-step diffusion SR | SD-derived | [unverified] | [unverified] | Yes | Control baked in at training, not sampling |
| **DNF-SR** | **CVPR 2026** | Dual-input + negative-aware feature fine-tuning for real-world SR | SD-derived | [unverified] | [unverified] | Yes | Conventionally fine-tuned, not distilled |
| **AlloSR²** / **MeanSR** | 2026 (arXiv 2604.19238, 2608.09405) | Rectify one-step SR "to stay real" via **allomorphic generative flows**; restoration-**trajectory** learning for one-step perceptual SR | Flow matching | [unverified] | [unverified] | Likely yes | Both attack one-step oversaturation/artificiality |
| **Bridging Restoration & Generation Manifolds** | 2026 (arXiv 2604.24136) | Bridge the restoration and generation manifolds in one-step SD SR | SD-derived | [unverified] | [unverified] | Yes | Reports **DIV2K-val PSNR ≈ 22.2–24.9 dB** across one-step variants |
| **PixelIR** | 2026 (arXiv 2608.30782) | Fidelity–perception *decoupling* via pixel-space image-residual flow matching | Pixel flow | [unverified] | **CC BY-NC-SA 4.0** | Yes (pixel-space = no VAE) | Non-commercial licence |
| **DriftSR** | 2026 (arXiv 2610.04819, **Oct 2026 — current frontier**) | One-step real-world SR via **distribution drifting** | [unverified] | [unverified] | [unverified] | Likely yes | Brand new; no replication yet |
| Continuous Degradation Modeling via Latent Flow Matching | **AAAI 2026** (v40i7) | Learn a *continuous* degradation space with flow matching for SR training | Flow matching | [unverified] | [unverified] | n/a (training method) | Degradation modelling, not inference |
| Bridging the Distribution Gap (ICLR 2026) / Selective Diffusion Distillation (AAAI 2026) | **ICLR + AAAI 2026** | Close the train/test prior-distribution gap; selectively distil for ×8-class upscaling | — | [unverified] | [unverified] | n/a | — |
| **SRSR** | **NeurIPS 2025** (arXiv 2510.22534) | Spatially re-focused text conditioning for **semantic accuracy** | SD-derived | [unverified] | [unverified] | Yes | Faithfulness-oriented |
| **Coloring the Noise** | **ICML 2026** | Adversarial **Sobolev alignment** for faithful SR (penalises distributional drift) | — | [unverified] | [unverified] | — | Faithfulness theory |
| **4KAgent** | **NeurIPS 2025** (arXiv 2507.07105) | Agentic pipeline: degrade-analysis → expert routing → tiled restoration → 4K | System | [unverified] | [unverified] | Partial (per-tile) | Best evidence that 4K is an orchestration problem |

---

## 4. Video / temporal SR (and what it gives image SR)

| Method | Year + venue | Core idea | Backbone | Repo/weights | License | 12 GB local? | Weaknesses |
|---|---|---|---|---|---|---|---|
| **Upscale-A-Video** | **CVPR 2024** | Temporal-aware + flow-guided latent propagation for consistent VSR | SD-based | `sczhou/Upscale-A-Video` | [unverified] | Marginal | Slow; long-video memory; flicker at scene cuts |
| **MGLD-VSR** | **ECCV 2024** (arXiv 2312.00853) | Motion-guided latent diffusion for temporally consistent real-world VSR | Latent diffusion | [unverified] | [unverified] | Marginal | Motion-estimation cost; drift on fast motion |
| **VEnhancer** | 2024 | Space-time conditioning for video enhancement with a generative prior | SD + ControlNet | [unverified] | [unverified] | Marginal | Generative invention in low-detail regions |
| **SeedVR** | 2024 | Video restoration **Diffusion Transformer** with variable-length attention | DiT | [unverified] | [unverified] | No (large) | — |
| **STAR** | **ICCV 2025** *(not 2024)* | Spatial-temporal augmentation via text-to-video models for real-world VSR | T2V backbone | [unverified] | [unverified] | No | Heavy; T2V prior can invent motion |
| **SeedVR2** | 2025 (arXiv 2506.05301) | **One-step** video restoration via **diffusion adversarial post-training** | Video DiT, 3B / 7B (+community 1.4B) | HF `ByteDance-Seed/SeedVR2-3B`, `-7B`; ComfyUI `numz/SeedVR2_comfyUI` | **Apache-2.0** (confirmed on HF) | **3B plausible on 12 GB**; 7B needs offload | **Seam artifacts reported** ("seams too much in comfyui", HF discussion on SeedVR2-1.4B) |
| **FlashVSR** | 2025 | Diffusion-based **streaming** VSR, optimised for consumer VRAM | DiT/streaming | `ai-anchorite/FlashVSR_plus`; ComfyUI_FlashVSR | [unverified] | **Advertised: 12 GB VRAM / 32 GB RAM** | Young project; seam/flicker tuning |
| **DUO-VSR** | **CVPR 2026** | Dual-stream distillation for **one-step** VSR | — | [unverified] | [unverified] | Likely yes | — |
| **ReCaVSR** | 2026 (arXiv 2609.37831, Sep 2026) | One-step streaming VSR with **recycled latents + learned cache routing** | Streaming DiT | [unverified] | [unverified] | Likely yes | ~1,000-frame sequences (LongVSR60); new |

**Transfer to image SR:** the three ideas that crossed over are (a) **adversarial post-training to one step** (SeedVR2 → the AdcSR/AddSR/OSEDiff lineage), (b) **block/streaming attention + caching** to bound VRAM at high resolution, and (c) **temporal-consistency-style regularisation** repurposed as tile-seam consistency. Temporal metrics in video (tOF/tLP, warping error) have **no standard image-SR analogue**; tiling seams are still judged by eye or by tile-boundary-gradient heuristics **[unverified]**.

---

## 5. Degradation modelling, GAN-vs-diffusion, faithfulness, tiling/VAE

**Degradation modelling.** Real-ESRGAN's **high-order degradation pipeline (2021)** — blur → resize → noise → JPEG, applied twice, with sinc filters — remains the default synthetic training recipe a *half-decade* later; BSRGAN (2021) and LSDIR (CVPRW 2023, ~85 K images) are the other fixtures. 2024–2026 attacks:
- **Learned/continuous degradation** — "Continuous Degradation Modeling via Latent Flow Matching" (AAAI 2026) and "Unsupervised Real-World SR via Rectified Flow Degradation Modelling" (arXiv 2508.07214) treat the degradation *distribution* as the learnable object; **VLM/LLM degradation priors** (Sci. Rep. 2025) and **pixel-adaptive noise/degradation awareness** (Neurocomputing 2026) are the other 2025–2026 routes.
- **Dataset-side** — DreamClear's DIV2K-P (NeurIPS 2024) curates the *training* set rather than the degradation. Fixtures: DIV2K, Flickr2K, LSDIR (CVPRW 2023, ~85 K images), DIV8K, RealSR, DRealSR, RealLQ250. A named "Real-4K"/"UltraReal" dataset **could not be confirmed**, and NTIRE 2026's *Mobile Real-World SR* (arXiv 2604.17306) is the newest benchmark axis found.

**GAN vs diffusion (honest trade-off).**
| Axis | GAN (Real-ESRGAN, BSRGAN, SwinIR-GAN, HAT) | Diffusion / one-step generative |
|---|---|---|
| 1024² latency | ~0.05–0.5 s | ~1.3–1.5 s (one-step) to minutes (multi-step) |
| Perceptual metrics + faithfulness | Good; over-sharpened "GAN look", but structurally faithful | Best CLIPIQA/MUSIQ/MANIQA scores in most 2024–2026 tables; **hallucinated texture/semantics is the defining failure** |
| Robustness on real degradations | Degrades into smearing/plastic look | Recovers plausible texture |
| VRAM | ~2–4 GB at 4K with tiling | ~8–12 GB one-step; 16–24 GB+ multi-step |
| 4K+/batch | Practical | Still an orchestration problem (4KAgent, tiling) |

No single 2025–2026 paper found that settles this head-to-head **with a controlled protocol** — comparisons are typically per-paper tables **[unverified]**.

**Faithfulness / hallucination control.** Active line: Sobolev-style distributional alignment (**Coloring the Noise**, ICML 2026), semantic-accuracy conditioning (**SRSR**, NeurIPS 2025), inference-time realism dials (**PiSA-SR**, **OFTSR**, **IFCSR**, arXiv 2512.14061), and classical CFG tricks (lower CFG scale, **CFG rescale**, negative prompts). Community practice also uses VAE fp16→fp32 upcasting to stop NaN/black frames.

**Tiling / VAE artifacts at 4K+.** Causes: (1) SD/SDXL VAE decodes tiles independently → luminance/color mismatch at seams; (2) fp16 VAE overflow → NaN/black images; (3) overlapping-tile blending without latent-space continuation; (4) per-tile prompt/CFG variation. Mitigations found in the wild: latent-space tiling with feathered blending, VAE upcast to fp32, fixed global seed + identical conditioning per tile, CFG rescale, and **pixel-space decoders (NVIDIA PiD)** that remove the VAE entirely. There is also a classical literature ("A Progressive Meta-Algorithm for Large and Seamless SR Images", IEEE 2025; FSTD, 2023). **No peer-reviewed 2025–2026 study quantifying tile-seam artifacts in diffusion SR was found** — this is a genuine gap.

---

## 6. Benchmarks and metrics

**Challenge results.** NTIRE runs an **Efficient SR** track every year: 11th edition at **CVPR 2026** (Ren et al., CVPRW 2026), 10th at **CVPR 2025** (arXiv 2504.10686), plus a separate **×4 SR** track (CVPRW 2024 Chen et al.; CVPRW 2025 Chen et al.; official repo `zhengchen1999/NTIRE2025_ImageSR_x4`). **NTIRE 2026** also ran the **First Challenge on Mobile Real-World Image SR** (arXiv 2604.17306), plus 3D-content SR and bitstream-corrupted video restoration; PLKSR-Rep and "Speed-Efficient Restoration with Compact High-Frequency for On-Device SR" are two 2026 NTIRE entries. **Xiaomi's imaging team took three CVPR 2026 NTIRE awards, including the portrait-restoration track title** (Chinese press, May 2026). **AIM 2025** ran "Efficient Perceptual Image Super Resolution" (Longarela et al., ICCVW 2025), scored with **PI + CLIPIQA + MANIQA** across team entries; AIM 2024 covered Video SR *Quality Assessment* (arXiv 2410.04225) and AIM 2025 a Robust Offline Video SR challenge. **Per-team winning method names for NTIRE 2025/2026 could not be extracted** from search snippets — the reports exist but their internals were not reachable.

**Typical real-world SR benchmark sets** are DIV2K-val (synthetic, clean reference), RealSR and DRealSR (real LR/HR pairs), and LSDIR-derived splits; 2026 adds **mobile real-world SR** (NTIRE 2026) and small **text-image SR** sets (TOLA, arXiv 2609.29240). **Reference-based numbers are not comparable across papers** on RealSR/DRealSR: different papers use different HR references, alignment, and crop sets. Treat cross-paper PSNR/LPIPS deltas of <0.2 dB / <0.01 as noise.

**Numbers I could verify.** One-step diffusion SR sits at **DIV2K-val PSNR ≈ 22.2–24.9 dB** (arXiv 2604.24136 table). One-step latency: **OSEDiff 1.3 s, S3Diff 1.4 s @1024²** (CVPR 2026 Face2Scene comparison). Concrete per-method PSNR/LPIPS tables for RealSR/DRealSR exist (e.g. in the 4KAgent paper arXiv 2507.07105, the RSD paper arXiv 2503.13358, an IJCV survey table with PSNR/SSIM/DISTS/NIQE/MUSIQ/MANIQA, and the AAAI 2026 SR table) but **the numeric cells were not retrievable** through search snippets.

**Metrics the field trusts (with criticisms).**

| Metric | Type | Status | Criticism |
|---|---|---|---|
| PSNR / SSIM | FR | Legacy baseline | Anti-correlated with perceptual quality for generative SR; kept mainly for comparability |
| **LPIPS** | FR | Standard but ageing | Converges/saturates on generative output; biased to texture statistics of its training set |
| **DISTS** | FR | Preferred over LPIPS in several 2025–2026 tables | Still a learned proxy; sensitive to the reference choice |
| FID | FR (distributional) | Common in SD-SR papers | Unstable at small sample counts; not an image-level metric |
| **NIQE** | NR | Retained for legacy comparability | **Effectively broken on generative output** — rewards smooth/unnatural images |
| **CLIPIQA / CLIPIQA+ / MUSIQ / MANIQA** | NR | Dominant in 2024–2026 tables; CLIPIQA+ and CLIPIQA+\_rn50\_512 are the current variants | Reward sharpness/"AI look"; gameable. A **CVPR 2026 supplement explicitly shows PI, NIQE, CLIP-IQA and MANIQA "can fail"** where subtle semantic structure matters |
| **TOPIQ / Q-Align / Q-Instruct** | NR (+LMM) | TOPIQ (IEEE TIP 2024) rising, semantically top-down; Q-Align/Q-Instruct growing for quality-description | Newer with fewer SR-specific validations; LMM bias, cost, reproducibility |
| Perceptual Index (PI) | NR | Used by AIM challenges | Composite of Ma/MANIQA+NIQE; inherited weaknesses |
| 2025–2026 replacements | — | No single accepted successor emerged | The **NTIRE 2026 PLKSR-Rep paper warns directly that optimising IQA objectives overfits the metric** |

---

## Practical recommendations: Windows + Python 3.10 + onnxruntime/ncnn local toolkit

1. **Split the toolkit in two.** (a) An **onnxruntime/ncnn fast path** for the 90 % case — Real-ESRGAN x4plus/anime, Real-CUGAN, SwinIR (`swinir-onnx` on HF), waifu2x (ncnn-Vulkan), plus ONNX/TensorRT exports of NVIDIA **PiD** for the decode stage. (b) A **separate optional GPU/PyTorch path** for the generative family, invoked only when the input is genuinely degraded.
2. **Do not try to ONNX-export diffusion SR.** OSEDiff/S3Diff/SUPIR-class graphs need CFG, schedulers, tiling, LoRA merging and dynamic control flow. There is **no maintained official ONNX pipeline** for any of them **[unverified but consistent across all searches]**. Route generative SR through ComfyUI or a pinned PyTorch venv instead.
3. **On a 12 GB RTX 4070 SUPER, the practical generative tier is: OSEDiff, S3Diff, PiSA-SR, InvSR (1–4 steps), RSD, AdcSR, plus SeedVR2-3B and FlashVSR for video.** All should run fp16 with tiling. **SUPIR is out** (documented OOM at 12 GB), and SeedVR2-7B needs offload. **Watch the new open-weight entries** — `dipta007/OracleZoom` (SD3-Medium backbone) and `TensorForger/FlowUpscaler` (flow matching) are the first credible SD3-class/flow SR checkpoints; benchmark them before trusting them **[unverified]**.
4. **CPU is only for the ONNX path.** Budget: Real-ESRGAN/SwinIR x4 on a modern desktop CPU is seconds-to-minutes per 1080p frame depending on model and thread count; a single 1024² diffusion UNet step in fp32 is far worse. Offer CPU = "GAN/CNN only".
5. **Ship a degradation classifier / router, not one model.** The strongest 2025–2026 evidence (4KAgent, NeurIPS 2025) is that a router + per-region experts beats a single monolithic model. A cheap heuristic (blur estimate, noise σ, JPEG blockiness) plus a face detector is enough to start.
6. **Expose the fidelity–realism dial as a first-class UI control**, not a hidden setting. PiSA-SR, OFTSR, IFCSR and arXiv 2512.14061 all converge on this; users care more about "don't invent" vs "make it pretty" than about PSNR.
7. **Tiling discipline for 4K+:** tile in **latent** space, decode with the VAE in **fp32**, feather overlaps ≥ 32 px, keep the seed and conditioning identical across tiles, and run a global colour-match pass (e.g. wavelet or histogram matching to the pre-upscale luminance) as a final stage. Report tile count to the user.
8. **Dependencies:** onnxruntime (DirectML EP for AMD/Intel, CUDA EP for NVIDIA), ncnn-Vulkan binaries for Real-ESRGAN/Real-CUGAN, Pillow/OpenCV, and `safetensors`. Avoid pulling `diffusers`/`torch` into the base environment — keep the generative tier optional and version-pinned.
9. **For batch/4K throughput, prefer GAN/CNN SR and reserve diffusion for hero images.** This is the single highest-leverage engineering decision in this report.
10. **Validate with ≥ 3 metrics spanning FR + NR + human spot-checks**, e.g. DISTS + CLIPIQA+ (or TOPIQ) + a manual pass. Do not tune on a single NR metric — that is exactly the failure mode NTIRE 2026 warns about.

---

## Sources

1. [One-Step Effective Diffusion Network for Real-World Image SR (OSEDiff) — NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/a8223b0ad64007423ffb308b0dd92298-Paper-Conference.pdf) · [poster listing](https://neurips.cc/virtual/2024/poster/95036)
2. [S3Diff: Degradation-Guided One-Step SR with Diffusion Priors, arXiv 2409.17058](https://ar5iv.labs.arxiv.org/html/2409.17058) · [repo](https://github.com/ArcticHare105/S3Diff) (2024/2025)
3. [TSD-SR: One-Step Diffusion with Target Score Distillation — CVPR 2025](https://www.openaccess.thecvf.com/content/CVPR2025/html/Dong_TSD-SR_One-Step_Diffusion_with_Target_Score_Distillation_for_Real-World_Image_CVPR_2025_paper.html) (2025)
4. [Pixel-level and Semantic-level Adjustable SR: A Dual-LoRA Approach (PiSA-SR) — CVPR 2025](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Sun_Pixel-level_and_Semantic-level_Adjustable_Super-resolution_A_Dual-LoRA_Approach_CVPR_2025_paper.pdf) (2025)
5. [Adversarial Diffusion Compression for Real-World SR (AdcSR), arXiv 2411.13383 — CVPR 2025](https://arxiv.org/pdf/2411.13383v1) · [paper note](https://github.com/zhaoyang97/Paper-Notes-en/blob/main/docs/CVPR2025/image_restoration/adversarial_diffusion_compression_for_real-world_image_super-resolution.md) (2025)
6. [AddSR: Accelerating diffusion-based blind SR with adversarial diffusion distillation — Pattern Recognition 175](https://www.sciencedirect.com/science/article/abs/pii/S0031320325016759) (2025)
7. [One-Step Residual Shifting Diffusion for SR via Distillation (RSD), arXiv 2503.13358](https://huggingface.co/papers/2503.13358) (2025)
8. [Arbitrary-steps Image Super-resolution via Diffusion Inversion (InvSR) — CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Yue_Arbitrary-steps_Image_Super-resolution_via_Diffusion_Inversion_CVPR_2025_paper.pdf) (2025)
9. [One-Step Flow for Image SR with Tunable Fidelity-Realism Trade-offs (OFTSR) — ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/93c359aee19c98479ec144ae221ac368-Abstract-Conference.html) · [arXiv 2412.09465](http://export.arxiv.org/pdf/2412.09465) (2024/2026)
10. [SinSR: Diffusion-Based Image Super-Resolution in a Single Step — CVPR 2024](https://www.openaccess.thecvf.com/content/CVPR2024/html/Wang_SinSR_Diffusion-Based_Image_Super-Resolution_in_a_Single_Step_CVPR_2024_paper.html) (2024)
11. [StableSR weights & model card, HF Iceclear/StableSR (license: other)](https://huggingface.co/Iceclear/StableSR/blob/main/README.md) (2024)
12. [DiffBIR: Toward Blind Image Restoration with Generative Diffusion Prior — ECCV 2024](https://dl.acm.org/doi/10.1007/978-3-031-73202-7_25) · [arXiv 2308.15070](https://ar5iv.labs.arxiv.org/html/2308.15070) (2023/2024)
13. [Efficient Diffusion Model for Image Restoration by Residual Shifting (ResShift), arXiv 2403.07319](https://ar5iv.labs.arxiv.org/html/2403.07319) · [NeurIPS 2023 proceedings](https://dl.acm.org/doi/abs/10.5555/3666122.3666705) (2023)
14. [SeeSR: Towards Semantics-Aware Real-World SR — CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_SeeSR_Towards_Semantics-Aware_Real-World_Image_Super-Resolution_CVPR_2024_paper.pdf) · [repo](https://github.com/cswry/SeeSR) (2024)
15. [Pixel-Aware Stable Diffusion for Realistic Image SR and Personalized Stylization (PASD) — ECCV 2024](https://dl.acm.org/doi/abs/10.1007/978-3-031-73247-8_5) (2024)
16. [Scaling Up to Excellence: Photo-Realistic Image Restoration (SUPIR) — CVPR 2024](https://github.com/fanghua-yu/supir) (2024)
17. [OutOfMemory with 12GB VRAM — kijai/ComfyUI-SUPIR issue #12](https://github.com/kijai/ComfyUI-SUPIR/issues/12) (2024)
18. [DreamClear — NeurIPS 2024 proceedings](https://papers.nips.cc/paper/2024/hash/6452474601429509f3035dc81c233226-Abstract-Conference.html) · [repo](https://github.com/shallowdream204/dreamclear) (2024)
19. [CoSeR: Bridging Image and Language for Cognitive SR — CVPR 2024](https://github.com/vinhyu/coser) (2024)
20. [DiT4SR: Taming Diffusion Transformer for Real-World SR — ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Duan_DiT4SR_Taming_Diffusion_Transformer_for_Real-World_Image_Super-Resolution_ICCV_2025_paper.pdf) (2025)
21. [Fast Image SR via Consistency Rectified Flow (FlowSR) — ICCV 2025](https://iccv.thecvf.com/virtual/2025/poster/1871) · [weights](https://huggingface.co/chunjie-spring/FlowSR) (2025)
22. [TEASR: Training-Efficient Any-Step Diffusion Transformer for Real-World SR, arXiv 2606.16188 — ECCV 2026](https://arxiv.org/html/2606.16188v1) (2026)
23. [One-Step Diffusion Transformer for Controllable Real-World SR — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Fang_One-Step_Diffusion_Transformer_for_Controllable_Real-World_Image_Super-Resolution_CVPR_2026_paper.html) (2026)
24. [IFCSR: Inference-Free Fidelity-Realism Control for One-Step Diffusion-based Real-World SR — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Back_IFCSR_Inference-Free_Fidelity-Realism_Control_for_One-Step_Diffusion-based_Real-World_Image_Super-Resolution_CVPR_2026_paper.html) (2026)
25. [DNF-SR: Dual-Input and Negative-Aware Feature Fine-Tuning for Real-World SR — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Han_DNF-SR_Dual-Input_and_Negative-Aware_Feature_Fine-Tuning_for_Real-World_Image_Super-Resolution_CVPR_2026_paper.html) (2026)
26. [AlloSR²: Rectifying One-Step SR to Stay Real via Allomorphic Generative Flows, arXiv 2604.19238](https://ui.adsabs.harvard.edu/abs/2026arXiv260419238W/abstract) (2026)
27. [Bridging Restoration and Generation Manifolds in One-Step Diffusion for Real-World SR, arXiv 2604.24136](https://arxiv.org/html/2604.24136v2) (2026)
28. [PixelIR: Fidelity–Perception Decoupling via Pixel-Space Image–Residual Flow Matching, arXiv 2608.30782, CC BY-NC-SA 4.0](https://browse-export.arxiv.org/pdf/2608.30782) (2026)
29. [MeanSR: Restoration Trajectory Learning for One-Step Perceptual SR, arXiv 2608.09405](https://arxiv-org.ezproxy.obspm.fr/html/2608.09405v1) (2026)
30. [DriftSR: One-Step Real-World Image SR via Distribution Drifting, arXiv 2610.04819 — date anomalous](https://arxiv.org/abs/2610.04819) (2026)
31. [Continuous Degradation Modeling via Latent Flow Matching for Real-World SR — AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37486) (2026)
32. [Bridging the Distribution Gap to Harness Pretrained Diffusion Priors for SR — ICLR 2026 paper note](https://raw.githubusercontent.com/zhaoyang97/Paper-Notes-en/refs/heads/main/docs/ICLR2026/image_generation/bridging_the_distribution_gap_to_harness_pretrained_diffusion_priors_for_super-r.md) (2026)
33. [SRSR: Enhancing Semantic Accuracy in Real-World SR with Spatially Re-Focused Text-Conditioning, arXiv 2510.22534 — NeurIPS 2025](https://neurips.cc/virtual/2025/loc/san-diego/poster/115714) (2025)
34. [Coloring the Noise: Adversarial Sobolev Alignment for Faithful Image Super Resolution — ICML 2026](https://icml.cc/virtual/2026/poster/66201) (2026)
35. [4KAgent: Agentic Any Image to 4K Super-Resolution, arXiv 2507.07105 — NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f0075fe4e59652cf43148dcfab8d3c93-Abstract-Conference.html) (2025)
36. [NVIDIA PiD: Fast and High-Resolution Latent Decoding with Pixel Diffusion](https://research.nvidia.com/labs/sil/projects/pid/) · [SDXL 4-step ONNX/TensorRT export](https://huggingface.co/Glebka/PiD-res2kto4k-ONNX) (2025/2026)
37. [Bridging the Perception — CVPR 2026 supplemental (PI, NIQE, CLIP-IQA, MANIQA failure cases)](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Su_Bridging_the_Perception_CVPR_2026_supplemental.pdf) (2026)
38. [PLKSR-Rep: Compact Large-Kernel CNN for Mobile Real-World SR — NTIRE CVPRW 2026 (IQA-overfitting warning)](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Deng_PLKSR-Rep_A_Compact_Large-Kernel_CNN_for_Mobile_Real-World_Image_Super-Resolution_CVPRW_2026_paper.pdf) (2026)
39. [TOPIQ: A Top-Down Approach From Semantics to Distortions for Image Quality Assessment — IEEE TIP](https://xplorestaging.ieee.org/document/10478301) (2024)
40. [The Eleventh NTIRE 2026 Efficient Super-Resolution Challenge Report — CVPRW 2026](https://www.openaccess.thecvf.com/content/CVPR2026W/NTIRE/html/Ren_The_Eleventh_NTIRE_2026_Efficient_Super-Resolution_Challenge_Report_CVPRW_2026_paper.html) (2026)
41. [The Tenth NTIRE 2025 Efficient Super-Resolution Challenge Report, arXiv 2504.10686 — CVPRW 2025](https://ar5iv.labs.arxiv.org/html/2504.10686) (2025)
42. [NTIRE 2025 Challenge on Image SR (×4) — CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Chen_NTIRE_2025_Challenge_on_Image_Super-Resolution_x4_Methods_and_Results_CVPRW_2025_paper.pdf) · [NTIRE 2024 ×4](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chen_NTIRE_2024_Challenge_on_Image_Super-Resolution_x4_Methods_and_Results_CVPRW_2024_paper.pdf) (2024/2025)
43. [Efficient Perceptual Image Super Resolution: AIM 2025 Study and Benchmark — ICCVW 2025](https://openaccess.thecvf.com/content/ICCV2025W/AIM/papers/Longarela_Efficient_Perceptual_Image_Super_Resolution_AIM_2025_Study_and_Benchmark_ICCVW_2025_paper.pdf) (2025)
44. [AIM 2024 Challenge on Video Super-Resolution Quality Assessment, arXiv 2410.04225](https://arxiv.org/html/2410.04225) (2024)
45. [AIM 2025 Challenge on Robust Offline Video Super-Resolution — ICCVW 2025](https://www.openaccess.thecvf.com/content/ICCV2025W/AIM/html/Karetin_AIM_2025_Challenge_on_Robust_Offline_Video_Super-Resolution_Dataset_Methods_and_Results_ICCVW_2025_paper.html) (2025)
46. [Xiaomi wins three CVPR 2026 NTIRE awards incl. portrait-restoration track — Chinese press](https://www.ithome.com/0/952/050.htm) (2026)
47. [SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training, arXiv 2506.05301](https://ar5iv.labs.arxiv.org/html/2506.05301) · [3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) · [7B](https://huggingface.co/ByteDance-Seed/SeedVR2-7B) — licence apache-2.0 (2025)
48. [SeedVR2-1.4B community port — "seams too much in comfyui" discussion](https://huggingface.co/lvladikov/SeedVR2-1.4B/discussions/1) (2026)
49. [Upscale-A-Video official repo (sczhou/Upscale-A-Video)](https://github.com/sczhou/Upscale-A-Video) (2024)
50. [Motion-Guided Latent Diffusion for Temporally Consistent Real-world VSR (MGLD-VSR), arXiv 2312.00853 — ECCV 2024](https://ar5iv.labs.arxiv.org/html/2312.00853) (2024)
51. [STAR: Spatial-Temporal Augmentation with Text-to-Video Models for Real-World VSR — ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Xie_STAR_Spatial-Temporal_Augmentation_with_Text-to-Video_Models_for_Real-World_Video_Super-Resolution_ICCV_2025_paper.pdf) (2025)
52. [DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution — CVPR 2026](https://www.openaccess.thecvf.com/content/CVPR2026/html/Lv_DUO-VSR_Dual-Stream_Distillation_for_One-Step_Video_Super-Resolution_CVPR_2026_paper.html) (2026)
53. [ReCaVSR: One-Step Streaming Diffusion VSR with Recycled Latents and Learned Cache Routing, arXiv 2609.37831](https://arxiv-org.ezproxy.obspm.fr/html/2609.37831v1) (2026)
54. [FlashVSR — Diffusion-Based Streaming Video Super-Resolution, runs on 12GB VRAM](https://github.com/ai-anchorite/flashvsr_plus) (2025)
55. [Accelerating Diffusion Models with One-Step Distillation for Image and Video SR — ACM MMSys 2026](https://dl.acm.org/doi/abs/10.1145/3793853.3798191) (2026)
56. [Unsupervised Real-World SR via Rectified Flow Degradation Modelling, arXiv 2508.07214](https://arxiv.org/html/2508.07214v1) (2025)
57. [LSDIR: A Large Scale Dataset for Image Restoration — CVPRW 2023](https://www.openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Li_LSDIR_A_Large_Scale_Dataset_for_Image_Restoration_CVPRW_2023_paper.html) (2023)
58. [Real-world SR with VLM-based degradation prior learning — Scientific Reports](https://link.springer.com/article/10.1038/s41598-025-14581-0/tables/2) (2025)
59. [Pixel-adaptive noise and degradation awareness for real-world image SR — Neurocomputing](https://www.sciencedirect.com/science/article/abs/pii/S0925231226023945) (2026)
60. [Model Conversion — xinntao/Real-ESRGAN (ONNX/ncnn export tooling)](https://deepwiki.com/xinntao/Real-ESRGAN/6.1-model-conversion) · [sr-vulkan waifu2x-ncnn bindings](https://github.com/tonquer/sr-vulkan) (2022–2025)
61. [swinir-onnx community export, HF Heliosoph/swinir-onnx](https://huggingface.co/Heliosoph/swinir-onnx) (2024)
62. [ONNX Runtime DirectML Execution Provider docs](https://mintlify.wiki/microsoft/onnxruntime/execution-providers/directml) · [chaiNNer node-based SR runner](https://chainner.app/) (2025)
63. [A Progressive Meta-Algorithm for Large and Seamless Super Resolution Images — IEEE](https://xplorestaging.ieee.org/document/10825847) (2025)
64. [Fast Seamless Tiled Diffusion (FSTD), arXiv 2308.16582](https://huggingface.co/buckets/huggingchat/papers-content/tree/2308/2308.16582.md) (2023)
65. [Awesome-Super-Resolution (ChaofWang) — paper/repo index](https://github.com/ChaofWang/Awesome-Super-Resolution) (2026)
66. [Face2Scene — CVPR 2026 (source of the OSEDiff 1.3 s / S3Diff 1.4 s @1024² comparison)](https://openaccess.thecvf.com/content/CVPR2026/papers/Kazerouni_Face2Scene_Using_Facial_Degradation_as_an_Oracle_for_Diffusion-Based_Scene_CVPR_2026_paper.pdf) (2026)
67. [Time-Aware One Step Diffusion Network for Real-World Image SR, arXiv 2508.16557](https://ar5iv.labs.arxiv.org/html/2508.16557) (2025)
68. [Realism Control One-step Diffusion for Real-World Image SR, arXiv 2509.10122](https://ar5iv.labs.arxiv.org/html/2509.10122) (2025)
69. [Bridging Fidelity-Reality with Controllable One-Step Diffusion for Image SR, arXiv 2512.14061](https://ar5iv.labs.arxiv.org/html/2512.14061) (2025)
70. [VARSR: Texture Vector-Quantization and Reconstruction Aware Prediction for Generative SR, arXiv 2509.23774](https://arxiv-org.ezproxy.obspm.fr/html/2509.23774v3) · [Dual Prompting Image Restoration with Diffusion Transformers — CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Kong_Dual_Prompting_Image_Restoration_with_Diffusion_Transformers_CVPR_2025_paper.pdf) (2025)
71. [Selective Diffusion Distillation for Real-World High-Scale Image SR — AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38351) (2026)
72. [UFOGen: You Forward Once — CVPR 2024 supplemental (the only "UFO" one-step match; not an SR model)](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Xu_UFOGen_You_Forward_CVPR_2024_supplemental.pdf) (2024)
73. **ECCV 2026 (Oct 2026):** [StrSR — Spectral and Trajectory Regularization for Diffusion Transformer SR (poster)](https://eccv.eventhosts.cc/virtual/2026/poster/5583) · [repo](https://github.com/jkwang28/StrSR) · [Adversarial Score Distillation for Stable One-Step Diffusion in Real-World SR](https://eccv.eventhosts.cc/virtual/2026/poster/4022) · [Fill2SR: Repurposing Inpainting Diffusion Transformers for Real-World SR](https://eccv2024.ecva.net/virtual/2026/poster/4460)
74. **Sept–Oct 2026 arXiv frontier:** [Super-Resolution in the Right Latent Space: A Frozen Vision-Foundation Substrate, arXiv 2610.04781](https://scirate.com/arxiv/2610.04781) · [DriftSR, arXiv 2610.04819](https://scirate.com/arxiv/2610.04819) · [DLC: Dynamic Loss Controller for Multi-Objective Image Restoration, arXiv 2610.11433](https://papers.cool/arxiv/2610.11433) · [Fill2SR, arXiv 2609.33582](https://arxiv-org.ezproxy.obspm.fr/abs/2609.33582) · [TOLA: Text-aware One-Step Latent Adaptation for Text Image SR, arXiv 2609.29240](https://arxiv-org.ezproxy.obspm.fr/html/2609.29240v1) · [CLIMB-flow, arXiv 2609.33834 (CC BY 4.0)](https://arxiv-org.ezproxy.obspm.fr/html/2609.33834v1)
75. **Open-weight SD3/flow SR:** [OracleZoom (base: stabilityai/stable-diffusion-3-medium-diffusers)](https://huggingface.co/dipta007/OracleZoom) · [FlowUpscaler](https://huggingface.co/TensorForger/FlowUpscaler) (2026) · [NVIDIA PiD — NeurIPS 2026, official repo nv-tlabs/PiD](https://github.com/nv-tlabs/pid)
76. [The First Challenge on Mobile Real-World Image SR at NTIRE 2026, arXiv 2604.17306](https://arxiv.org/pdf/2604.17306) · [Speed-Efficient Restoration with Compact High-Frequency for On-Device SR — CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Im_Speed-Efficient_Restoration_with_Compact_High-Frequency_for_On-Device_Super-Resolution_CVPRW_2026_paper.pdf) (2026)
