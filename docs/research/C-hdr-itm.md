# Single-Image / SDR→HDR Conversion (Inverse Tone Mapping): State of the Art, 2024–2026

*Compiled 2026-10-11 (rev. 2). Scope: single-image + video SDR→HDR (ITM), HDR reconstruction, semantic/generative HDR, content-specific failure modes, evaluation, and local practicality on a 12 GB RTX 4070 SUPER.*
*Rev. 2 adds the mid/late-2026 frontier: CVPR 2026, SIGGRAPH 2026, ECCV-cycle and autumn-2026 arXiv work, plus the local-tooling landscape (§7).*

> ### ⚠️ Evidence-grade notice (read first)
> This report was produced in a **network-restricted sandbox**. `web_fetch` was blocked for *every* host
> (arxiv.org, github.com, openaccess.thecvf.com, huggingface.co, ijcai.org, sciencedirect.com, catalyzex.com
> — all resolved to non-public IPs) and direct HTTPS from the shell was refused. **All evidence therefore comes from
> `web_search` result titles, URLs, and indexed page snippets — not from full-text reads.**
> Consequences you must respect when citing this document:
> - Method names, venues and years are **verified** (indexed title/URL of the primary source). Exact hyperparameters,
>   licenses, VRAM figures and numeric results are **mostly NOT verified** — tagged `[UNVERIFIED]`, `[SNIPPET]`
>   (seen in an indexed snippet), or `[INFERRED]`.
> - `[NOT FOUND]` means I could not confirm the work exists at all — treat as a lead, not a fact.
> - Repo licenses/weights were never opened. **Verify every license before redistribution.**

---

## TL;DR

- **The non-generative deep-ITM line has plateaued.** HDRTVNet (ICCV 2021) → HDRTVNet++/TMM 2023–24 → RealRep (AAAI 2026) are all variations on *global mapping + local detail refinement + reformulated supervision*; gains are now incremental PSNR/PU-PSNR moves on HDRTV1K-family benchmarks `[SNIPPET]`. [3][4][12][13]
- **2025–2026 belongs to generative/prior-based ITM.** Verified representatives: **LEDiff** (CVPR 2025, latent *exposure* diffusion over LDR stacks), **HDR Image Generation via Gain Map Decomposed Diffusion** (ICCV 2025), **Boosting ITM via Diffusion Regularization** (ICCVW 2025 / AIM 2025), **LumaFlux** (arXiv 2026, physically-guided DiT + camera-response prior), **LumaGuide** (arXiv 2026, *training-free* distribution shaping), **MLLM-ITM** (IJCAI 2026). [5][6][7][8][15][52]
- **The winning 2026 architecture is a hybrid, not an end-to-end generator.** The generative model proposes *where* highlights/emitters may exist (or a coarse gain field); a physically-motivated, monotone luminance mapping does the actual nit assignment. This is the "semantic HDR" idea and it is what the newest papers cluster around. [7][8][10][54]
- **Gain-map parameterization is the practical bridge, and in 2026 it became the frontier.** Gain-Map-Decomposed Diffusion (ICCV 2025) + the ICC `adaptiveGainCurveType`/ADGC spec + the UltraHDR ecosystem let you ship *SDR base + gain map* instead of a new PQ master — safer for content with no true HDR information. CVPR 2026 then turned gain maps into an *efficiency* story: **FastGaMer** (Fast GainMap, practical ITM) [58] and **GMODiff** (one-step gain-map refinement with diffusion priors) [59]. This is the single most important shift since the 2025 diffusion wave.
- **Output convention is still split.** Legacy deep-ITM outputs a **linear/µ-law HDR *image*** meant for benchmarking (often 1000–4000 nit normalized float). Modern generative methods increasingly emit **display-referred HDR10 PQ** or **gain maps**. There is **no single community-wide output contract**, and this is a real interoperability problem. `[INFERRED from method descriptions]`
- **SDR paper-white → HDR reference white: 203 nits (75% PQ signal) is the ITU-R BT.2408 answer, and BT.2408-9 (03/2026) is current.** The alternatives are 100 nits (where "SDR = diffuse white, boost only above it" matters) — `Lw=100 cd/m² for downstream use` appears verbatim in the BT.2408 text — and 200 nit, a display-side/consumer convention, not the broadcast reference. [19][20][21]
- **Flat cel-shaded / UI / text content is the dominant failure class.** The signal is *clipped and structureless*: a pure-white fill has no gradient, no texture, no specular falloff, so any ITM must **invent** information. Mitigations converge on: highlight-support masks, morphological open/close on a binary eligibility map, guided filtering to lock original edges, and **diffuse-white / flat-region gating** that deliberately caps gain at 1.0. That gating is **literally in the patent literature** — a granted filing computes a per-region "positive diffuse white" luminance (`YposDW`) from the SDR input and drives the ITM curve from it, alongside morphological open/close filtering to build the mask. [23][24][27]
- **Video in 2026 grew a text/MLLM branch and a measurement branch.** Flicker is still not solved by per-frame ITM; current work adds global colour mapping + multi-frame interaction, spatial-temporal fusion and dedicated deflickering, and shot-change handling remains a *pipeline* concern (scene-cut detect → reset temporal state) rather than a learned one. New in 2026: **text-driven ITM for compressed SDR video via multi-modal description** (Pattern Recognition) [66], **HDR Video Generation via Latent Alignment with Logarithmic Encoding** [70], and — crucially — **Inverse-Tone-Mapped HDR Video Quality Assessment for Broadcast Television**, a dataset plus an **SDR-referenced** quality method built specifically around *ITM'd* video rather than natively captured HDR [65]. [16][17][18]
- **Evaluation moved to perceptually uniform encodings, then past them.** PU21 (PU-PSNR / PU-SSIM) is the de-facto benchmark metric, HDR-VDP-3 the reference perceptual model, δE-ITP in ICtCp the colour metric, HDR-VQM the video one, plus a 2026 HDR noise metric at HVEI; **PU21 is in tooling** (VMAFx metric #923). In 2026 the field also started correcting the metric *premise*: *Adapting Quality Metrics to Tone Mapping* (SIGGRAPH 2026) [68] and *Perceptual Assessment and Optimization of HDR Image Rendering* (CVPR 2024) [71] both address the fact that standard metrics assume display-encoded input and mis-score tone-mapped/HDR pairs. Reporting PU-PSNR alone in 2026 is behind the frontier. New resources: **xDR** (natively graded cinematic HDR+SDR pairs) and the **AIM 2025 ITM challenge report** [1][2][14][28][29][30][31][32][33].
- **Adjacent 2026 frontiers worth a line in a report:** *ExpandDiff* — dynamic-range-expanding diffusion for single-image HDR (Oct 2026 arXiv) [57]; LDR→**HDRI panorama** promotion with *rendering-space* losses [64]; **constrained spectral uplifting for HDR environment maps** (CGF/EGSR) [69]; **CVPR 2026 residual diffusion bridge** restoration models, directly reusable for ITM [63]; and non-photographic applications such as **CBCT medical ITM** [67].
- **Local practicality on a 12 GB RTX 4070 SUPER:** classical/deep ITM nets, **efficient gain-map models (FastGaMer, GMODiff, Gain-MLP)** and gain-map MLPs are trivially fine (seconds, <2 GB). SDXL/FLUX-scale diffusion ITM is the pinch point — 12 GB forces fp8/GGUF quantized UNet + tiled VAE + ≤768–1024 px working resolution `[INFERRED]`. The **hybrid route (small model or quantized diffusion proposes a highlight mask; you do the luminance math in numpy/OCIO)** is the best value per GB. See [Practical recommendations](#practical-recommendations-for-a-local-windows-pipeline).
- **Bottom line for a technical report:** cite the plateau (2021–2024 deep ITM) as settled, the generative wave (2025–2026) as active but *not reproducible locally without quantization*, the **efficient-gain-map turn (CVPR 2026: FastGaMer, GMODiff)** as the current frontier, and the hybrid semantic-gating design as the only approach that is simultaneously recent, principled, and shippable on consumer hardware. `[INFERRED]`

---

## 1. Conventions: how SDR white becomes HDR white

This section answers "how the field converts SDR paper-white to HDR" before any method table, because it determines whether a method's numbers are comparable.

| Convention | Value | Where used | Notes |
|---|---|---|---|
| **ITU-R BT.2408 reference white** | **203 cd/m² = 75% PQ signal** | Broadcast/operational HDR production | Current edition **BT.2408-9 (03/2026)**; used by ITU-R BT.2446 conversion methods. [19][20][21] |
| BT.2408 low-white variant | `Lw = 100 cd/m²` "for downstream use" appears verbatim in BT.2408 | Operational guidance for dark-environment / SDR-adjacent grades | `[SNIPPET]` from the BT.2408-8/-9 PDF text. [20] |
| **"SDR diffuse white = 100 nits"** | 100 cd/m² | Academic ITM papers that model SDR as *scene/display* diffuse white and only expand above it | Widely assumed in deep-ITM training targets; keeps mid-tones untouched. `[INFERRED]` |
| Consumer/display "200 nit" | 200 cd/m² | Display-side SDR-in-HDR container mapping, some consumer pipelines | Close to 203 but **not** the standard value — do not label an output "BT.2408 compliant" if you used 200. |
| Peak/graphics white | 1000 cd/m² typical; 4000–10000 possible | HDR10 mastering peak | Graphics/UI (BT.709/sRGB/AdobeRGB) content "may need re-mapping" per the Ultra HD Forum Blue Book. [27] |
| MaxCLL/MaxFALL | content metadata | HDR10 signalling | Must be computed from your *actual* output, not assumed. |

**Practical reading:** treat SDR 1.0 as **203 nits** for photographic/cinematic content (ITU-consistent), and consider **100 nits** when the image is graphics-like (UI, text, flat fills) so you are not forced to brighten things that were never meant to be bright. `[INFERRED — my recommendation, not a published rule]`

**Scene-referred linear vs display-referred PQ.** *Scene-referred linear/float* is what training and benchmarking use — linear HDR floats (or µ-law intermediates) with a normalized peak (1.0 ↔ 1000/4000 nits); good for losses, bad as a delivery format. *Display-referred PQ (ST.2084)* is required for HDR10 delivery and is what you want on disk. A *gain map* (SDR base + auxiliary gain channel) is the most robust delivery for *generated* HDR where you cannot guarantee the base was unclipped — ICC now standardizes exactly this via the **Adaptive Gain Curve Tag (ADGC / `adaptiveGainCurveType`)** in ICC.1 [25], with tooling in `pyultrahdr` [26]. *HLG* is scene-referred and backward-compatible but rarely the target of academic ITM. `[INFERRED]` — no single surveyed paper states this taxonomy; it is assembled from the output formats the method sources name.

**How the community judges a "correct" HDR grade,** ranked by how much trust to place in each:
1. **Full-reference perceptual metrics against natively graded HDR** — the only defensible objective proxy: PU-PSNR / PU-SSIM on perceptual encodings, plus HDR-VDP-3. [28][29][30]
2. **Natively graded paired datasets** — HDRTV1K-family and now **xDR** (cinematic, natively graded HDR *and* SDR, explicitly built to evaluate ITM). [1][14]
3. **Challenge rankings** — the AIM 2025 ITM report gives the current empirical ordering. [1][2]
4. **Subjective viewing** — under-used and the source of the field's biggest known bias (see §5). The classic subjective reference is the 2009 "reverse tone mapping through varying exposure conditions" study, which is **old (SIGGRAPH Asia 2009)** — background, not current evidence. [34]

---

## 2. Classic & deep ITM (non-generative)

All rows: **input** 8-bit SDR, **output** linear/µ-law HDR float unless noted. **No license in this section is verified** — I could not open a single repo, so every "unverified" means "check `LICENSE` yourself". `[UNVERIFIED]`

**Pre-deep classics (all *older* — background, not current SOTA).** **Global reverse tone mapping** (linear scaling / Reinhard-style inverse) applies one fixed inverse curve: trivially cheap (✅ any GPU), but clips harder, and a global curve cannot know *which* whites are emitters → **the canonical "everything glows."** [34] **Banterle et al. reverse tone mapping** (2006–2011) expands, *inverts the camera response*, then adds synthetic glare: ✅ negligible cost, but hallucinated glare/light-shafts, wrong on graphics, per-image parameters. [35]

| Method | Year + venue | Core idea | Repo / weights | Compute | 12 GB? | Failure modes |
|---|---|---|---|---|---|---|
| **HDRCNN / Endo et al.** | 2017 SIGGRAPH Asia (*older*) | Two-stage: CNN classifies clipped regions, a second branch reconstructs HDR detail | community reimplementations only `[NOT FOUND as official]` | small CNN, <2 GB | ✅ | Blurry hallucinated texture in clipped regions; no semantics for *what* is blown out. |
| **ExpandNet** | 2017 SIGGRAPH Asia (*older*) | Fully-convolutional encoder–decoder **with global feature extraction**; first end-to-end single-image HDR | `github.com/fxtmtrsine/hdr-expandnet` (community code) [36] | small, <2 GB | ✅ | Over-smooths; invents plausible-but-wrong texture; fails on large uniform clipped areas. |
| **Deep SR-ITM** | 2019 ICCV (*older*, foundational) | **Joint** super-resolution + ITM; base/detail decomposition, SR on base, ITM on detail | official code `[NOT FOUND]` | moderate; 4K needs tiling | ⚠️ tiled | Task coupling leaks SR artifacts into HDR; architecture superseded. [37] |
| **Lightness-Modulated Deep ITM** | 2021 arXiv (*older*) | Modulate the ITM curve by predicted **lightness** so mid-tones stay put | `[NOT FOUND]` | small | ✅ | Lightness alone cannot separate diffuse white from emitter. [39] |
| **FMNet** | 2022 ACM MM | **Frequency-aware modulation** for SDR→HDR, band-wise | `[NOT FOUND]` | small–moderate | ✅ | Amplifies compression noise; still no semantic gating. [38] |
| **HDRTVNet** | **2021 ICCV** | Reformulate SDRTV→HDRTV as **global colour mapping + local highlight/detail enhancement**; introduces HDRTV1K | `github.com/chxy95/HDRTVNet` [3] | small CNN, <2 GB, fast | ✅ | Two-stage; highlight stage over-triggers on flat white; tuned to HDRTV1K content. |
| **HDRTVNet++** (Efficient SDRTV-to-HDRTV) | **2023 arXiv / IEEE TMM (≈2024–25)** | "Learning from image formation": explicit **degradation + quantization modelling**, reformulated supervision, lighter pipeline | `github.com/xiaom233/HDRTVNet-plus` [4][12][13] | small CNN | ✅ | Still no semantic prior; HDRTV1K bias; flat-white over-triggering persists. |
| **DIDNet** (Dual Inverse Degradation) | 2023–24 arXiv/IEEE | Model **real-world** SDRTV degradation (compression + display) as a dual inverse problem | `[NOT FOUND]` | small–moderate | ✅ | Degradation model mis-specified on clean synthetic renders; ring/halo on text. [40] |
| **Practical SDR-to-HDRTV Up-Conversion** | 2023 CVPR | New dataset + explicit degradation modelling for practical up-conversion | `[NOT FOUND]` | small | ✅ | Degradation assumptions are both the contribution and the fragility. [41] |
| **Spatial-Temporal Feature Fusion** | 2022 arXiv | Adds temporal fusion to ITM | `[NOT FOUND]` | moderate | ⚠️ | Early temporal attempt; flicker reduced, not solved. [17] |
| **Deep Conditional HDRI** | 2024 IEEE TMM | ITM via **dual encoder–decoder conditioning** | `[NOT FOUND]` | moderate | ✅ likely | Decoder can ignore conditioning on extreme inputs; no public weights. [43] |
| **RealRep** (attribute/style-disentangled) | **2026 AAAI-40**; arXiv 2025 | Disentangle **luminance vs chrominance/style** so one model generalizes across SDR styles | author page `github.com/kepengxu` `[SNIPPET]` [13][44] | moderate | ✅ likely | May strip intentional stylization; generalization needs independent replication. |
| **EIN** (Exposure-Induced Network) | 2025 ACM TOMM | Model the **exposure process** to recover over-/under-exposed regions | `[NOT FOUND]` | small–moderate | ✅ likely | Targets *camera* LDR — wrong exposure model for rendered/cel-shaded content. [45] |

**Unconfirmed — do not cite as established.** **HSIF** (hierarchical semantic inverse tone mapping): `[NOT FOUND]`; could not confirm the work's existence, venue or code — search hits conflate "HSIF" with unrelated hybrid-systems material. **SingleHDR**: `[NOT FOUND]` as an official paper; appears only in third-party citation prose ("Recently, SingleHDR [7] dissected the LDR generation process into several steps") and as a "SingleHDR*" row in benchmark tables `[SNIPPET]`. Probably real, but I could not reach its own venue page or repo.

**Row-level warning:** every "License" cell above that says *unverified* means exactly that. I could not open a single
repository in this session. **HDRTVNet / HDRTVNet++ are the two most likely to be usable locally, and they are also
the two you most need to license-check.** `[UNVERIFIED]`

---

## 3. Generative / diffusion / flow-matching / MLLM-guided HDR

This is where 2024–2026 activity is concentrated. Grouped by *what the generative model is asked to do*.

### 3a. Generative model produces HDR (or a gain map) end-to-end

**I could not locate an official repository for *most* methods in this subsection** — code URLs are `[NOT FOUND]` unless stated, and **no license is verified except where a search snippet exposed one** (ExpandDiff: CC BY-NC-SA 4.0; HDR-Video-Latent-Alignment: CC BY 4.0). All I/O descriptions are from the papers' own framing, not from running code. `[UNVERIFIED]`

| Method | Year + venue | Core idea | I/O | Compute / 12 GB 4070S? | Failure modes |
|---|---|---|---|---|---|
| **LEDiff: Latent Exposure Diffusion** | **2025 CVPR** | Diffusion in a **latent exposure** space over multiple LDR exposures, merged into HDR | LDR bracketed/stacked → HDR | diffusion-scale but latent; ⚠️ likely OK at reduced res/fp8 `[INFERRED]` | Needs **multiple exposures** → not single-image ITM as-is; exposure-alignment errors; generative texture hallucination. [5] |
| **HDR Image Generation via Gain Map Decomposed Diffusion** | **2025 ICCV** | Decompose: diffuse **SDR base** + diffusion-predicted **gain map** | SDR → SDR base **+ gain map** (→ HDR) | diffusion + small gain head; ⚠️ yes `[INFERRED]` | Gain map can encode spurious local bumps; gain-map delivery ecosystem still maturing; base stays clipped. [10][11] |
| **Boosting ITM via Diffusion Regularization** | **2025 ICCVW** (AIM 2025 challenge) | Use diffusion as a **regularizer/prior on top of an ITM backbone**, not as the generator | SDR → HDR | backbone + sampler; ⚠️ `[INFERRED]` | Regularization strength is a new knob; over-regularizing pulls toward the training grade. Trained 512×512 patches, batch 3, Adam `[SNIPPET]`. [6][46] |
| **LumaFlux** | **2026 arXiv 2604.02787** (UT Austin + Google) | **Physically-guided Diffusion Transformer**: a camera-response / photometric-calibration prior steers a DiT to lift 8-bit to HDR; explicitly benchmarks against CNN baselines incl. HDRTVNet++ | 8-bit SDR → HDR (evaluated vs expert tone-mapped SDR as perceptual ground truth) | **DiT-scale — heaviest class here**; ⚠️ fp8/quantization + tiling essentially mandatory `[INFERRED]` | DiT cost; prior assumes a plausible camera response (**weaker on synthetic renders / cel-shaded art**); still generative → can invent. [7][15] |
| **LumaGuide** | **2026 arXiv 2607.26237** | **Training-free**: shape the diffusion sampling distribution to yield HDR without fine-tuning | SDR → HDR | base model + modified sampler, **no training**; ⚠️ base-model dependent | Inherits the base model's exposure/colour biases wholesale; no per-content adaptation. [8] |
| **Generating HDR Video from SDR Video** | **2026 arXiv 2605.14703** | Extends generative HDR to the **video** domain from SDR sources | SDR video → HDR video | video diffusion — very heavy; ❌ not realistic on 12 GB at useful length/res `[INFERRED]` | Temporal consistency of a generative model; cost; flicker. [18] |
| **Dual-Output Multi-Exposure HDR Reconstruction** | **2026 arXiv 2608.05626** | Two outputs: fused multi-exposure reconstruction **and** a gain-map ITM path | LDR stack / SDR → HDR + gain map | moderate; ✅ likely | Multi-exposure input; two-output consistency. [47] |
| **Gain-MLP** | **2025 ICCV** | Lightweight **MLP for HDR gain-map *encoding*** (representing/compressing, not generating) | HDR → SDR + compact gain map | tiny MLP; ✅ trivial | Solves *encoding*, not *reconstruction* — pair with an ITM front-end. [48] |
| **SDXL_HDR_VAE** (community) | 2024–25 HF model | An SDXL **VAE variant for HDR** latents | HDR latents ↔ images | VAE-scale; ✅ | Community artifact, not peer-reviewed; HDR latent calibration undocumented. [49] |
| **LumiPic** (community LoRA) | 2025–26 HF model | **Single-image SDR→HDR LoRA**; the "semantic HDR via a LoRA" approach | SDR → HDR | LoRA on SDXL-class base, fp8 + tiled ≈ fits 12 GB; ✅ with quantization `[INFERRED]` | Community-grade; **its own discussion thread reports there is no EXR output** `[SNIPPET]` → may not be scene-linear; no objective eval; training grade/data unknown. [50][51] |

**The 2026 frontier — mid/late-2026 (what changed since the 2025 diffusion wave).** The headline shift is **efficiency**: instead of a full multi-step generator, 2026 work keeps a cheap gain-map predictor and adds a generative *touch*.
- **FastGaMer — *Efficient GainMap Learning for Practical Inverse Tone Mapping*** (**CVPR 2026**): takes the ICCV-2025 gain-map decomposition and strips its cost; a lightweight gain predictor rather than a full generator. SDR → SDR base **+ gain map**. ✅ likely comfortable on 12 GB `[INFERRED]`. Fails by capacity ceiling: gain maps still commit to the SDR base's clipping; numeric claims unverified (no full text). [58]
- **GMODiff — *One-Step Gain Map Refinement with Diffusion Priors*** (**arXiv 2512.16357, 2025 → ECCV-cycle 2026 venue**): keeps a cheap gain-map predictor but **refines it in ONE diffusion step**, getting generative quality at near-non-generative cost. SDR → HDR via gain map. ✅ plausible on 12 GB `[INFERRED]`. Fails when the prior is the ceiling; one-step limits how much structure can be invented; "one-step" gains are easy to over-claim. [59]
- **ExpandDiff — *Dynamic Range Expanding Diffusion*** (**arXiv 2609.39624, Oct 2026**): frames the diffusion process explicitly as **dynamic-range expansion** for single-image HDR. SDR/LDR → HDR. ⚠️ quantize + tile `[INFERRED]`. Very new, minimally replicated. **License visible in search: CC BY-NC-SA 4.0** `[SNIPPET]` — **non-commercial**, a real constraint if you ship it. [57]
- **HDR Video Generation via Latent Alignment with Logarithmic Encoding** (**arXiv 2604.11788, 2026**): **latent alignment** plus a **logarithmic** encoding, so the log domain stabilizes the dynamic range for a latent model. SDR video → HDR video. ❌ not realistic on 12 GB `[INFERRED]`. License: **CC BY 4.0** `[SNIPPET]`. [70]
- **Text-driven ITM for compressed SDR videos via multi-modal description** (**Pattern Recognition 2026**): supplies semantics as **text** rather than learning them implicitly, targeting *compressed* SDR video. ⚠️ partial on 12 GB. Fails with description quality; compression artifacts still amplified; text-conditioned hallucination. Repo `[NOT FOUND]`. [66]
- **Residual Diffusion Bridge Model (RDBM)** (**CVPR 2026**): a residual diffusion *bridge* for restoration, directly adaptable as an ITM backbone (a bridge is a near-deterministic path from degraded to clean). Not an ITM paper per se. **Official repo: `github.com/MiliLab/RDBM`** `[SNIPPET]` — one of the very few located official repos in the generative space. [63]
- **Adjacent, lower-priority:** *Promoting LDR Panoramas to HDRI with Rendering-Space Losses* (2026, Eyeline Labs — supervises on **how the HDRI relights**, not on pixels; panoramic/IBL-specific, needs a differentiable renderer) [64]; *Constrained Spectral Uplifting for HDR Environment Maps* (CGF / EGSR 2026, spectral env-map niche) [69]; *Physical Exposure-Prior + Gradual-Patch Network* and the **NTIRE 2026** burst-HDR challenge (multi-frame, not single-image ITM) [72].

### 3b. LLM / MLLM-guided exposure & lighting estimation

**MLLM-ITM** (**2026 IJCAI**): a **multimodal LLM promotes inverse tone mapping**, supplying the semantic/exposure reasoning the ITM network lacks. SDR (+MLLM guidance) → HDR. Repo `[NOT FOUND]`; author copy at Univ. Warwick WRAP [52][53]. Compute: VLM inference + ITM net, the VLM being the memory hog — ⚠️ a 7B-class VLM in 4-bit likely fits 12 GB, larger does not `[INFERRED]`. Failure modes: LLM hallucination transferred into pixels, latency, non-determinism, hard to evaluate. [52][53]
The broader 2025–26 trend — ask an MLLM *"what is this, and what should be bright?"*, then map the answer to a curve or gain field — shares the same hallucination and prompt-sensitivity risks and has no standard benchmark yet. `[INFERRED from MLLM-ITM + the semantic-HDR trend]`

### 3c. "Semantic HDR" — the hybrid pattern (most promising for a local pipeline)

**Semantic Aware Diffusion Inverse Tone Mapping** (**2024 arXiv 2405.15468**; journal version at IOP [54][55]) builds an **ordered semantic graph** data-driven, then lets diffusion do ITM *conditioned on semantics*; SDR → HDR; the clearest published statement of the semantic-conditioning idea; repo `[NOT FOUND]` [54].

**The hybrid *where-not-how* pattern** (my synthesis of LumaFlux + Gain-Map-Decomposed + Semantic-Aware + the diffuse-white patents, 2024–2026): the **generative model predicts WHERE highlights/emitters belong** (a mask or coarse gain field), while a **physically-based, monotone luminance mapping assigns the actual nits**; output is HDR10 PQ or SDR + gain map. This keeps generative hallucination confined to *localization* and photometric correctness in explicit, testable math. [7][8][10][23][24][54]

**Why the hybrid wins:** an end-to-end generator must learn both *semantics* ("that's a window, that's a white shirt") and *photometry* (203→1000 nits, PQ encoding). Split them and the fragile, data-hungry part becomes a small model or a quantized diffusion pass, while the photometric part becomes deterministic and unit-checkable.

---

## 4. Content-specific HDR

| Content type | Dominant failure | Published / documented mitigations |
|---|---|---|
| **Anime / illustration / cel-shaded** | Large **flat regions of pure/near-pure white** with *no gradient*. An ITM cannot tell "white shirt", "white sky", "paper" from "light source". Curves over-trigger → every panel glows. Generative models **invent texture that the art style forbids** (line art becomes noisy). | Highlight-**eligibility** gating that deliberately returns gain ≈ 1.0 on flat regions; **morphological open/close on the binary eligibility map** to remove speckle and keep connected bright objects; **guided filtering** (SDR as guide) to lock generated gain to original line edges; conservative 100-nit diffuse-white reading. See §5. [23][24][27] |
| **Game screenshots / UI / text-heavy** | **UI chrome (HUD, subtitles, menus) is deliberately pure white and flat.** Boosting it is objectively wrong — it is not a light source. Text antialiasing turns into ringing/halo. | Treat graphics as graphics: BT.709/sRGB/AdobeRGB content "may need **re-mapping**" per the Ultra HD Forum Blue Book [27]; AWS MediaConvert exposes an explicit **"reference white level in nits for all SDR inputs"** control, the industry's standard escape hatch [56]; use 100-nit white for UI layers and composite HDR only for the rendered 3-D layer. |
| **Paintings / scanned art** | Degraded, non-linear, often *photographed* under unknown illumination; no reliable camera response. | Least-served content class in the literature. Sap the ambition: gentle global expansion + paper-white anchoring; do **not** run a generative ITM on it. `[INFERRED]` |
| **Product / portrait photos** | Skin tones shift with aggressive curves; specular highlights on skin/eyes/products must stay believable; "beauty" over-brightening is immediately visible. | Perceptual colour metrics (δE-ITP in ICtCp) to bound skin-tone error [32]; keep the mapping monotone and skin-tone-anchored; evaluate with PU-PSNR on a matched pair if you have one. |
| **HDR-ready content with true highlights (night scenes, sunsets)** | Under-delivering: capping gain at 1.0 wastes real headroom. | This is the case where generative/gain-map methods genuinely pay off — the semantics are unambiguous. [7][10][54] |

### What specifically fails on flat cel-shaded whites

1. **Structureless input.** A flat white fill is a single constant. No gradient ⇒ no cue for direction or falloff.
2. **Saturation ambiguity.** Value 255 in an 8-bit SDR render means "nominal white", not "clipped luminance".
   The renderer may never have had >1.0 radiance there. Any ITM that assumes 255 ⇒ clipped emitter is wrong here.
3. **Morphology.** A 1-pixel antialiased black outline around a white region is often destroyed by any smoothing
   used to give the highlight a soft falloff.
4. **Generative hallucination is stylistically illegal.** Diffusion priors trained on photographs add texture/grain
   that breaks the cel-shaded look. [7][8][54]

---

## 5. Failure-mode catalogue (with mitigations)

| # | Failure mode | Signature | Mitigation |
|---|---|---|---|
| F1 | **"Everything glows"** | All near-white pixels boosted; image looks foggy/washed | Diffuse-white/flat-region gating; cap gain at 1.0 where no *supported* highlight exists; conservative reference white (100 nit) for flat content. [23][34] |
| F2 | **Flat-region over-trigger on cel-shaded whites** | Whole panels/UI glow | Binary **eligibility map** + **morphological opening/closing**; require a local gradient/curvature cue, not just a high value. [23][24] |
| F3 | **Halo / ringing at high-contrast edges** | Bright fringe around line art and text | **Guided filtering** with the original SDR as guide so the gain field inherits original edges; filter the *gain*, not the image. [24] |
| F4 | **Texture hallucination in clipped regions** | Invented detail inconsistent with the rest of the image | Confine the generative model to **localization** (mask/gain), not pixel synthesis; per-content gating; disable generation for art/UI. [7][54] |
| F5 | **Colour/hue shift** | Skin/foliage drift, especially with per-channel curves | Apply gain in a **luminance-preserving** space (ICtCp or a luma/chroma split), bound with **δE-ITP**; avoid independent per-RGB-channel expansion. [32] |
| F6 | **Temporal flicker (video)** | Highlights shimmer frame to frame | Global colour mapping + **multi-frame interaction**; spatial-temporal fusion; dedicated deflickering; hysteresis on the mask across frames. [16][17] |
| F7 | **Shot-change / scene-cut discontinuity** | Grade jumps at cuts | Pipeline-level: scene-cut detect → **reset temporal state / hysteresis**, re-estimate global mapping per shot. (No learned shot-change ITM was found — treat as an engineering task.) `[INFERRED]` |
| F8 | **Compression-noise amplification + banding** | Blocking/banding visible after expansion; Mach bands in skies | Degradation-aware models (DIDNet, CVPR 2023 practical up-conversion); denoise *before* gain; 10-bit minimum + dither; **PU21-based banding checks** (PU21 models banding explicitly). [30][33][40][41] |
| F9 | **Brightness preference bias** | Human raters prefer the *brighter* version even when wrong | Never tune to single-stimulus preference; use paired comparisons + full-reference metrics against a native grade. The classic 2009 study warns of exactly this. [34][42] |
| F10 | **Regulatory/metadata mismatch** | Player tone-maps your HDR back to SDR and it looks worse than the original SDR | Correct MaxCLL/MaxFALL, correct reference white, and **always keep the SDR base** (gain-map delivery) so the fallback is the untouched original. [25][26][27] |

---

## 6. Evaluation: HDR-specific metrics

| Metric | Year | What it measures | Status in ITM work | Notes |
|---|---|---|---|---|
| **PU21** (PU-PSNR / PU-SSIM / PU-MS-SSIM) | 2021 | Banding-aware **perceptually uniform encoding** that lets PSNR/SSIM be applied to HDR | **The de-facto ITM benchmark metric** — appears in method tables (PSNR / PU-PSNR / SSIM / PU-SSIM / SR-SIM) `[SNIPPET]` | Explicitly models **banding**, the dominant HDR artifact. Now in tooling (VMAFx commit #923). [30][33] |
| **HDR-VDP-3** | 2023 | Full-reference **multi-metric**: image differences, quality, *and* contrast distortions | The reference perceptual model; author PDF public. [28][29] | Supersedes HDR-VDP-2; use v3, and say so if you cite v2. |
| **δE-ITP (ICtCp)** | ITU/Dolby-derived | Perceptual **colour** difference in ICtCp | Standard colour metric in HDR pipelines | Available in `colour-science` (`colour.difference.delta_E_ITP`). [32][57] |
| **HDR-VQM** | 2015 (*older*) | Objective **video** quality for HDR | The video metric; predates deep ITM | Pair with a temporal-difference/flicker measure. [31] |
| HDR noise metric (adaptation + gradient) | **2026** (HVEI, IS&T) | Visually accurate **noise** metric for HDR, accounting for luminance adaptation and gradient effects | Newest perceptual-HDR metric found; noise-focused, not ITM-focused | Evidence the field is still building its metric stack in 2026. |
| **Tone-mapping-aware quality metrics** | **2026** (SIGGRAPH) | Quality metrics **adapted to tone-mapped content** — fixes the mismatch where standard metrics assume display-encoded input | The metric-premise correction for 2026; use alongside PU21, not instead of it | [68]; related: [71]. **This is the single most reportable 2026 metrics development.** |
| **SDR-referenced ITM video quality (broadcast)** | **2026** (ACM) | A **comprehensive dataset** plus an **SDR-referenced** quality method for inverse-tone-mapped HDR video in broadcast television | The first evaluation resource built specifically around *ITM'd* video rather than natively captured HDR | [65]. Use this for broadcast-video acceptance testing. |
| **Subjective studies** | — | Paired-comparison preference / annoyance | **Thin.** The classic study is 2009 (*older*) and warns about brightness bias. [34] | "Everything glows" remains well known anecdotally and **under-quantified in the literature I could reach.** |
| **ITM benchmarks / challenges** | 2025–2026 | Paired SDR/HDR data + rankings | **AIM 2025 ITM challenge report** (ICCVW 2025) is the key snapshot; **xDR** (2026) is a natively graded cinematic paired dataset; NTIRE 2026 covers **burst** HDR rather than ITM | ⚠️ **I found no AIM 2026 ITM challenge report** — as of Oct 2026 the AIM 2025 report remains the most recent ITM challenge I could locate. [1][2][14][72] |
| `tHDR` | — | *(named in the brief as an ITM benchmark)* | ⚠️ **`[NOT FOUND]`** — I could not locate a benchmark by this name. Possibly a misremembering, an internal/private benchmark, or too new/obscure to index. **Do not cite it without a source.** |

---

## Practical recommendations for a local Windows pipeline

Target: **12 GB RTX 4070 SUPER**, Windows, PyTorch + ComfyUI or a small standalone script.

1. **Do not start with an end-to-end generative HDR model.** Start with the hybrid design (§3c): a **highlight-support mask** + explicit luminance mapping. It fits in <2 GB, is deterministic, and is debuggable.
2. **Anchor white explicitly.** Default **203 nits** for photographic content; **100 nits** for graphics/UI/flat art. Make it a single parameter and record it in output metadata. `Lw = 100 cd/m²` is a documented BT.2408 choice. [20]
3. **Build the eligibility map conservatively, and filter the *gain*, never the image:** threshold near-white (`Y > ~0.9` after linearization) → binary map; **morphological open then close** to kill speckle and heal pinholes [23]; reject components that are **too flat** (low local gradient energy) or **too large** (sky, walls, flat fills); then blend with a **guided filter** using the original SDR as guide so edges come from the source [24]. Apply as `gain = 1 + (peak-1) * mask`, smooth `gain`, then multiply — that is what preserves line art.
4. **Use a small model for localization only.** HDRTVNet/HDRTVNet++ (small CNN) or a **quantized SDXL/FLUX LoRA** pass at ≤1024 px whose *only* output you consume is a coarse "where are the emitters" map — discarding its pixels. This is where LumiPic-style LoRA semantic HDR fits, but **verify its output format first** — its own model discussion notes there is no EXR, so it may not give you scene-linear data. [50][51]
5. **12 GB budgeting rules of thumb** `[INFERRED]`: fp8 or GGUF-quantized UNet/DiT with `--lowvram`-style offloading; **tiled VAE** decode (tile 512, overlap 64) — the VAE decode is usually what OOMs, not the UNet; cap working resolution at **768–1024 px** on the long edge and upscale the *mapped* result, not the HDR latents; an SDXL-class LoRA at 1024 px fp8 + tiled VAE fits comfortably, while a full **DiT (LumaFlux-class)** model should be treated as *not* comfortably local.
6. **Encode as SDR base + gain map, not a new PQ master**, whenever the source might be clipped. The ICC **ADGC** tag is the standards-track way to do this [25], `pyultrahdr` gives you tooling [26], and you keep a perfect SDR fallback (mitigates F10).
7. **Video:** process **shot by shot**, reset temporal hysteresis at cuts, apply **scene-level** global colour mapping with per-frame local refinement (the multi-frame-interaction pattern) [16], and check flicker with a temporal-difference metric before delivery.
8. **Validate with a metric stack, in this order:** PU-PSNR / PU-SSIM (if you have a reference) → HDR-VDP-3 → δE-ITP for colour → a **tone-mapping-aware perceptual** metric (SIGGRAPH 2026 *Adapting Quality Metrics to Tone Mapping* [68]; *Perceptual Assessment and Optimization of HDR Image Rendering* [71]) → HDR-VQM plus flicker for video → a banding check. PU21 is available in tooling [33]; HDR-VDP-3 has an author release [29]. For **broadcast video**, use the SDR-referenced ITM video-quality method and its dataset [65] rather than self-devised metrics. Then **always A/B against the original SDR** on both an HDR display *and* an SDR tone-mapped fallback — if the fallback is worse than the original SDR, your pipeline is net-negative regardless of HDR metrics.

---

## 7. Ready-made local tooling (2026)

Shippable SDR→HDR options I could locate. **None is a peer-reviewed ITM method, and all licenses/terms are unverified** `[UNVERIFIED]`.

| Tool | Type | Notes | 12 GB 4070S? |
|---|---|---|---|
| **NVIDIA RTX Video HDR / TrueHDR** | Driver/GPU feature (`nvidia-vfx` wheel, ComfyUI NVIDIA VFX nodes, `RTXVideoProcessor` CLI) | Real-time SDR→HDR on RTX hardware — the most "just works" local option, but a **fixed vendor algorithm**: you cannot gate flat cel-shaded whites or set reference white as §5 recommends. [61][62] | ✅ RTX-native |
| **LTX SDR-to-HDR** (open) + ComfyUI LTX nodes, and the `gregt/T2_IC_SDR2HDR_LTX_LoRA` community LoRA | Open video model + LoRA | Documented first-class SDR→HDR mode in an open video model with ComfyUI integration — the closest thing to an open, local, video-capable pipeline. Check the model card for terms. [60][75] | ⚠️ quantize, short clips |
| **Beeble SDR to HDR / SwitchHDR** | Commercial SaaS (2026) | Not local, no weights. Useful as a quality **ceiling** to benchmark your own pipeline against. [73] | ❌ cloud |

**Recommendation:** use RTX Video HDR or LTX as a **baseline to beat**, then apply the §5 gating on top (or instead) for flat cel-shaded / UI / text content, where a fixed vendor curve will over-brighten. `[INFERRED]`

---

## Sources

1. [AIM 2025 Challenge on Inverse Tone Mapping Report: Methods and Results — ICCVW 2025 / IEEE Xplore](https://ieeexplore.ieee.org/document/11375278) (2025)
2. [AIM 2025 ITM Challenge Report (arXiv 2508.13479) — alphaXiv mirror](https://www.alphaxiv.org/abs/2508.13479) (2025)
3. [github.com/chxy95/HDRTVNet — ICCV 2021 + TMM official code](https://github.com/chxy95/hdrtvnet) (2021/2023)
4. [github.com/xiaom233/HDRTVNet-plus](https://github.com/xiaom233/HDRTVNet-plus) (2023–2025)
5. [LEDiff: Latent Exposure Diffusion for HDR Generation — CVPR 2025](https://openaccess.thecvf.com//content/CVPR2025/html/Wang_LEDiff_Latent_Exposure_Diffusion_for_HDR_Generation_CVPR_2025_paper.html) (2025)
6. [Boosting Inverse Tone Mapping via Diffusion Regularization — ICCVW 2025 (AIM)](https://openaccess.thecvf.com/content/ICCV2025W/AIM/html/Lu_Boosting_Inverse_Tone_Mapping_via_Diffusion_Regularization_ICCVW_2025_paper.html) (2025)
7. [LumaFlux: Lifting 8-Bit Worlds to HDR Reality with Physically-Guided Diffusion Transformers — arXiv 2604.02787 (UT Austin + Google)](https://ar5iv.labs.arxiv.org/html/2604.02787) (2026)
8. [LumaGuide: Distribution Shaping for Training-Free HDR Generation in Diffusion Models — arXiv 2607.26237](https://arxiv.org/pdf/2607.26237) (2026)
9. [HDR Image Generation from Latent Codes (thesis, C. Wang, Saarland) — generative HDR background](https://publikationen.sulb.uni-saarland.de/bitstream/20.500.11880/40243/1/Thesis_Chao_Wang.pdf) (2024–25)
10. [HDR Image Generation via Gain Map Decomposed Diffusion — ICCV 2025](https://www.openaccess.thecvf.com/content/ICCV2025/html/Guan_HDR_Image_Generation_via_Gain_Map_Decomposed_Diffusion_ICCV_2025_paper.html) (2025)
11. [Gain-Map-Decomposed Diffusion — supplementary material, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Guan_HDR_Image_Generation_ICCV_2025_supplemental.pdf) (2025)
12. [Towards Efficient SDRTV-to-HDRTV by Learning from Image Formation — arXiv 2309.04084 (HDRTVNet++)](https://ar5iv.labs.arxiv.org/html/2309.04084) (2023)
13. [RealRep: Generalized SDR-to-HDR Conversion with Style Disentangled Representation Learning — arXiv 2505.07322 / AAAI-40](https://ar5iv.labs.arxiv.org/html/2505.07322) (2025/2026)
14. [The xDR dataset: A cinematic natively graded HDR & SDR dataset for evaluation of inverse tone mapping — Image Communication vol. 144](https://www.sciencedirect.com/science/article/abs/pii/S0923596526000536) (2026)
15. [LumaFlux — Semantic Scholar record](https://www.semanticscholar.org/paper/LumaFlux%3A-Lifting-8-Bit-Worlds-to-HDR-Reality-with-Saini-Gedik/ba14ca2c11c0b8b1e2f5395ce2b0522a319e6619) (2026)
16. [Video Inverse Tone Mapping Network With Global Color Mapping and Multi-Frame Interaction — IEEE](https://ieeexplore.ieee.org/abstract/document/11011675) (2025)
17. [SDRTV-to-HDRTV Conversion via Spatial-Temporal Feature Fusion — arXiv 2211.02297](https://ar5iv.labs.arxiv.org/html/2211.02297) (2022)
18. [Generating HDR Video from SDR Video — arXiv 2605.14703](https://ar5iv.labs.arxiv.org/html/2605.14703) (2026)
19. [ITU-R BT.2408-9 (03/2026) Guidelines for operational practices in HDR television production](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2408-9-2026-TOC-HTM-E.htm) (2026)
20. [ITU-R BT.2408-8 (2024) full PDF — incl. `Lw=100 cd/m² for downstream use`](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2408-8-2024-PDF-E.pdf) (2024)
21. [ITU-R BT.2446 — methods for converting SDR to HDR and vice versa](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2446-2019-PDF-E.pdf) (2019 — *older*)
22. [ITU-R BT.2390 — HDR TV broadcasting (SDR-in-HDR mapping, EETF)](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2390-8-2020-PDF-E.pdf) (2020 — *older*)
23. [US8050512 — ITM with diffuse-white maximum + morphological open/close filtering](http://patentimages.storage.googleapis.com/85/7c/7d/039bf11d050abf/US8050512.pdf) (patent)
24. [JP2023516184A / Method and apparatus for inverse tone mapping — per-region `YposDW` diffuse-white luminance](https://patents.google.com/patent/JP2023516184A/en) (patent)
25. [ICC.1 Adaptive Gain Curve Tag (ADGC) / `adaptiveGainCurveType` specification](https://www.color.org/specification/ICC.1_Adaptive_Gain_Curve.pdf) (2025)
26. [pyultrahdr — Ultra HDR / gain-map tooling (PyPI)](https://pypi.org/project/pyultrahdr/) (2025)
27. [Ultra HD Forum Blue Book 3.3.0 — SDR-BT.709/sRGB/AdobeRGB graphics remapping](https://ultrahdforum.org/wp-content/uploads/BlueBook_Ultra_HD_Production_and_Post_Production_3.3.0_Final.pdf) (2025)
28. [HDR-VDP-3: a multi-metric for image differences, quality and contrast distortions — arXiv 2304.13625](https://ar5iv.labs.arxiv.org/html/2304.13625) (2023)
29. [HDR-VDP-3 author PDF (R. Mantiuk, Cambridge)](https://www.cl.cam.ac.uk/~rkm38/pdfs/mantiuk2023hdrvdp3.pdf) (2023)
30. [PU21: perceptually uniform encoding for adapting existing quality metrics for HDR](https://api.repository.cam.ac.uk/server/api/core/bitstreams/fb5401af-2ef7-4af0-ac34-49f384c9660a/content) (2021)
31. [HDR-VQM: an objective quality measure for HDR video — Signal Processing: Image Communication](https://www.sciencedirect.com/science/article/abs/pii/S0923596515000703) (2015 — *older*)
32. [colour-science `delta_E_ITP` (δE-ITP in ICtCp) documentation](https://colour.readthedocs.io/en/v0.4.2/generated/colour.difference.delta_E_ITP.html) (tooling)
33. [VMAFx — PU21 PU-PSNR/PU-SSIM HDR metric implementation (docs, commit #923)](https://github.com/VMAFx/vmafx/blob/master/docs/metrics/pu21.md) (2025)
34. [Evaluation of reverse tone mapping through varying exposure conditions — ACM TOG / SIGGRAPH Asia 2009](https://dl.acm.org/doi/10.1145/1618452.1618506) (2009 — *older*, classic subjective study)
35. [Banterle et al., reverse tone mapping — SIGGRAPH Asia 2011 talk slides](http://vcg.isti.cnr.it/Publications/2011/BAADEGMM11/talk4_rtmo.pdf) (2011 — *older*)
36. [github.com/fxtmtrsine/hdr-expandnet — ExpandNet training/inference code](https://github.com/fxtmtrsine/hdr-expandnet) (2017 — *older*)
37. [Deep SR-ITM: joint super-resolution + inverse tone mapping — ICCV 2019](https://www.openaccess.thecvf.com/content_ICCV_2019/html/Kim_Deep_SR-ITM_Joint_Learning_of_Super-Resolution_and_Inverse_Tone-Mapping_for_ICCV_2019_paper.html) (2019 — *older*)
38. [FMNet: Frequency-Aware Modulation Network for SDR-to-HDR Translation — ACM MM](https://dl.acm.org/doi/pdf/10.1145/3503161.3548016) (2022)
39. [Lightness Modulated Deep Inverse Tone Mapping — arXiv 2107.07907](https://ar5iv.labs.arxiv.org/html/2107.07907) (2021 — *older*)
40. [Dual Inverse Degradation Network for Real-World SDRTV-to-HDRTV Conversion — arXiv 2307.03394](https://ar5iv.labs.arxiv.org/html/2307.03394) (2023)
41. [Learning a Practical SDR-to-HDRTV Up-Conversion Using New Dataset and Degradation — CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Guo_Learning_a_Practical_SDR-to-HDRTV_Up-Conversion_Using_New_Dataset_and_Degradation_CVPR_2023_paper.pdf) (2023)
42. [Deep Learning for HDR Imaging: State-of-the-Art and Future Trends — arXiv 2110.10394](https://ar5iv.labs.arxiv.org/html/2110.10394) (2021 — *older* survey)
43. [Deep Conditional HDRI: ITM via Dual Encoder-Decoder Conditioning — IEEE TMM](https://ieeexplore.ieee.org/document/10476730) (2024)
44. [github.com/kepengxu — RealRep author page](https://github.com/kepengxu) (2025–2026)
45. [EIN: Exposure-Induced Network for Single-Image HDR Reconstruction — ACM TOMM](https://dl.acm.org/doi/full/10.1145/3763240) (2025)
46. [Boosting ITM via Diffusion Regularization — ICCVW 2025 PDF (training details)](https://openaccess.thecvf.com/content/ICCV2025W/AIM/papers/Lu_Boosting_Inverse_Tone_Mapping_via_Diffusion_Regularization_ICCVW_2025_paper.pdf) (2025)
47. [Dual-Output Multi-Exposure HDR Reconstruction via SDR Fusion and Gain Map ITM — arXiv 2608.05626](https://arxiv.org/abs/2608.05626) (2026)
48. [Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP — ICCV 2025 project page](https://gain-mlp.github.io/) (2025)
49. [Felldude/SDXL_HDR_VAE — Hugging Face model](https://huggingface.co/Felldude/SDXL_HDR_VAE) (2024–25, community)
50. [oumoumad/LumiPic — Hugging Face, Single-Image SDR→HDR LoRA](https://huggingface.co/oumoumad/LumiPic) (2025–26, community)
51. [LumiPic discussion: "there is no EXR"](https://huggingface.co/oumoumad/LumiPic/discussions/2) (2026)
52. [MLLM-ITM: Multimodal Large Language Model Promotes Inverse Tone Mapping — IJCAI 2026](https://www.ijcai.org/proceedings/2026/169) (2026)
53. [MLLM-ITM — University of Warwick WRAP record](https://wrap.warwick.ac.uk/id/eprint/200769/) (2026)
54. [Semantic Aware Diffusion Inverse Tone Mapping — arXiv 2405.15468](https://ar5iv.labs.arxiv.org/html/2405.15468v1) (2024)
55. [Semantic Aware Diffusion Inverse Tone Mapping — IOP published version](https://beta.iopscience.iop.org/article/10.1088/1742-6596/3128/1/012009/meta) (2025)
56. [AWS Elemental MediaConvert API reference — "reference white level, in nits, for all of your SDR inputs"](https://docs.aws.amazon.com/mediaconvert/latest/apireference/) (tooling/doc)
57. [colour-science — colour difference / HDR tooling registry entry](https://tessl.io/registry/tessl/pypi-colour-science/0.4.0/files/docs/colour-difference.md) (tooling)

**Added in rev. 2 (mid/late-2026 frontier)** — numbering continues from 57:

| # | Source | Year |
|---|---|---|
| 58 | [FastGaMer: Efficient GainMap Learning for Practical Inverse Tone Mapping — CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Guan_FastGaMer_Efficient_GainMap_Learning_for_Practical_Inverse_Tone_Mapping_CVPR_2026_paper.html) · [supplementary PDF](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Guan_FastGaMer_Efficient_GainMap_CVPR_2026_supplemental.pdf) | 2026 |
| 59 | [GMODiff: One-Step Gain Map Refinement with Diffusion Priors for HDR Reconstruction — arXiv 2512.16357, ECCV-cycle venue](https://ar5iv.labs.arxiv.org/html/2512.16357) | 2025/26 |
| 60 | [SDR to HDR — LTX open-source model docs](https://docs.ltx.io/open-source-model/vfx-post-production/sdr-to-hdr) · [LTX ComfyUI nodes](https://docs.ltx.io/open-source-model/integration-tools/ltx-comfy-ui-nodes) · [ComfyUI built-in nodes](https://docs.comfy.org/built-in-nodes/ImageColorSpace) | 2026 |
| 61 | [nvidia-vfx Python wheel (RTX Video HDR / VSR)](https://pypi.org/project/nvidia-vfx/) · [Comfy-Org/ComfyUI_Nvidia_VFX_Nodes](https://github.com/Comfy-Org/ComfyUI_Nvidia_VFX_Nodes) | 2026 |
| 62 | [DrC0ns0le/RTXVideoProcessor — CLI for RTX VSR + TrueHDR tone mapping](https://github.com/DrC0ns0le/RTXVideoProcessor) | 2026 |
| 63 | [MiliLab/RDBM — CVPR 2026 Residual Diffusion Bridge Model, official repo](https://github.com/MiliLab/RDBM) | 2026 |
| 64 | [Promoting LDR Panoramas to HDRI with Rendering-Space Losses — Eyeline Labs](https://eyeline-labs.github.io/HDRI/assets/poster.pdf) | 2026 |
| 65 | [Inverse-Tone-Mapped HDR Video Quality Assessment for Broadcast Television: dataset + SDR-referenced method — ACM](https://dl.acm.org/doi/abs/10.1145/3746027.3755844) | 2026 |
| 66 | [Text-driven inverse tone mapping for compressed SDR videos via multi-modal description — Pattern Recognition](https://www.sciencedirect.com/science/article/abs/pii/S0031320326019874) | 2026 |
| 67 | [A CBCT-Tailored Inverse Tone Mapping Method for Enhancing Contrast and Brightness — IEEE](https://xplorestaging.ieee.org/document/11667261) | 2026 |
| 68 | [Adapting Quality Metrics to Tone Mapping — SIGGRAPH 2026](https://www.immersivecomputinglab.org/publication/adapting-quality-metrics-to-tone-mapping/) | 2026 |
| 69 | [Constrained Spectral Uplifting for HDR Environment Maps — CGF / EGSR](https://onlinelibrary.wiley.com/doi/10.1111/cgf.15280) | 2026 |
| 70 | [HDR Video Generation via Latent Alignment with Logarithmic Encoding — arXiv 2604.11788 (CC BY 4.0)](https://arxiv.org/html/2604.11788v2) | 2026 |
| 71 | [Perceptual Assessment and Optimization of HDR Image Rendering — CVPR 2024](https://openaccess.thecvf.com//content/CVPR2024/html/Cao_Perceptual_Assessment_and_Optimization_of_HDR_Image_Rendering_CVPR_2024_paper.html) | 2024 |
| 72 | [Physical Exposure-Prior + Gradual-Patch Network for Burst HDR Reconstruction — CVPRW 2026 (NTIRE)](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Gao_Physical_Exposure-Prior_and_Gradual-Patch_Network_for_Burst_HDR_Reconstruction_CVPRW_2026_paper.pdf) · [NTIRE 2026 Efficient Burst HDR challenge report](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Park_NTIRE_2026_Challenge_on_Efficient_Burst_HDR_and_Restoration_Datasets_CVPRW_2026_paper.pdf) | 2026 |
| 73 | [Introducing SwitchHDR / Beeble SDR to HDR — vendor docs](https://beeble.ai/updates/2026-07-03-introducing-switchhdr) | 2026 |
| 74 | [3D-LUT-based inverse tone-mapping for HDR/WCG display — arXiv 2309.17160](https://arxiv.org/pdf/2309.17160v1) | 2023 |
| 75 | [gregt/T2_IC_SDR2HDR_LTX_LoRA — Hugging Face LoRA model card](https://huggingface.co/gregt/T2_IC_SDR2HDR_LTX_LoRA/blob/main/README.md) | 2026 |
| 76 | [高动态范围成像研究进展及展望 (Progress and prospects in HDR imaging) — Journal of NUDT, Chinese survey](http://journal.nudt.edu.cn/gfkjdxxb/article/html/20260304?tab=abstract) | 2026 |
68. [Adapting Quality Metrics to Tone Mapping — SIGGRAPH 2026 (Immersive Computing Lab)](https://www.immersivecomputinglab.org/publication/adapting-quality-metrics-to-tone-mapping/) (2026)
69. [Constrained Spectral Uplifting for HDR Environment Maps — Computer Graphics Forum / EG Symposium on Rendering](https://onlinelibrary.wiley.com/doi/10.1111/cgf.15280) (2026)
70. [HDR Video Generation via Latent Alignment with Logarithmic Encoding — arXiv 2604.11788 (CC BY 4.0)](https://arxiv.org/html/2604.11788v2) (2026)
71. [Perceptual Assessment and Optimization of HDR Image Rendering — CVPR 2024](https://openaccess.thecvf.com//content/CVPR2024/html/Cao_Perceptual_Assessment_and_Optimization_of_HDR_Image_Rendering_CVPR_2024_paper.html) (2024 — *older*, but still the reference on this point)
72. [Physical Exposure-Prior and Gradual-Patch Network for Burst HDR Reconstruction — CVPRW 2026 (NTIRE)](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Gao_Physical_Exposure-Prior_and_Gradual-Patch_Network_for_Burst_HDR_Reconstruction_CVPRW_2026_paper.pdf) (2026) · [NTIRE 2026 Challenge on Efficient Burst HDR and Restoration](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Park_NTIRE_2026_Challenge_on_Efficient_Burst_HDR_and_Restoration_Datasets_CVPRW_2026_paper.pdf)
73. [Introducing SwitchHDR / Beeble SDR to HDR — vendor documentation](https://beeble.ai/updates/2026-07-03-introducing-switchhdr) (2026)
74. [Redistributing the Precision and Content in 3D-LUT-based Inverse Tone-mapping for HDR/WCG Display — arXiv 2309.17160](https://arxiv.org/pdf/2309.17160v1) (2023 — *older*, LUT-based ITM branch)
75. [gregt/T2_IC_SDR2HDR_LTX_LoRA — Hugging Face LoRA model card](https://huggingface.co/gregt/T2_IC_SDR2HDR_LTX_LoRA/blob/main/README.md) (2026)
76. [高动态范围成像研究进展及展望 (Progress and prospects in HDR imaging) — Journal of NUDT, 2026 survey, Chinese](http://journal.nudt.edu.cn/gfkjdxxb/article/html/20260304?tab=abstract) (2026)

*Source access note: URLs above are the canonical locations returned by search. Where an `ar5iv.labs.arxiv.org/html/<id>` link is used, the canonical form is `arxiv.org/abs/<id>`. No URL in this document was successfully fetched and read in full — see the Evidence-grade notice at the top.*
