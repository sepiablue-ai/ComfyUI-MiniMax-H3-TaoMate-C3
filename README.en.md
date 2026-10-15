# ComfyUI MiniMax H3 TaoMate C3

[日本語](README.md) | English

**A ComfyUI optimization package that generates TaoMate-H3 (MiniMax H3 distilled to 3 steps) at 1024×1792 with audio in about 2 min 27 s on an RTX 4070 12GB.** Includes a Windows installer and sample workflows.

| FL2VA (one first frame), 1024×1792, 124 frames | Generation time |
|---|---:|
| 4 steps, W4A4 + Streaming VSA 5% (no TaoMate) | 218.2 s (3 min 38.2 s) |
| TaoMate 3 steps + 10% sparse attention | 183.3 s (3 min 3.3 s) |
| **This repo: TaoMate 3 steps + W4A4 + 10% sparse + held weights (C2/C3)** | **146.9 s (2 min 26.9 s)** |

Measured on [Sayaka Benchmark](https://sayakabenchmark.pages.dev/), FL2VA tab, case01 (2026-09-30, one run each). The rows are FL2VA [018](https://sayakabenchmark.pages.dev/benchmarks/fl2va/018-w4a4-streaming-vsa5-i2v/workflow.json), [020](https://sayakabenchmark.pages.dev/benchmarks/fl2va/020-taomate-h3-fl2va-i2v-sparse10/workflow.json) and [021](https://sayakabenchmark.pages.dev/benchmarks/fl2va/021-taomate-h3-fl2va-i2v-c3-fastest/workflow.json) (entry numbers on the site). The step from row 2 to row 3 is about 20% and also includes the video VAE change (standard VAE → FastVAE).

- **Reference:** sampling at 544×960 and saving 1088×1920 with [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream) (FL2VA [016](https://sayakabenchmark.pages.dev/benchmarks/fl2va/016-taomate-3step-c3-sparse10-x2vae-int8-stream/workflow.json)) took 51.7 s. That workflow is not bundled here.

**Terms**

| Term | Meaning |
|---|---|
| TaoMate 3-step | Alibaba TaoLive's LoRA that lets H3 generate in 3 steps |
| FC1 W4A4 | 4-bit weights and activations for the first FFN projection (FC1); converted once at install time |
| 10% sparse attention | Computes only the important attention blocks (about 10% kept) |
| C2 (held FFN8) | Runs the FFN in 8 chunks and keeps its weights where needed, reducing repeated transfers and casts |
| C3 (held attention-out) | Also keeps the attention output layer's INT8 weights; about 3% faster than C2 |

> Author: Kamimoto (かみもと) | Quick results on **[X (@sep_is_heim)](https://x.com/sep_is_heim)**, write-ups (Japanese) on **[note](https://note.com/sepiablue)** ([article index](https://note.com/sepiablue/n/n8e612c619a3e)).

The package includes a Windows installer, locked custom-node dependencies, GUI/API workflows, and the original test first-frame image. Model weights and converted caches are separate. No files from the research project are required after copying this folder.

This uses ordinary H3 sparse attention. It is a different configuration from the older Streaming VSA 5% package. The supplied workflow is **I2V using the FL2VA model**, not Ref2VA.

## Credits and this package's contribution

This package uses the work of the following authors and projects.

| Author / project | Contribution used here |
|---|---|
| [MiniMax AI](https://huggingface.co/MiniMaxAI/MiniMax-H3) | MiniMax H3 base model and weights |
| [Alibaba TaoLive AIGC Team](https://huggingface.co/TaoLiveAIGC/TaoMate-H3) | TaoMate-H3 three-step distilled LoRA |
| [CZMartin22](https://huggingface.co/CZMartin22/TaoMate-H3-3step-ComfyUI) | Conversion of the TaoMate LoRA to ComfyUI format |
| [Zironic / H3-Optimizations](https://github.com/Zironic/H3-Optimizations) | Sparse attention, memory optimization and HeldMLP / INT8 primitives |
| [MotionCache-FastVAE](https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE) | Fast video VAE decoding |

This repository assembles **FC1 W4A4 cache installation/loading, weight holding across FFN8 chunks (C2), and a private INT8 attention-out weight copy (C3)** into an installable configuration built on those existing primitives. It includes pinned dependency installation, cache reuse, GUI/API workflows, an input image and measured RTX 4070 conditions. TaoMate training/distillation and the ComfyUI LoRA conversion belong to the authors credited above. See [sources and licenses](THIRD_PARTY_NOTICES.md).

## RTX 4070 12GB measurements

The sample uses 1024×1792, 124 frames, 24 fps, seed 43 and three-step `res_multistep` sampling.

| Measurement | Result |
|---|---:|
| Matched C2, normal-cache warm run | 149.392 seconds |
| Matched C3, normal-cache warm run | **145.021 seconds** |
| Time saved by adding C3 | **4.371 seconds / approximately 2.93%** |
| 2026-09-30, first generation in a fresh ComfyUI 0.38.0 checkout, including model loading | **172.29 seconds** |

Ten comparison videos were used during adoption; decoded frames and PCM matched within same-seed C2/C3 pairs. The speed difference and equality checks concern the **C2→C3 change**. They do not establish an overall speedup over unoptimized TaoMate or identical quality to dense H3. Warm and first-run timings have different conditions.

After installation through `setup.bat` in the fresh ComfyUI checkout, one submission of the supplied API workflow produced one successful video with AAC audio and a passing full FFmpeg video/audio decode. The existing dedicated venv and FC1 cache were reused, with no model downloads. A completely new Python environment and fresh 50-layer cache conversion were outside this check. Details are in [comparison results](docs/RESULTS.md), [installation and generation validation](VALIDATION.md), and [environment versions](compatibility.json).

## Install

1. Prepare a MiniMax H3-capable ComfyUI and its own venv or portable embedded Python, then place the five files from [docs/MODELS.md](docs/MODELS.md). Validated platform: Windows 11, RTX 4070 12GB. Exact versions are in [compatibility.json](compatibility.json).
2. Clone this repository or extract its ZIP anywhere, including directly under `ComfyUI/custom_nodes`. Git for Windows is needed when dependencies are missing.
3. Stop ComfyUI and double-click `setup.bat`. Select the **ComfyUI directory containing `main.py`**, not its portable parent. When Python selection is ambiguous, specify it explicitly:

```bat
setup.bat -ComfyRoot "C:\AI\ComfyUI" -Python "C:\AI\.venv\Scripts\python.exe"
```

In PowerShell, use `.\setup.bat`. A portable example is `-ComfyRoot "C:\AI\ComfyUI_windows_portable\ComfyUI" -Python "C:\AI\ComfyUI_windows_portable\python_embeded\python.exe"`.

Setup installs this package, the dependencies below, and the example image at `ComfyUI/input/taomate_c3/first_frame.png`. When all five models are available and the FC1 cache is missing, setup automatically performs the first offline GPU conversion. Stop competing GPU workloads before running setup. Missing models do not prevent node installation: setup then prints **GENERATION NOT READY** with the remaining requirements. Use `-CheckOnly` for a strict readiness check.

| Dependency | Locked commit | Used for |
|---|---|---|
| [H3-Optimizations](https://github.com/Zironic/H3-Optimizations) | `862774944a331bc1a66cee1cf805b94bd138ebbe` | H3MemoryOptimization, H3SparseAttention and native INT8 operations |
| [MotionCache-FastVAE](https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE) | `b719329e0ecf35f0ae08d241c363ed1e56adbb95` | MiniMaxH3FastVAEDecode only |

KJNodes, the older W4A4-VSA pack, and a VSA Gate model are unnecessary for this workflow. Setup fetches these exact commits and verifies Python sources and the Windows DLL against [dependency-lock.json](dependency-lock.json). It refuses to reset or overwrite incompatible existing installations.

The included `.gitattributes` preserves the original package bytes across Git checkouts. Dependency Python files allow only LF/CRLF differences; DLL fingerprints remain byte-exact.

## Models and FC1 cache

Download the five assets listed in [docs/MODELS.md](docs/MODELS.md). Existing `extra_model_paths.yaml` mappings are honored.

- `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- `TaoMate-H3-3step-ComfyUI.safetensors`
- `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `minimax_h3_video_vae_int8_convrot.safetensors` (**the tested Kijai file**)
- `minimax_h3_audio_vae_fp32.safetensors`

Setup does not download weights. Identically named Video VAE files from different publishers can contain different weights. File sizes are checked; add `-VerifyModels` for full SHA-256 checks.

FC1 is converted offline once; generation does not quantize model weights at runtime. The destination is `ComfyUI/models/h3_preconverted/taomate_c3_fc1`, approximately **3.86 GB / 3.59 GiB**.

**Reuse an existing unmerged FL2VA C3 cache:**

```bat
setup.bat -ComfyRoot "C:\AI\ComfyUI" -Python "C:\AI\.venv\Scripts\python.exe" -CacheSource "D:\H3-cache\fl2va_fc1_gate"
```

All 50 FC1 shard hashes are verified. Files are hardlinked on the same volume when supported, otherwise copied. A new FC1-only manifest is written without changing the source manifest; Gate weights are omitted. Do not overwrite hardlinked shards later. Premerged-LoRA and Ref2VA caches are incompatible.

**Build a new cache (also automatic during normal setup when all models are present):**

```bat
setup.bat -ComfyRoot "C:\AI\ComfyUI" -Python "C:\AI\.venv\Scripts\python.exe" -BuildCache -Gpu 0
```

This uses the GPU. Stop competing GPU workloads first. The converter processes 50 layers, checking native INT4 execution, save/load equality and CPU offload/reload equality. Incomplete conversions are not published as a ready cache. GPU selection applies only to the child process.

## Validate and run

```bat
setup.bat -ComfyRoot "C:\AI\ComfyUI" -Python "C:\AI\.venv\Scripts\python.exe" -CheckOnly
"C:\AI\.venv\Scripts\python.exe" -B -X utf8 tools\check_runtime.py --comfy-root "C:\AI\ComfyUI" --native --gpu 0
```

`-CheckOnly` checks pinned dependencies, model sizes, all FC1 shard hashes and the example input without writing files. `-VerifyModels` additionally reads and hashes all five large model files. `check_runtime.py` loads real ComfyUI classes and validates the 22-node API prompt, including the lazy dependency import. `--native` adds a small native INT4 GPU round trip and H3 DLL ABI check. Neither command starts a server or generates video.

Start ComfyUI with your usual launcher and load [TaoMate_C3.json](workflows/TaoMate_C3.json). For API benchmarks use [TaoMate_C3.api.json](workflows/TaoMate_C3.api.json). If your models are in subdirectories, select their actual relative names in the nodes. The original test image and prompt are included. Other images/prompts may have different quality and timing.

| Parameter | Supplied workflow |
|---|---|
| Resolution / frames / fps | 1024×1792 / 124 / 24 |
| Seed / sampler | 43 / res_multistep |
| Sigmas | `1.0, 0.9611650705337524, 0.8533332943916321, 0.0` |
| Sigma shift | video 12 / audio 3 |
| TaoMate LoRA | strength 1.0, original unmerged model |
| Sparse | budget 0.1, early/late boost OFF, 4096 rows |
| MLP / attention out | Held FFN8, threshold 4096 / held INT8 out |
| FastVAE | tile_batch_size 2 |

For warm timings with normal caching, run seed 43 then seed 44 so the sampler executes again. A repeated identical prompt that hits every cache is not a generation benchmark. The adopted environment used `--disable-pinned-memory`. Do not add `--cache-none` or `--fast fp16_accumulation` to the default launch for this baseline.

## Existing C3 workflows and benchmarks

This pack registers `H3TaoMateW4A4Loader`, `H3C2HeldFFN8` and `H3C3HeldOut`. The old C3 loader ID, `H3V2PreconvertedLoader`, belongs to the separate W4A4-VSA pack. Use the supplied API graph or migrate a copy:

```bat
"C:\AI\.venv\Scripts\python.exe" -B -X utf8 tools\migrate_workflow.py "old_c3.api.json" "portable_c3.api.json"
```

Only the loader ID, cache directory, and original test input filename are replaced. Prompts, seeds and numerical settings are preserved. The input file is never overwritten. This utility does not turn a VSA or C2-only graph into C3.

The manually installed `ComfyUI-Sayaka-H3-C2C3` pack or standalone `h3_c2_held_ffn8.py` / `h3_c3_held_out.py` files register duplicate IDs. Setup detects these and stops. With ComfyUI stopped, back them up and rename them with a `.disabled` suffix, or move them outside active custom_nodes. Setup never removes them automatically. Extra custom-node directories from `extra_model_paths.yaml` are also checked.

## Evidence and scope

In the original controlled C2→C3 comparison, normal-cache warm time changed from **149.392 to 145.021 seconds (about 2.93%)**. Ten completed videos were compared; decoded frames and native PCM matched within each same-seed C2/C3 pair. This evaluates the C3 holding change relative to C2; it does not claim that TaoMate, W4A4 or sparse attention is lossless relative to dense full-step H3. A later separate C3 session measured **143.521 seconds** warm; it is not a guaranteed runtime.

See [VALIDATION.md](VALIDATION.md) for packaging checks and untested items, [RESULTS.md](docs/RESULTS.md) for adopted/rejected options, and [source-provenance.json](docs/source-provenance.json) for original code hashes.

Setup never uses global pip, upgrades ComfyUI/Torch/Kitchen, or edits standard launchers, shared models or ACLs. Revalidate C3's internal API dependencies before updating the locked upstream versions.

## Author and related repositories

I'm Kamimoto (かみもと). I measure and publish MiniMax H3 and ComfyUI speed-ups and VRAM reductions on an RTX 4070 12 GB.

- X: [@sep_is_heim](https://x.com/sep_is_heim) — quick results with step counts, seconds and speed-ups
- note: [かみもと｜ローカルAI生成](https://note.com/sepiablue) — setup guides and measured write-ups in Japanese ([article index](https://note.com/sepiablue/n/n8e612c619a3e))
- [Sayaka Benchmark](https://sayakabenchmark.pages.dev/) — speed and quality comparison of H3 workflows

| Repository | What it does |
|---|---|
| [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream) | INT8 X2 VAE and asynchronous NVENC saving; 5 s 1088×1920 video with audio in about 60 s |
| [ComfyUI-MiniMax-H3-W4A4-VSA](https://github.com/sepiablue-ai/ComfyUI-MiniMax-H3-W4A4-VSA) | Ref2VA acceleration with FC1 W4A4 and Streaming VSA |
| [ComfyUI-H3-LBH-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-LBH-X2-Stream) | 3 low-resolution steps + 1 high-resolution step; about 25% faster for 14 s video |
| [ComfyUI-MiniMax-H3-PDMD-X2](https://github.com/sepiablue-ai/ComfyUI-MiniMax-H3-PDMD-X2) | PDMD 2-step fused model; 5 s 1088×1920 video in about 44–50 s |

This package is [GPL-3.0-only](LICENSE). External dependencies are fetched separately with their original distributions and notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
