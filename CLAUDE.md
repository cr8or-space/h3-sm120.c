# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A single-binary C/CUDA inference engine for **MiniMax-H3** (text/image/video →
video+audio). It is a fork of the DGX Spark (GB10, `sm_121`, unified memory,
ARM64) CUDA port, which itself ports [antirez/h3.c](https://github.com/antirez/h3.c)
from Metal. **This fork targets SM120: the RTX PRO 6000 Blackwell Max-Q** —
compute capability 12.0, 96 GB of discrete GDDR7 over PCIe on an x86_64 host.

Almost all inherited prose (README, `CHANGELOG.md`, everything in `docs/`),
comments, timings, md5 references and tuning choices describe **GB10**. Do not
quote them as SM120 results. Where GB10 behaviour hinged on unified memory, Spark
NVMe, or GB10's instruction set, re-measure on SM120 instead of assuming.

**Do not add personally identifying information to the repo:** no machine or
host names, user names, home-directory paths, internal URLs or network details.
Naming the GPU model and its configuration is fine. Never hard-code a
weights path; resolve it through `H3_MODEL_ROOT`.

## Weights

The official BF16 checkpoint is already in the Hugging Face cache under
`$HF_HOME`. Its snapshot directory contains `FL2VA/` and `Ref2VA/` and is the
model root. `Makefile.linux` and `scripts/model_root.sh` (sourced by the other
scripts) both default `H3_MODEL_ROOT` to the newest
`$HF_HOME/hub/models--MiniMaxAI--MiniMax-H3/snapshots/*`, else `./MiniMax-H3`.
The make default is exported to test recipes. When running `./h3` or a test
binary by hand, set it yourself:

```bash
export H3_MODEL_ROOT=$(ls -d "$HF_HOME"/hub/models--MiniMaxAI--MiniMax-H3/snapshots/* | tail -1)
```

`h3_load_dir()` in `h3.c` requires `FL2VA/{transformer,text_encoder,video_vae/source,audio_vae,tokenizer}`;
`Ref2VA/transformer` is optional (only loaded for `--ref-*` requests). Load BF16 only, never
community int8/fp8/nvfp4 repacks. Quantization happens at load time.
`$HF_HOME` is on network storage, so the first load of a shard is slow. Time
warm repeat runs, not the first run.

## Build

Only `Makefile.linux` matters. The root `Makefile`, `h3_gpu.m`, `h3_shaders.metal`,
`h3_metal.m` and `h3_tokenizer.m` are the Apple/Metal originals and are kept for
reference. They are not built here.

```bash
make -f Makefile.linux -j$(nproc) h3
./h3 --info -d "$H3_MODEL_ROOT"
```

`CUDA_ARCH` (default `120`) sets `-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)`.
`CUDA_ARCH=121` rebuilds for GB10. After changing it, run `make -f Makefile.linux clean`,
because objects don't depend on the flags. Toolchain:
`/usr/local/cuda/bin/nvcc` (CUDA 13.x), gcc, `libicu-dev`, cuBLAS/cuBLASLt. At
runtime `ffmpeg`/`ffprobe` must be on `PATH` (override with `H3_FFMPEG` /
`H3_FFPROBE`). Media I/O is spawned, never linked, so without them the MP4
mux and `test_av_mux` fail.

## Tests

Each test is a standalone binary built from `tests/<name>.c` and linked
against every library object. There is no test framework. A test prints its
failures and exits non-zero.

```bash
make -f Makefile.linux test               # host tests + CUDA ops + per-module real-weight smokes
make -f Makefile.linux test-step          # DiT block/forward full + int8, text encoder full
make -f Makefile.linux test-conditional   # scripts/smoke_conditional.sh: FL2VA + Ref2VA end to end (minutes)
H3_CONDITIONAL_SKIP_REF_VIDEO=0 make -f Makefile.linux test-conditional   # + long --ref-video case
```

Run a single test by building its target and passing the model root (or setting `H3_MODEL_ROOT`):

```bash
make -f Makefile.linux h3_cuda_ops && ./h3_cuda_ops            # kernel unit tests, no weights
make -f Makefile.linux h3_tests && ./h3_tests                  # host-only: layout, sigmas, RNG, safetensors, resize
make -f Makefile.linux h3_cuda_dit_block_smoke && ./h3_cuda_dit_block_smoke "$H3_MODEL_ROOT" [full]
make -f Makefile.linux h3_cuda_dit_forward_smoke && ./h3_cuda_dit_forward_smoke "$H3_MODEL_ROOT" [full|int8]
```

Weight-dependent targets print `skip:` and pass when weights aren't found.
Check for skips rather than trusting a green exit. Most `tests/test_real_*`,
`test_metal.c`, `test_bf16.c` and `test_text_metal.c` are Metal-era parity tests
against MLX fixtures in `misc/fixtures/`. Those fixtures are not present, so the
Linux build does not wire them up.

Microbenches: `make -f Makefile.linux h3_sdpa_bench && ./h3_sdpa_bench <seq> <heads> <head_dim> <iters>`
(for example `44800 56 128 3`, the 15 s sequence length), and `h3_gemm_precision`.

## Architecture

```
main.c / h3_cli.c / linenoise.c   CLI flags + interactive REPL (`!sol-attn on`, etc.)
h3.c / h3_multimodal.c            public API (h3.h), generate pipeline, FL2VA vs Ref2VA routing
h3_host.c                         layout, sigmas, RNG, Euler solver, RGB resize
h3_safetensors.c / h3_weights.c   header parse + pread; shard discovery -> h3_gpu_tensor_load_*
h3_dit.c / h3_dit_schedule.c      50-block DiT denoise loop, fusion/int8 gating, layer skip + reuse
h3_text_encoder.c                 Qwen3 text encoder (with host-thread weight prefetch ring)
h3_vision_encoder.c               Qwen3-VL vision for reference images/video
h3_video_vae.c / h3_video_encoder.c / h3_audio_vae.c   decode/encode stacks
h3_ffmpeg.c / h3_terminal.c       spawn ffmpeg for media; --show terminal preview
─────────────── h3_gpu.h ───────────────   the only GPU boundary (~87 functions)
h3_gpu.cu                         the entire CUDA backend: context, allocator, loader, every kernel
h3_gpu_stubs.c                    functions from h3_gpu.h not implemented on CUDA
h3_cuda.c / h3_device.c           device probe -> h3_device_info (Metal field names are reused)
```

Key cross-file facts:

- **Model code never touches CUDA directly.** Everything goes through
  `h3_gpu.h`. A new op means a declaration in `h3_gpu.h`, an implementation in
  `h3_gpu.cu`, and a check in `tests/test_cuda_ops.c`. `h3_gpu_stubs.c` is
  generated by `scripts/gen_gpu_stubs.py` from an `IMPLEMENTED` set. When
  an op goes from stub to real, update that set, or regenerate by hand. The
  remaining stubs (F32 DiT AdaLN/gate/QKV and Metal NAX MLP) are intentional:
  see `docs/KNOWN_ISSUES.md`.
- **Metal names survive as capability flags.** `h3_gpu_is_m5()` means "fast
  path". It and `h3_gpu_has_int8_mlp()` are true when `props.major >= 12`,
  which includes SM120. `h3_device_info.metal4` / `apple_gpu_family` hold
  the CUDA capability. `"h3_shaders.metal"` is still passed to `h3_gpu_create`
  and is ignored, because the kernels are compiled in by nvcc.
- **Default DiT path is runtime INT8** (weights quantized from BF16 at load;
  dynamic activation quant; tiled int8 GEMM) with hand-fused QKV+RMS+RoPE,
  AdaLN+gate+quant, patch and final-head kernels. SDPA is the "MMA" tile
  kernel (64-token tiles, head_dim 128, QK/V B-maps via `ldmatrix.x2`).
  `--ssd-streaming` forces BF16 (incompatible with int8).
- **Every fusion has an env-var oracle** (`H3_DISABLE_FUSED_*`,
  `H3_DISABLE_INT8_*`, `H3_BF16_MLP`, `H3_SDPA_LDMATRIX=0`, `H3_SDPA_HALF`, …;
  `grep -n 'getenv("H3_' h3_dit.c h3_gpu.cu`). Keep an oracle for any new fused
  path so A/B against the unfused path stays possible.
- **Sessions keep models on the card.** Without `-p`, `./h3` is a REPL that
  also reads prompts piped on stdin. It holds the Qwen weights resident
  (`h3_text_encoder_load`; `H3_TEXT_RESIDENT=0|1`), rebinds the retained DiT
  to each new prompt (`h3_dit_rebind`), and keeps the video decoder.
  Single-shot `-p` runs still load everything per process. Loader fan-out
  was re-measured on SM120 (2026-09-17) and is bound by page-cache copy
  speed, not by `H3_LOAD_STAGE_MIB` or `H3_LOAD_READ_THREADS`: leave it.
- **GB10-shaped assumptions to revisit on SM120:** the 24 GiB memory-pool
  release threshold in `h3_gpu_create`;
  MMA/GEMM tile widths that were chosen because wider variants (`tcgen05`,
  WGMMA, `ldmatrix.x4`) were closed on GB10. Re-probe those on SM120
  (`tools/h3_ldmatrix_map.cu` is the probe pattern) rather than inheriting the
  verdict. `--info` still prints `h3-spark`. Its "unified memory yes" is
  CUDA's unified *addressing* flag, not shared physical memory: on SM120 the
  96 GB is separate from host RAM.

## Open SM120 work

Remaining items from the first SM120 baseline, best first. Measurements go in
`docs/PERF_BASELINE.md`. When an item is finished, delete it from this list.

1. **INT8 GEMM options on SM120.** INT8 GEMM is ~60% of short-clip denoise.
   GB10 found that the `i32` accumulator write alone is 19% of INT8 GEMM time,
   and that cuBLASLt there could not write BF16/F32 or take per-vector scales
   (2026-08-25 REJECT). Re-probe that heuristic table on `sm_120` with the
   current CUDA before assuming the same answer.
2. **Video VAE decode (~22 s on a 15 s clip).** The F32 path lost on GB10's
   memory limits. SM120 has 70+ GiB of VRAM headroom, so re-price the higher
   precision variants and the tile choice there. Bigger tiles showed seams:
   keep that REJECT.
3. **Anchored session prompts still reload.** With `--first`/`--last` or
   references, each new prompt re-runs the vision encoder and VAE encoder
   loads (fox-s2 knobs: 14.8 s per new prompt, against ~3 s for plain T2VA).
4. **Long-N SDPA needs a new kernel shape, not a retune.** Tile, warp,
   occupancy and barrier probes were all REJECT on 2026-09-17. The open idea
   is fewer K/V staging bytes per FLOP, e.g. several query tiles sharing one
   staged KV tile.
5. **Memory-pool release threshold.** The 24 GiB value in `h3_gpu_create`
   is still unmeasured on SM120.

## Generate presets and quality rules

The `--profile` flag prints per-phase wall time (text encoder, DiT denoise
split into `linear`/`sdpa`, VAE). Reference presets (seed 42):

```bash
# fox-s2: short pipeline smoke
./h3 --profile -d "$H3_MODEL_ROOT" -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 --steps 2 --layers 35 --reuse 1 --seed 42 -o outputs/fox-s2.mp4
# fox-fast: the bit-gate preset
./h3 --profile -d "$H3_MODEL_ROOT" \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 --frames 22 --steps 20 --layers 45 --reuse 2 --seed 42 -o outputs/fox-fast.mp4
```

`outputs/` is gitignored. `scripts/perf2_fox.sh <name> [extra args]` runs
fox-fast and computes PSNR/SSIM against a reference MP4 (`PERF2_REF`, output
dir `PERF2_OUT`).

Inherited performance discipline:
- Default flags are the **quality path**. `--token-reduction`, `--sol-attn`,
  `--reuse 3` and `H3_INT8_VAE=1` stay opt-in and are never bit-identical.
- A default-path optimization is KEEP only if fox-fast stays bit-identical, or
  measures **PSNR ≥ 24 dB and SSIM ≥ 0.85** against the reference. Otherwise
  it is opt-in or REJECT. Record KEEP/REJECT with numbers.
- The SM120 bit gate is fox-fast md5 prefix **`4facfc896f6f`** (fox-s2
  `146495086e36`), recorded from unmodified v0.2.2 kernels on 2026-09-17. It
  is deterministic run to run. GB10's `f5282774d3a4` does not apply here,
  because cuBLASLt picks different algorithms. The first SM120 table in
  `docs/PERF_BASELINE.md` has the warm walls and phase splits to compare against.
- `docs/PERF_BASELINE.md` is a dated log, newest sections at the top. Add
  SM120 measurements as new dated sections labelled with the GPU. Don't
  rewrite GB10 history. `H3_VERSION` lives in `h3.h`; releases are cut in
  `CHANGELOG.md`.
