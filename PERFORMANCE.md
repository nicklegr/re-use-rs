# Performance loop — RTF 0.25 target in progress

Current target: 300 s offline.exe process <= 75 s, including startup/I/O.
Start from 54de173 (RTF 0.498059). Same original-reference precision gate.
Prior RTF 0.50 results below are the starting baseline, not the new target.

RTF 0.25 loop: Mamba dimensions/reverse/taps/rank specialized at compilation;
dt weights cached per thread. Interleaved old/new/new/old 10 s fixture:
3.8944 / 3.7997 / 3.8017 / 3.8896 s (about 2.3% faster).
All 130 s of reference fixtures bit-identical; 60 s = 29.049 s under this load.
Logs: mamba-specialized-ab-*.log, mamba-specialized-validation.log.
Nsight Compute counters unavailable (ERR_NVGPUCTRPERM); driver policy unchanged.
Accepted: scan block 32 after balanced 256/128/64/32/32/64/128/256 trials.
10 s times: 256=3.629/3.632, 128=3.519/3.526, 64=3.508/3.514,
32=3.384/3.375 s; all bit-identical. Full 130 s gate passes; 60 s=25.802 s
(RTF 0.4300). Log: mamba-scan32-validation.log. Temporary tuning env removed.
Accepted: CUDA forward/backward branches use two persistent scoped streams;
their projections overlap, with the original final addition order unchanged.
60 s=25.275 s (RTF 0.4212); all 130 s reference fixtures bit-identical.
Log: mamba-streams-validation.log. Other GPU backends retain sequential branches.
Accepted: fuse CUDA projection sums, layer norm, outer residual and transpose,
preserving the original four-lane reduction and separately rounded products.
Mamba stage 0.634 -> 0.596 s; 60 s=24.367 s (RTF 0.4061).
All 130 s reference fixtures bit-identical; strict CUDA clippy passes.
Log: mamba-finish-validation.log.
Rejected Mamba probes: unchecked indexing and padded projection width offered
no speedup; separate delta/gate precomputation slowed Mamba to 0.713/0.759 s.

Rules: AGENTS.md. Reference: original CUDA implementation at 88638f3.
Hardware: i5-12400, RTX 5060 Ti 16 GB, CUDA 13, Windows. All runs at 44.1 kHz,
5 s chunks, 50% overlap; batch worker stopped; existing kernel cache.

## Final verification

- Normal offline.exe, 300 s corpus (119 windows): inference 147.0 s;
  complete process including load, warmup and WAV I/O **149.417689 s**, RTF
  **0.498059**. Valid 13,230,000-sample finite output. Log: production-300s.log.
- 60 s corpus, three repeats: 28.355048 / 28.389864 / 28.366988 s
  (RTF 0.472584 / 0.473164 / 0.472783).
- Five speech/noise previews, silence, quiet, impulses, DC, and three 60 s runs:
  11,025,000 sample comparisons; max float difference 0, max PCM16 difference 0,
  bit changes 0. Log: final-validation.log. 300 s run checks throughput/validity;
  its full waveform was not separately compared with a 300 s original reference.
- cargo fmt --check; strict clippy --all-targets --release with and without
  cuda; CPU tests and 7 CUDA-enabled unit tests pass.

## Accepted improvements (separate commits)

| Change | Five 10 s previews, total processing |
|---|---:|
| Original | 102.472 s |
| Cache-bounded CPU im2col, scratch reuse, contiguous row copies | 93.531 s |
| CUDA encoder/decoders and GPU-resident activations | 27.978 s |
| Shared-memory FP32 GEMM with exact K=256 rounding | 27.305 s |
| Pack pitched input once per convolution, not per tile | 18.520 s |

Original 60 s reference: 159.093 s. Earlier production 300 s: about 833 s.
Nsight identified repeated pitched-to-contiguous copies as 33.7% of GPU time.
Packing once removes up to 55 copies of a large activation per convolution.
All accepted variants passed original-reference comparisons.

Numerics: preserve matrixmultiply's 256-wide K blocks and sequential FP32 FMA;
use FP64 normalization moments and separate FP32 multiply/add rounding.
Unblocked GEMM/fused normalization changed up to 50 PCM16 LSB through later
Mamba TF32 rounding, so was rejected. Bounds-check removal alone was slower
and was discarded. Keep the original Mamba arithmetic.

## Reproduce

Build: ../build-cuda.ps1. Also build --release --features cuda --example perf.
All fixture/baseline binaries, WAVs, logs: ../work/perf-loop (not tracked).

- `perf record|check <model> <reference-dir> <input.wav>...`: chunked RTF and
  per-sample float/PCM16/bit comparison; never overwrite immutable references.
- `perf fixture <out.wav> <seconds> <input.wav>...`: concatenate/repeat without
  resampling. corpus-60s uses the five previews; corpus-300s repeats corpus-60s.
- `perf signals <dir>`: deterministic 44.1 kHz edge fixtures.
- `perf exec <offline.exe> <model> <input.wav> <out.wav>`: full process wall time,
  output validity, and a failing exit code if RTF > 0.25.
- REUSE_PROFILE=1: synchronized encoder/Mamba/decoder times; unset for timing.
