# Performance state — RTF 0.25 work paused

Paused at the user's request after the current experiment. Target remains
RTF <= 0.25; it has NOT been reached. Rules: AGENTS.md.

## Final measurements

- Normal offline.exe on 300 s: inference 101.2 s; complete child process
  **106.441088 s, RTF 0.354804**, including load, warmup and WAV I/O.
  Valid 13,230,000-sample finite output. The Rust 0.25 gate correctly fails.
  Log: ../work/perf-loop/production25-paused-300s.log.
- Previous RTF 0.50 endpoint (54de173): 300 s process 149.417689 s,
  RTF 0.498059. The final process is about 28.8% faster.
- Latest 60 s chunked accuracy run: 18.133295 s, RTF 0.302222.
  Do not extrapolate this short run to the 300 s process result.
- Original CUDA reference is commit 88638f3. Five speech/noise previews,
  silence/quiet/impulse/DC and the 60 s corpus total 130 s / 5,733,000 samples:
  max float difference 0, max PCM16 difference 0, changed bits 0.
  Log: moments-coalesced-validation.log. The 300 s run measures throughput
  and validity; its full waveform was not separately compared to an original
  300 s reference.
- Final fmt check, strict release clippy with/without CUDA, 6 CPU tests and
  9 CUDA-enabled tests pass. Failed experiments removed; accepted code committed.

Hardware: i5-12400, RTX 5060 Ti 16 GB, CUDA 13, Windows. 44.1 kHz, original
weights, 5 s windows / 50% overlap throughout. User's batch worker remains stopped.

## Accepted changes since 54de173 (one commit each)

- 1b71dc7: specialize Mamba shapes/direction/taps/rank; cache dt weights.
- a54b961: scan blocks of 32 threads.
- 3a2fc71: forward/backward Mamba branches on persistent CUDA streams.
- a7903e9: fuse projection sums, layer norm, outer residual and transpose.
- 04c8513: implicit convolution, without im2col/output-tile concatenation.
- 3b25c55: Mamba CMMA tilewise reads and wider row tiles.
- 30ce8cb: scan A coefficients in shared memory to free registers.
- 9071b4a: parallel FFT/IFFT frames and independent CPU spectral operations;
  retain original overlap-add order.
- 7fb8183: wider CMMA column tiles.
- b967c55: fuse convolution scale/shift/PReLU, rounding product with FMA-to-zero.
- e566135: coalesced FP32 reads for FP64 normalization moments.

Numerics: convolution GEMM retains sequential FP32 FMA within K=256 partial
sums. Keep FP64 moments and separately rounded products. Mamba retains the
original TF32 CMMA instruction/K order, four-lane norm reduction and sequential
scan. Every accepted change passed the original-reference fixture gate.

## Important rejected probes

- Raw MMA changed up to 2.07 PCM16 LSB: fails precision; do not enable it.
- Projection padding, separate delta/gate preparation, unchecked kernels:
  no gain or slower. Larger convolution tiles add excessive JIT time.
- Two-window processing: VRAM pressure and no throughput improvement on retry.
- Final input-projection/causal-convolution fusion: bit-identical on two 10 s
  previews but Mamba 0.524 -> 0.554 s (64 rows, K=8), or 0.630 s (32 rows,
  K=32). Discarded. Work files: fused-input-probe.patch, gpu-input32-k32.rs,
  gpu-input-64-k8.rs; logs fused-input*-probe.log.
- Nsight Compute counters unavailable (ERR_NVGPUCTRPERM); driver policy untouched.
  Nsight Systems works; latest trace nsys-implicit.nsys-rep/.sqlite.

## FP32 summation-order trial — rejected

User-requested bounded experiment; convolution multiplication/accumulation stayed
FP32, with Mamba and normalization unchanged. Original acceptance gate unchanged.
Removing K=256 partial sums alone gave essentially no speedup. Two interleaved
sums were also negligible; CubeK SimpleUnit FP32 GEMM with bounded im2col tiles
was slower (20 s fixtures: 5.521 s vs 4.748 s).
On the 60 s corpus: original 18.143636 s; wider tiles with original sum order
17.737818 s (bit-identical); wider tiles plus continuous sum 17.498688 s.
Combined gain 3.55%; reassociation adds only 1.35% over the wider-tile control.
Full 130 s candidate comparison: max float difference 121.190079 LSB,
max PCM16 difference 121 (silence); speech/noise previews reached 37 PCM16 LSB.
GPU warmup also grew from 4.878 to 7.453 s. Candidate rejected; original code
and default benchmark binary restored. Restored two-preview comparison is exact.
Logs: fp32-order-{control-*,wide-validation,cubek-probe,restored-check}.log.
Reproduction artifacts outside Git: fp32-order-wide-probe.patch and
fp32-order-wide-perf.exe (candidate REUSE_FP32_ORDER=2, REUSE_FP32_TILE=1).

## Reproduce / artifacts

All binaries, immutable original references, WAV fixtures and logs are outside
Git in ../work/perf-loop. Never overwrite reference WAVs. Latest full precision
log: moments-coalesced-validation.log. Normal output: production25-paused-300s.wav.

Build in a developer PowerShell: `. ../work/perf-loop/env.ps1`, then
`cargo +stable build --offline --locked --release --features cuda --example perf --example offline`.
The local work toolchain is Rust 1.99.1; do not use the old global toolchain.

- `perf check <model> <reference-dir> <input.wav>...`: fixed 5 s / 50% overlap,
  finite/length check and every-sample float/PCM16/bit comparison.
- `perf exec <offline.exe> <model> <input.wav> <new-output.wav>`: complete process
  time and validity, fails when RTF > 0.25; never overwrites output.
- `perf fixture <new-out.wav> <seconds> <input.wav>...`: concatenate/repeat,
  no resampling. corpus-300s repeats corpus-60s five times.
- REUSE_PROFILE=1 synchronizes encoder/Mamba/decoder boundaries; unset for timing.
- All temporary tuning switches and fused-input code have been removed.
