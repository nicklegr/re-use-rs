# Performance loop

Target and acceptance rules: AGENTS.md. Original source: 88638f3.
Hardware: i5-12400, RTX 5060 Ti 16 GB, CUDA 13, Windows.
Baseline single 5 s window: 6.934 s; chunked 300 s: about 833 s (RTF 2.78).
CPU encoder/decoder account for 84% of single-window time.

Build: ../build-cuda.ps1 (release, --locked, --features cuda).
Fixtures and immutable baseline binaries/outputs: ../work/perf-loop (not tracked).
Rust harness: `cargo build --release --features cuda --example perf`, then
`perf record|check <model> <reference-dir> <input.wav>...`. Uses 5 s / 50%
chunking, one warmup, reports chunked RTF and fails any sample beyond 1 LSB.

Accepted: cache-bounded CPU im2col tiles, reusable scratch, contiguous row copies.
Five original 10 s previews (15 windows): 102.472 -> 93.531 s (-8.7%).
All 2,205,000 samples bit-identical; 6 CPU unit tests pass. Logs: baseline.log,
cpu-tiles.log. Short fixtures have only 1.5x work from overlap; long-file target
still requires further improvement.

Accepted: CUDA encoder/decoders, strict FP32 GEMM, FP64 norm moments, GPU-resident
activations through Mamba. Preserve CPU's K=256 accumulation blocks and separate
norm multiply/add rounding. 5x10 s: 93.531 -> 27.978 s; all samples bit-identical.
CUDA GEMM/norm unit test passes. Log: gpu-conv-3.log. Long-file target remains.
Rejected prototype: unblocked FP32 GEMM/fused norm changed up to 50 LSB because
small encoder differences affect the original Mamba's TF32 rounding.
Long fixture: `perf fixture ../work/perf-loop/corpus-60s.wav 60 <five previews>`.
It concatenates/repeats original samples in Rust, with no resampling. Preserve
its reference using the immutable baseline-perf.exe in record mode.
Original 60 s reference: 159.093 s, RTF 2.652.
Accepted: shared-memory FP32 GEMM with CPU K=256 rounding in one launch.
5x10 s: 27.978 -> 27.305 s; 60 s: 41.835 s (RTF 0.697), all samples identical.
Removed the direct cubek dependency. REUSE_PROFILE=1 prints synchronized CUDA
encoder/Mamba/decoder times; leave unset for benchmarks. Log: gpu-blocked-mm.log.

Accepted: pack pitched convolution input once, before the im2col tile loop.
Nsight: repeated scalar copies consumed 33.7% of GPU time. Packing once removes
up to 55 copies of the same large activation per convolution.
5x10 s: 18.520 s; 60 s: 28.354 s, RTF 0.4726 (target met).
All 4,851,000 compared samples are bit-identical. 7 CUDA/CPU unit tests pass.
Log: gpu-pack-once.log. Bounds-check removal alone was slower and was discarded.
