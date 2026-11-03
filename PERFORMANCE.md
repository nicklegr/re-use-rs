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
