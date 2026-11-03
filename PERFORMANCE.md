# Performance loop

Target and acceptance rules: AGENTS.md. Original source: 88638f3.
Hardware: i5-12400, RTX 5060 Ti 16 GB, CUDA 13, Windows.
Baseline single 5 s window: 6.934 s; chunked 300 s: about 833 s (RTF 2.78).
CPU encoder/decoder account for 84% of single-window time.

Build: ../build-cuda.ps1 (release, --locked, --features cuda).
Fixtures and immutable baseline binaries/outputs: ../work/perf-loop (not tracked).
