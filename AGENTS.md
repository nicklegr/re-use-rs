# Performance work rules

- Target: processing time / audio duration <= 0.25 on the local RTX 5060 Ti,
  at 44.1 kHz with the original weights, 5-second chunks and 50% overlap.
  Measure `enhance_chunked`, including overlap and completed GPU work. Also
  report startup separately; never present a single-window rate as chunked RTF.
  Final acceptance must also time the normal offline executable on 300 seconds,
  including model load, warmup and WAV input/output, with total wall RTF <= 0.25.
  Start with Mamba, then investigate convolution and CPU/GPU overlap as needed.
- Preserve output against the original CUDA implementation at commit 88638f3:
  every sample must differ by <= 1/32768 in float amplitude and <= 1 count after
  signed 16-bit PCM rounding. Check finite values and identical sample counts.
  No resampling, reduced overlap, model changes or lowered-precision shortcuts.
- Run a measurement / improvement / output-comparison loop. Keep only verified
  improvements. Use several representative speech/noise recordings and an
  end-to-end chunked benchmark, with the batch job stopped.
- Implement improvements and the benchmark/comparison logic inside Rust.
  External scripts may invoke builds, but must not implement the optimization.
- Commit each accepted improvement separately, including its validation.
  Never commit model weights, audio fixtures or build products.
- Documentation should be only enough for an agent to resume: commands,
  baseline, accepted results and important failed experiments in PERFORMANCE.md.
- Do not restart the user's stopped batch job as part of performance testing.
