# Benchmark Sources And Assumptions

This suite uses public `catalyst-brain` APIs plus explicit comparison models.
The comparison data is intentionally narrow: memory footprint only, not
end-to-end model quality, perplexity, task score, or serving throughput.

## Catalyst Brain

- Package under test: `catalyst-brain==1.3.3` from PyPI.
- Public APIs used by the suite: `CatalystTokenKernel`, `ToolSpec`, `PyHKVC`,
  `PyQuantumAttentionHead`, `RainPayload`, Rain serialization helpers,
  `PyHoloSwarm`, and HDC primitive operations from the public wheel.
- HKVC comparison state: fixed 4096-dimensional public SDK state, modeled as
  4096 float32 values.
- Rain state size: measured by `CatalystTokenKernel.export_rain_snapshot(...)`
  and direct `RainPayload` round trips during benchmark execution.
- HKVC exact-key hits and missing-key fallback are measured separately so the
  indexed path and approximate fallback path are not conflated.
- Tool-selection accuracy uses a deterministic public MCP-style catalog and
  query set. It checks whether compact discovery preserves routing quality for
  that harness; it does not claim general semantic retrieval quality.
- Quantum attention head benchmarks use deterministic HDC keys/values, noisy
  query probes, and a pure-Python cosine softmax reference. They measure
  public-wheel routing behavior and latency, not physical quantum execution.

## Transformer KV-Cache Model

The baseline memory model is:

```text
tokens * 40 layers * 4096 hidden * 2 K/V tensors * 2 FP16 bytes
```

This is a stated model for reproducible comparison. It is not process RSS.

## Published Comparison Anchors

- TurboQuant, arXiv:2504.19874
  <https://arxiv.org/abs/2504.19874>

  The chart models the paper's 3.5-bit per-channel KV-cache point as
  `3.5 / 16` of the FP16 KV-cache baseline.

- KIVI, arXiv:2402.02750
  <https://arxiv.org/abs/2402.02750>

  The chart models the KV-only 2-bit setting as `2 / 16` of the FP16 KV-cache
  baseline. This isolates KV-cache storage rather than total process memory.

- PyramidKV, arXiv:2406.02069
  <https://arxiv.org/abs/2406.02069>

  The chart models the published 12% retained-cache setting as `0.12` of the
  FP16 KV-cache baseline.

## Claim Boundary

These charts are designed to be useful and reproducible without exposing
Catalyst Brain source code or trade secrets. They show integration behavior,
public SDK measurements, and explicit memory models. They do not claim physical
quantum behavior or disclose the closed-source SDK internals.
