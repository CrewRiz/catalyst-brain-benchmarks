# Evidence Map

This repo is the public, reproducible evidence harness for the closed-source
`catalyst-brain` SDK. It should prove useful integration behavior without
publishing SDK internals, operational secrets, or private business runbooks.

## Current Public Evidence

| Claim area | Current benchmark evidence | Boundary |
|---|---|---|
| HKVC memory scaling | Fixed Catalyst state is compared with FP16 KV cache, TurboQuant, KIVI, and PyramidKV memory models. | Memory footprint model only; not an end-to-end model quality benchmark. |
| HKVC exact-key retrieval | Exact indexed hit latency and value correctness are measured as entries increase. | O(1) language applies to the exact-key path measured here. |
| HKVC fallback behavior | Missing-key fallback latency is reported separately from exact hits. | Fallback should not be presented as the same path as indexed lookup. |
| HKVC recency position | First/middle/last retrieval correctness and latency are measured. | This is a position probe, not a full adversarial long-context evaluation. |
| Token efficiency | Progressive tool discovery and deferred output records are compared with full context payloads. | Savings are byte/token estimates for agent context, not provider billing statements. |
| Tool-routing quality | Deterministic MCP-style query set checks top-1 and top-3 selection accuracy. | Does not claim arbitrary semantic search quality. |
| Quantum-inspired attention | `PyQuantumAttentionHead` routing accuracy and latency are measured against a cosine softmax reference. | Measures classical public-wheel behavior, not physical quantum acceleration. |
| Rain state transfer | Rain binary/header size and round-trip checks are measured across dimensions. | Header viability is checked against an 8 KB common limit, not every proxy configuration. |
| HDC correctness | Bind/unbind recovery and chained composition resonance are tested. | Does not disclose implementation internals. |

## Next Evidence Targets

| Priority | Benchmark to add | Why it matters |
|---|---|---|
| 1 | Real model adapter smoke for `catalyst-kv-cache` against a tiny local Transformer. | Shows drop-in integration behavior beyond memory modeling. |
| 2 | RULER-style synthetic long-context retrieval with controlled keys/values. | Tests whether the cache preserves useful retrieval under long contexts. |
| 3 | SWE-bench or tau-bench style agent loop with `CatalystTokenKernel`. | Connects token savings to pass rate and iteration quality. |
| 4 | Rain multi-agent handoff across serverless request boundaries. | Validates the strongest deployment wedge for edge/serverless workflows. |
| 5 | Import/telemetry nonblocking tests from a clean wheel install. | Protects the freemium production funnel and verifies opt-out behavior. |

## Claim Hygiene

Public claims should name the measured scope. Prefer phrases like "public-wheel
benchmark", "explicit memory model", "exact-key indexed path", and
"quantum-inspired" when those are the actual measured conditions.

Avoid claiming solved end-to-end model quality, arbitrary O(1) behavior, or
physical quantum speedups until a public benchmark in this repo measures that
specific thing.
