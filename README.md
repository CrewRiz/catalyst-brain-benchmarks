# Catalyst Brain Benchmarks

Open benchmark suite for verifying Catalyst Brain SDK behavior from the public
PyPI wheels.

The benchmark code is MIT licensed. The `catalyst-brain` SDK is a separate
closed-source, freemium commercial SDK distributed through PyPI. These tests use
only the public SDK API and do not require source access.

## What This Verifies

| Suite | What it checks |
|---|---|
| Install smoke | `pip install catalyst-brain==1.3.2`, package versions, HDC import, HoloSwarm API parity |
| Token discovery | Progressive tool discovery versus repeatedly sending full verbose schemas |
| Deferred outputs | Code/tool outputs stay out of context until explicitly fetched |
| HDC primitives | Bind, unbind, bundle, resonance latency and throughput |
| Bind/unbind correctness | Exact recovery through direct and chained HDC binding |
| HKVC scaling | Median and p95 query latency as stored entries increase |
| Memory model | Explicit FP16 KV-cache model compared with fixed Catalyst Rain state |
| KV-cache comparison | FP16, TurboQuant, KIVI, PyramidKV, and Catalyst HKVC memory-model comparison |

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
catalyst-brain-bench --mode quick --out results/local
```

The command writes:

- `results/local/latest.json`
- `results/local/*.csv`
- `results/local/README.md`
- `results/local/charts/*.svg`

For a larger run:

```bash
catalyst-brain-bench --mode full --out results/local-full
```

## Latest Published Results

See [results/README.md](results/README.md).

Charts from the latest checked-in run:

- [Token savings](charts/token_savings.svg)
- [Deferred output savings](charts/deferred_output_savings.svg)
- [HKVC query latency](charts/hkvc_query_latency.svg)
- [HDC primitive latency](charts/hdc_primitive_latency.svg)
- [Bind/unbind chain correctness](charts/chain_correctness.svg)
- [Memory model](charts/memory_model.svg)
- [KV-cache comparison](charts/kv_cache_comparison.svg)

Comparison assumptions and primary paper links are documented in
[docs/SOURCES.md](docs/SOURCES.md).

## Claim Discipline

These benchmarks are meant to be reproducible, not rhetorical.

- The suite uses public APIs only.
- Results vary by hardware, Python version, and operating system.
- The memory chart is an explicit model, not process RSS. It compares a stated
  FP16 transformer KV-cache formula with the SDK's fixed world-vector size and
  measured compressed Rain header size.
- The KV-cache comparison chart models memory footprint from published
  compression or retention targets. It does not claim equal model quality,
  equal serving behavior, or source access to competitor systems.
- Catalyst Brain uses classical HDC and quantum-inspired algorithms. This suite
  does not claim physical quantum behavior.
- Token savings are byte/token estimates for agent context payloads, not LLM
  billing statements from a provider.

## CI

The GitHub workflow installs `catalyst-brain==1.3.2` from PyPI, runs the quick
benchmark, and uploads generated results as artifacts. It does not need SDK
source code or private credentials.

## License

Benchmark code is MIT licensed. The Catalyst Brain SDK is governed by its own
commercial license and terms.
