# Catalyst Brain Benchmark Results

- SDK version: `1.3.3`
- Mode: `quick`
- Platform: `macOS-26.4-arm64-arm-64bit`
- Python: `3.9.6`
- Started: `2026-05-11T07:52:53.133277+00:00`

## Headline Checks

| Check | Result |
|---|---:|
| Best progressive discovery context saved | 99.56% |
| Best deferred-output context saved | 99.72% |
| Tool discovery top-1 accuracy | 100.00% |
| Largest modeled KV context | 128000 tokens |
| Catalyst HKVC memory at largest context | 0.016384 MB |
| Catalyst HKVC reduction versus FP16 model | 5120000.00x |
| HKVC median latency span across quick scaling run | 0.5000 us |
| HKVC largest hit vs miss-fallback latency | 12.0420 us / 47633.7505 us |
| Rain largest header under 8 KB | `True` |

## Install Smoke

| Check | Value |
|---|---:|
| Distribution version | `1.3.3` |
| Module version | `1.3.3` |
| Version OK | `True` |
| Bind/unbind resonance | 1.0 |
| HoloSwarm property length | 128 |
| HoloSwarm method length | 128 |

## Token Discovery Savings

| Tools | Full manifest bytes | Compact page bytes | Compact saved | Discovery saved | One-schema saved |
|---:|---:|---:|---:|---:|---:|
| 10 | 36774 | 2893 | 92.13% | 95.57% | 89.91% |
| 50 | 184665 | 5807 | 96.86% | 99.12% | 97.98% |
| 100 | 369531 | 5807 | 98.43% | 99.56% | 98.99% |

## Tool Selection Accuracy

| Tools | Top-1 accuracy | Top-3 accuracy | Median compact saved |
|---:|---:|---:|---:|
| 8 | 100.00% | 100.00% | 66.29% |
| 50 | 100.00% | 100.00% | 99.42% |
| 100 | 100.00% | 100.00% | 99.73% |

## Deferred Output Savings

| Stdout bytes | Full result bytes | Compact status bytes | Saved | Saved tokens |
|---:|---:|---:|---:|---:|
| 1000 | 1281 | 301 | 76.50% | 245 |
| 10000 | 11032 | 302 | 97.26% | 2682 |
| 100000 | 108533 | 303 | 99.72% | 27058 |

## HKVC Query Scaling

| Entries | Median query us | p95 query us | Value OK | Confidence |
|---:|---:|---:|---:|---:|
| 100 | 11.6250 | 12.8330 | `True` | 0.0117 |
| 1000 | 12.2080 | 12.3750 | `True` | 0.0002 |
| 5000 | 12.1250 | 15.2500 | `True` | 0.0001 |

## HKVC Path Breakdown

Exact-key hits use the indexed public API path. Missing-key fallback is reported separately because it performs approximate recall work.

| Entries | Exact hit median us | Miss fallback median us | Fallback / exact | Exact value OK |
|---:|---:|---:|---:|---:|
| 100 | 11.7080 | 830.7705 | 70.96x | `True` |
| 1000 | 12.0840 | 9331.3955 | 772.21x | `True` |
| 5000 | 12.0420 | 47633.7505 | 3955.63x | `True` |

## HKVC Recency Position Probe

| Entries | Bucket | Position | Median query us | Value OK | Confidence |
|---:|---|---:|---:|---:|---:|
| 1000 | first | 0 | 11.7500 | `True` | 0.000000 |
| 1000 | middle | 500 | 12.3750 | `True` | 0.000219 |
| 1000 | last | 999 | 12.1250 | `True` | 0.000840 |
| 5000 | first | 0 | 11.7500 | `True` | 0.000000 |
| 5000 | middle | 2500 | 12.2080 | `True` | 0.000078 |
| 5000 | last | 4999 | 12.0420 | `True` | 0.000550 |

## KV-Cache Competitor Model

This table is a stated memory model at the largest context in the run. It compares published compression or retention targets against the fixed Catalyst Brain public SDK state size. It is not an end-to-end quality benchmark.

| Method | Memory MB | Reduction vs FP16 | Shape | Source |
|---|---:|---:|---|---|
| FP16 KV cache | 83886.080000 | 1.00x | linear | standard transformer KV-cache model |
| TurboQuant 3.5-bit | 18350.080000 | 4.57x | linear compressed | arXiv:2504.19874 |
| KIVI 2-bit | 10485.760000 | 8.00x | linear compressed | arXiv:2402.02750 |
| PyramidKV 12% | 10066.329600 | 8.33x | linear retained | arXiv:2406.02069 |
| Catalyst Brain HKVC | 0.016384 | 5120000.00x | fixed state | public catalyst-brain SDK |

## HDC Correctness

- Bind/unbind perfect trials: `100/100`
- Perfect percentage: `100.0%`
- Minimum resonance: `1.0`

| Chain depth | Resonance |
|---:|---:|
| 1 | 1.000000 |
| 2 | 1.000000 |
| 5 | 1.000000 |
| 10 | 1.000000 |

## Rain State Transfer

| Dimension | JSON bytes | Rain binary bytes | Rain header bytes | Header under 8 KB | Roundtrip median us |
|---:|---:|---:|---:|---:|---:|
| 1024 | 4741 | 545 | 728 | `True` | 19.6045 |
| 4096 | 18534 | 1300 | 1736 | `True` | 62.9580 |
| 10000 | 45127 | 2764 | 3688 | `True` | 141.6460 |

## Memory Model

This is an explicit model, not process RSS: standard FP16 KV cache is computed as tokens * 40 layers * 4096 hidden * 2 K/V tensors * 2 bytes. Catalyst values are fixed-size: the uncompressed 4096-dim world vector and the measured compressed Rain header from the SDK.

| Tokens | Standard KV cache MB | Catalyst world vector MB | Rain header MB | World-vector reduction | Rain-header reduction |
|---:|---:|---:|---:|---:|---:|
| 1000 | 655.3600 | 0.016384 | 0.001768 | 40000.00x | 370678.73x |
| 5000 | 3276.8000 | 0.016384 | 0.001768 | 200000.00x | 1853393.67x |
| 10000 | 6553.6000 | 0.016384 | 0.001768 | 400000.00x | 3706787.33x |
| 50000 | 32768.0000 | 0.016384 | 0.001768 | 2000000.00x | 18533936.65x |
| 128000 | 83886.0800 | 0.016384 | 0.001768 | 5120000.00x | 47446877.83x |
