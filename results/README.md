# Catalyst Brain Benchmark Results

- SDK version: `1.3.2`
- Mode: `quick`
- Platform: `macOS-26.4-arm64-arm-64bit-Mach-O`
- Python: `3.13.13`
- Started: `2026-05-10T23:50:00.973460+00:00`

## Install Smoke

| Check | Value |
|---|---:|
| Distribution version | `1.3.2` |
| Module version | `1.3.2` |
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

## Deferred Output Savings

| Stdout bytes | Full result bytes | Compact status bytes | Saved | Saved tokens |
|---:|---:|---:|---:|---:|
| 1000 | 1281 | 301 | 76.50% | 245 |
| 10000 | 11032 | 302 | 97.26% | 2682 |
| 100000 | 108533 | 303 | 99.72% | 27058 |

## HKVC Query Scaling

| Entries | Median query us | p95 query us | Value OK | Confidence |
|---:|---:|---:|---:|---:|
| 100 | 11.5410 | 11.9580 | `True` | 0.0117 |
| 1000 | 12.0410 | 12.2090 | `True` | 0.0002 |
| 5000 | 12.0420 | 12.1670 | `True` | 0.0001 |

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

## Memory Model

This is an explicit model, not process RSS: standard FP16 KV cache is computed as tokens * 40 layers * 4096 hidden * 2 K/V tensors * 2 bytes. Catalyst values are fixed-size: the uncompressed 4096-dim world vector and the measured compressed Rain header from the SDK.

| Tokens | Standard KV cache MB | Catalyst world vector MB | Rain header MB | World-vector reduction | Rain-header reduction |
|---:|---:|---:|---:|---:|---:|
| 1000 | 655.3600 | 0.016384 | 0.001764 | 40000.00x | 371519.27x |
| 5000 | 3276.8000 | 0.016384 | 0.001764 | 200000.00x | 1857596.37x |
| 10000 | 6553.6000 | 0.016384 | 0.001764 | 400000.00x | 3715192.74x |
| 50000 | 32768.0000 | 0.016384 | 0.001764 | 2000000.00x | 18575963.72x |
