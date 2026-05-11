from __future__ import annotations

import csv
import importlib.metadata as metadata
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import catalyst_brain
import catalyst_hdc as hdc
from catalyst_brain import CatalystTokenKernel, ToolSpec


SDK_VERSION = "1.3.2"
KV_MODEL_LAYERS = 40
KV_MODEL_HIDDEN_SIZE = 4096
KV_MODEL_KV_TENSORS = 2
KV_MODEL_FP16_BYTES = 2
CATALYST_STATE_DIM = 4096


KV_CACHE_METHODS = (
    {
        "method": "FP16 KV cache",
        "family": "baseline",
        "shape": "linear",
        "bytes_multiplier": 1.0,
        "source": "standard transformer KV-cache model",
        "assumption": "tokens * layers * hidden * 2 K/V tensors * 2 FP16 bytes",
    },
    {
        "method": "TurboQuant 3.5-bit",
        "family": "quantization",
        "shape": "linear compressed",
        "bytes_multiplier": 3.5 / 16.0,
        "source": "arXiv:2504.19874",
        "assumption": "3.5 bits per channel quality-neutral KV-cache point",
    },
    {
        "method": "KIVI 2-bit",
        "family": "quantization",
        "shape": "linear compressed",
        "bytes_multiplier": 2.0 / 16.0,
        "source": "arXiv:2402.02750",
        "assumption": "KV-only 2-bit quantization model versus FP16 KV cache",
    },
    {
        "method": "PyramidKV 12%",
        "family": "retention",
        "shape": "linear retained",
        "bytes_multiplier": 0.12,
        "source": "arXiv:2406.02069",
        "assumption": "12% retained KV cache setting reported against full KV cache",
    },
)


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _token_estimate(byte_count: int) -> int:
    return max(1, (byte_count + 3) // 4)


def _percent_saved(compact_bytes: int, full_bytes: int) -> float:
    if full_bytes <= 0:
        return 0.0
    return round(max(0.0, 100.0 * (1.0 - compact_bytes / full_bytes)), 4)


def _measure_us(func: Callable[[], Any], *, repeats: int, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        func()
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        func()
        timings.append((time.perf_counter_ns() - start) / 1000.0)
    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "median_us": round(statistics.median(ordered), 4),
        "mean_us": round(statistics.fmean(ordered), 4),
        "p95_us": round(ordered[p95_index], 4),
        "min_us": round(ordered[0], 4),
        "max_us": round(ordered[-1], 4),
    }


def _make_tool_spec(index: int) -> ToolSpec:
    verbs = ("read", "write", "search", "execute", "summarize", "inspect", "patch")
    domain = ("repo", "browser", "shell", "docs", "tests", "cloud", "billing")[index % 7]
    verb = verbs[index % len(verbs)]
    properties = {
        f"field_{j}": {
            "type": "string" if j % 3 else "integer",
            "description": (
                f"Structured parameter {j} for tool {index}. This intentionally "
                "resembles verbose MCP/OpenAPI schemas that should not be sent "
                "on every agent turn."
            ),
        }
        for j in range(18)
    }
    return ToolSpec(
        name=f"{domain}.{verb}_{index}",
        description=(
            f"{verb.title()} capability for {domain} workflows. The descriptor "
            "contains enough detail to rank and execute the tool, but agents "
            "should request the full schema only after discovery selects it."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": ["field_0", "field_1"],
        },
        tags=(domain, verb, "agent-tool", "progressive-discovery"),
    )


def benchmark_install_smoke() -> dict[str, Any]:
    dim = 128
    a = hdc.hv_hash_string("alpha", dim)
    b = hdc.hv_hash_string("beta", dim)
    bound = hdc.hdc_bind(a, b)
    swarm = hdc.PyHoloSwarm(dim)
    swarm.add_agent("planner", a, "policy", b, "skill", a)
    return {
        "distribution_version": metadata.version("catalyst-brain"),
        "module_version": catalyst_brain.__version__,
        "version_ok": catalyst_brain.__version__ == SDK_VERSION,
        "bind_unbind_resonance": round(hdc.resonance(hdc.hdc_unbind(bound, a), b), 6),
        "holo_swarm_property_len": len(swarm.swarm_vector),
        "holo_swarm_method_len": len(swarm.get_swarm_vector()),
    }


def benchmark_token_discovery(*, mode: str) -> list[dict[str, Any]]:
    tool_counts = [10, 50, 100] if mode == "quick" else [10, 50, 100, 250, 500]
    rows: list[dict[str, Any]] = []
    for count in tool_counts:
        kernel = CatalystTokenKernel(dim=1024)
        specs = [_make_tool_spec(i) for i in range(count)]
        for spec in specs:
            kernel.register_tool(spec)

        full_manifest = [spec.full_descriptor() for spec in specs]
        full_manifest_bytes = _json_bytes(full_manifest)
        compact_page = kernel.list_tools(limit=min(20, count))
        compact_discovery = kernel.discover("execute python code in sandbox later", limit=5)
        expanded_discovery = kernel.discover(
            "execute python code in sandbox later",
            limit=1,
            include_schema=True,
        )
        compact_discovery_bytes = _json_bytes(compact_discovery)
        expanded_discovery_bytes = _json_bytes(expanded_discovery)

        rows.append(
            {
                "tool_count": count,
                "full_manifest_bytes": full_manifest_bytes,
                "compact_page_bytes": compact_page.compact_bytes,
                "compact_page_saved_pct": _percent_saved(
                    compact_page.compact_bytes,
                    full_manifest_bytes,
                ),
                "compact_discovery_bytes": compact_discovery_bytes,
                "compact_discovery_saved_pct": _percent_saved(
                    compact_discovery_bytes,
                    full_manifest_bytes,
                ),
                "expanded_one_schema_bytes": expanded_discovery_bytes,
                "expanded_one_schema_saved_pct": _percent_saved(
                    expanded_discovery_bytes,
                    full_manifest_bytes,
                ),
                "compact_page_saved_tokens": _token_estimate(full_manifest_bytes)
                - _token_estimate(compact_page.compact_bytes),
            }
        )
    return rows


def benchmark_deferred_outputs(*, mode: str) -> list[dict[str, Any]]:
    output_sizes = [1_000, 10_000, 100_000] if mode == "quick" else [1_000, 10_000, 100_000, 500_000]
    rows: list[dict[str, Any]] = []
    kernel = CatalystTokenKernel(dim=1024)
    for size in output_sizes:
        stdout = ("result-line\n" * ((size // 12) + 1))[:size]
        task = kernel.create_code_execution_task(
            code="print('large output produced by an external tool')",
            stdout=stdout,
            stderr="",
            status="completed",
            metadata={"benchmark": "deferred-output", "stdout_bytes": size},
        )
        full_result = kernel.fetch_task_result(task["task_id"])
        compact_bytes = _json_bytes(task)
        full_bytes = _json_bytes(full_result)
        rows.append(
            {
                "stdout_bytes": size,
                "full_result_bytes": full_bytes,
                "compact_status_bytes": compact_bytes,
                "saved_pct": _percent_saved(compact_bytes, full_bytes),
                "saved_tokens": _token_estimate(full_bytes) - _token_estimate(compact_bytes),
                "status": task["status"],
                "output_available": task["output_available"],
            }
        )
    return rows


def benchmark_hdc_primitives(*, mode: str) -> list[dict[str, Any]]:
    dims = [128, 1024, 4096] if mode == "quick" else [128, 1024, 4096, 10000]
    repeats = 50 if mode == "quick" else 120
    rows: list[dict[str, Any]] = []
    for dim in dims:
        a = hdc.rand_bipolar(dim)
        b = hdc.rand_bipolar(dim)
        bound = hdc.hdc_bind(a, b)
        ops: list[tuple[str, Callable[[], Any]]] = [
            ("bind", lambda a=a, b=b: hdc.hdc_bind(a, b)),
            ("unbind", lambda bound=bound, a=a: hdc.hdc_unbind(bound, a)),
            ("bundle", lambda a=a, b=b: hdc.hdc_bundle(a, b)),
            ("resonance", lambda a=a, b=b: hdc.resonance(a, b)),
        ]
        for name, func in ops:
            measured = _measure_us(func, repeats=repeats)
            rows.append(
                {
                    "operation": name,
                    "dimension": dim,
                    **measured,
                    "elements_per_second_median": round(dim / (measured["median_us"] / 1_000_000.0), 2),
                }
            )
    return rows


def benchmark_bind_unbind_correctness(*, mode: str) -> dict[str, Any]:
    trials = 100 if mode == "quick" else 1000
    depths = [1, 2, 5, 10] if mode == "quick" else [1, 2, 5, 10, 25, 50, 100]
    dim = 1024
    perfect = 0
    min_resonance = 1.0
    for i in range(trials):
        key = hdc.hv_hash_string(f"key-{i}", dim)
        value = hdc.hv_hash_string(f"value-{i}", dim)
        recovered = hdc.hdc_unbind(hdc.hdc_bind(key, value), key)
        resonance = hdc.resonance(recovered, value)
        min_resonance = min(min_resonance, resonance)
        if resonance > 0.999999:
            perfect += 1

    chained: list[dict[str, Any]] = []
    for depth in depths:
        value = hdc.hv_hash_string(f"chain-value-{depth}", dim)
        keys = [hdc.hv_hash_string(f"chain-key-{depth}-{j}", dim) for j in range(depth)]
        compound = value
        for key in keys:
            compound = hdc.hdc_bind(compound, key)
        recovered = compound
        for key in reversed(keys):
            recovered = hdc.hdc_unbind(recovered, key)
        chained.append(
            {
                "depth": depth,
                "resonance": round(hdc.resonance(recovered, value), 6),
            }
        )

    return {
        "dimension": dim,
        "trials": trials,
        "perfect_trials": perfect,
        "perfect_pct": round(100.0 * perfect / trials, 4),
        "min_resonance": round(min_resonance, 6),
        "chained_composition": chained,
    }


def benchmark_hkvc_scaling(*, mode: str) -> list[dict[str, Any]]:
    counts = [100, 1000, 5000] if mode == "quick" else [100, 1000, 5000, 10000, 25000]
    repeats = 80 if mode == "quick" else 160
    dim = 1024
    rows: list[dict[str, Any]] = []
    for count in counts:
        kv = hdc.PyHKVC(dim)
        start = time.perf_counter()
        for i in range(count):
            kv.store(f"key-{i}", f"value-{i}", i)
        store_seconds = time.perf_counter() - start
        target = count // 2

        def query() -> tuple[str, float]:
            return kv.query(f"key-{target}")

        measured = _measure_us(query, repeats=repeats)
        value, confidence = query()
        rows.append(
            {
                "entries": count,
                "dimension": dim,
                "store_seconds": round(store_seconds, 6),
                "store_per_second": round(count / store_seconds, 2) if store_seconds > 0 else 0.0,
                **measured,
                "value_ok": value == f"value-{target}",
                "confidence": round(confidence, 6),
            }
        )
    return rows


def _fp16_kv_cache_bytes(token_count: int) -> int:
    return (
        token_count
        * KV_MODEL_LAYERS
        * KV_MODEL_HIDDEN_SIZE
        * KV_MODEL_KV_TENSORS
        * KV_MODEL_FP16_BYTES
    )


def benchmark_memory_model() -> list[dict[str, Any]]:
    tokens = [1000, 5000, 10000, 50000, 128000]
    kernel = CatalystTokenKernel(dim=CATALYST_STATE_DIM)
    catalyst_rain_header_bytes = kernel.export_rain_snapshot(agent_id="memory-model")[
        "compact_state_bytes"
    ]
    catalyst_world_vector_bytes = CATALYST_STATE_DIM * 4
    rows: list[dict[str, Any]] = []
    for token_count in tokens:
        standard_bytes = _fp16_kv_cache_bytes(token_count)
        rows.append(
            {
                "tokens": token_count,
                "standard_kv_cache_mb": round(standard_bytes / 1_000_000.0, 4),
                "catalyst_world_vector_mb": round(catalyst_world_vector_bytes / 1_000_000.0, 6),
                "catalyst_rain_header_mb": round(catalyst_rain_header_bytes / 1_000_000.0, 6),
                "world_vector_reduction_x": round(standard_bytes / catalyst_world_vector_bytes, 2),
                "rain_header_reduction_x": round(standard_bytes / catalyst_rain_header_bytes, 2),
                "assumption": "40 layers, hidden size 4096, FP16 K+V tensors",
            }
        )
    return rows


def benchmark_kv_cache_comparison() -> list[dict[str, Any]]:
    token_counts = [1000, 4000, 16000, 32000, 64000, 128000]
    catalyst_state_bytes = CATALYST_STATE_DIM * 4
    rows: list[dict[str, Any]] = []
    for token_count in token_counts:
        baseline_bytes = _fp16_kv_cache_bytes(token_count)
        for method in KV_CACHE_METHODS:
            modeled_bytes = int(baseline_bytes * method["bytes_multiplier"])
            rows.append(
                {
                    "tokens": token_count,
                    "method": method["method"],
                    "family": method["family"],
                    "shape": method["shape"],
                    "memory_mb": round(modeled_bytes / 1_000_000.0, 6),
                    "relative_to_fp16_pct": round(100.0 * method["bytes_multiplier"], 4),
                    "reduction_vs_fp16_x": round(1.0 / method["bytes_multiplier"], 4),
                    "source": method["source"],
                    "assumption": method["assumption"],
                }
            )
        rows.append(
            {
                "tokens": token_count,
                "method": "Catalyst Brain HKVC",
                "family": "holographic fixed state",
                "shape": "fixed state",
                "memory_mb": round(catalyst_state_bytes / 1_000_000.0, 6),
                "relative_to_fp16_pct": round(100.0 * catalyst_state_bytes / baseline_bytes, 8),
                "reduction_vs_fp16_x": round(baseline_bytes / catalyst_state_bytes, 4),
                "source": "public catalyst-brain SDK",
                "assumption": "4096-dim public SDK world-vector state, not SDK source internals",
            }
        )
    return rows


def summarize_benchmark_claims(results: dict[str, Any]) -> dict[str, Any]:
    token_best = max(
        row["compact_discovery_saved_pct"] for row in results["token_discovery"]
    )
    deferred_best = max(row["saved_pct"] for row in results["deferred_outputs"])
    largest_kv = max(
        (
            row
            for row in results["kv_cache_comparison"]
            if row["method"] == "Catalyst Brain HKVC"
        ),
        key=lambda row: row["tokens"],
    )
    hkvc_rows = results["hkvc_scaling"]
    latency_span_us = hkvc_rows[-1]["median_us"] - hkvc_rows[0]["median_us"]
    return {
        "best_compact_discovery_saved_pct": round(token_best, 4),
        "best_deferred_output_saved_pct": round(deferred_best, 4),
        "largest_kv_context_tokens": largest_kv["tokens"],
        "largest_kv_catalyst_memory_mb": largest_kv["memory_mb"],
        "largest_kv_reduction_vs_fp16_x": largest_kv["reduction_vs_fp16_x"],
        "hkvc_first_entries": hkvc_rows[0]["entries"],
        "hkvc_last_entries": hkvc_rows[-1]["entries"],
        "hkvc_median_latency_span_us": round(latency_span_us, 4),
        "claim_boundary": (
            "Benchmarks use public catalyst-brain APIs and explicit memory models; "
            "they are not SDK source disclosures or provider billing statements."
        ),
    }


def run_suite(*, mode: str = "quick") -> dict[str, Any]:
    if mode not in {"quick", "full"}:
        raise ValueError("mode must be 'quick' or 'full'")
    started = datetime.now(timezone.utc)
    results = {
        "metadata": {
            "suite_version": "0.2.0",
            "mode": mode,
            "started_at": started.isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "catalyst_brain_version": catalyst_brain.__version__,
            "catalyst_brain_distribution": metadata.version("catalyst-brain"),
            "sdk_version_expected": SDK_VERSION,
        },
        "install_smoke": benchmark_install_smoke(),
        "token_discovery": benchmark_token_discovery(mode=mode),
        "deferred_outputs": benchmark_deferred_outputs(mode=mode),
        "hdc_primitives": benchmark_hdc_primitives(mode=mode),
        "bind_unbind_correctness": benchmark_bind_unbind_correctness(mode=mode),
        "hkvc_scaling": benchmark_hkvc_scaling(mode=mode),
        "memory_model": benchmark_memory_model(),
        "kv_cache_comparison": benchmark_kv_cache_comparison(),
    }
    results["claim_summary"] = summarize_benchmark_claims(results)
    results["metadata"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    return results


def write_outputs(results: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    for key, value in results.items():
        if isinstance(value, list):
            _write_csv(out_dir / f"{key}.csv", value)
        elif isinstance(value, dict) and key not in {"metadata"}:
            nested = value.get("chained_composition")
            if isinstance(nested, list):
                _write_csv(out_dir / f"{key}_chained_composition.csv", nested)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown_report(results: dict[str, Any]) -> str:
    meta = results["metadata"]
    smoke = results["install_smoke"]
    correctness = results["bind_unbind_correctness"]
    summary = results["claim_summary"]
    lines = [
        "# Catalyst Brain Benchmark Results",
        "",
        f"- SDK version: `{meta['catalyst_brain_version']}`",
        f"- Mode: `{meta['mode']}`",
        f"- Platform: `{meta['platform']}`",
        f"- Python: `{meta['python']}`",
        f"- Started: `{meta['started_at']}`",
        "",
        "## Headline Checks",
        "",
        "| Check | Result |",
        "|---|---:|",
        (
            "| Best progressive discovery context saved | "
            f"{summary['best_compact_discovery_saved_pct']:.2f}% |"
        ),
        (
            "| Best deferred-output context saved | "
            f"{summary['best_deferred_output_saved_pct']:.2f}% |"
        ),
        (
            "| Largest modeled KV context | "
            f"{summary['largest_kv_context_tokens']} tokens |"
        ),
        (
            "| Catalyst HKVC memory at largest context | "
            f"{summary['largest_kv_catalyst_memory_mb']:.6f} MB |"
        ),
        (
            "| Catalyst HKVC reduction versus FP16 model | "
            f"{summary['largest_kv_reduction_vs_fp16_x']:.2f}x |"
        ),
        (
            "| HKVC median latency span across quick scaling run | "
            f"{summary['hkvc_median_latency_span_us']:.4f} us |"
        ),
        "",
        "## Install Smoke",
        "",
        "| Check | Value |",
        "|---|---:|",
        f"| Distribution version | `{smoke['distribution_version']}` |",
        f"| Module version | `{smoke['module_version']}` |",
        f"| Version OK | `{smoke['version_ok']}` |",
        f"| Bind/unbind resonance | {smoke['bind_unbind_resonance']} |",
        f"| HoloSwarm property length | {smoke['holo_swarm_property_len']} |",
        f"| HoloSwarm method length | {smoke['holo_swarm_method_len']} |",
        "",
        "## Token Discovery Savings",
        "",
        "| Tools | Full manifest bytes | Compact page bytes | Compact saved | Discovery saved | One-schema saved |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results["token_discovery"]:
        lines.append(
            "| {tool_count} | {full_manifest_bytes} | {compact_page_bytes} | "
            "{compact_page_saved_pct:.2f}% | {compact_discovery_saved_pct:.2f}% | "
            "{expanded_one_schema_saved_pct:.2f}% |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Deferred Output Savings",
            "",
            "| Stdout bytes | Full result bytes | Compact status bytes | Saved | Saved tokens |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["deferred_outputs"]:
        lines.append(
            "| {stdout_bytes} | {full_result_bytes} | {compact_status_bytes} | "
            "{saved_pct:.2f}% | {saved_tokens} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## HKVC Query Scaling",
            "",
            "| Entries | Median query us | p95 query us | Value OK | Confidence |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["hkvc_scaling"]:
        lines.append(
            "| {entries} | {median_us:.4f} | {p95_us:.4f} | `{value_ok}` | {confidence:.4f} |".format(
                **row
            )
        )
    largest_tokens = max(row["tokens"] for row in results["kv_cache_comparison"])
    largest_rows = [
        row for row in results["kv_cache_comparison"] if row["tokens"] == largest_tokens
    ]
    lines.extend(
        [
            "",
            "## KV-Cache Competitor Model",
            "",
            "This table is a stated memory model at the largest context in the run. "
            "It compares published compression or retention targets against the "
            "fixed Catalyst Brain public SDK state size. It is not an end-to-end "
            "quality benchmark.",
            "",
            "| Method | Memory MB | Reduction vs FP16 | Shape | Source |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in largest_rows:
        lines.append(
            "| {method} | {memory_mb:.6f} | {reduction_vs_fp16_x:.2f}x | "
            "{shape} | {source} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## HDC Correctness",
            "",
            f"- Bind/unbind perfect trials: `{correctness['perfect_trials']}/{correctness['trials']}`",
            f"- Perfect percentage: `{correctness['perfect_pct']}%`",
            f"- Minimum resonance: `{correctness['min_resonance']}`",
            "",
            "| Chain depth | Resonance |",
            "|---:|---:|",
        ]
    )
    for row in correctness["chained_composition"]:
        lines.append("| {depth} | {resonance:.6f} |".format(**row))
    lines.extend(
        [
            "",
            "## Memory Model",
            "",
            "This is an explicit model, not process RSS: standard FP16 KV cache is "
            "computed as tokens * 40 layers * 4096 hidden * 2 K/V tensors * 2 bytes. "
            "Catalyst values are fixed-size: the uncompressed 4096-dim world vector "
            "and the measured compressed Rain header from the SDK.",
            "",
            "| Tokens | Standard KV cache MB | Catalyst world vector MB | Rain header MB | World-vector reduction | Rain-header reduction |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["memory_model"]:
        lines.append(
            "| {tokens} | {standard_kv_cache_mb:.4f} | {catalyst_world_vector_mb:.6f} | "
            "{catalyst_rain_header_mb:.6f} | {world_vector_reduction_x:.2f}x | "
            "{rain_header_reduction_x:.2f}x |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)
