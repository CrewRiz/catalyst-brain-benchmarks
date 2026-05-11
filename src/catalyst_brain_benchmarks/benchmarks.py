from __future__ import annotations

import csv
import importlib.metadata as metadata
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import catalyst_brain
import catalyst_hdc as hdc
from catalyst_brain import (
    CatalystTokenKernel,
    RainPayload,
    ToolSpec,
    rain_dumps,
    rain_loads,
    rain_to_header,
)


SDK_VERSION = "1.3.3"
KV_MODEL_LAYERS = 40
KV_MODEL_HIDDEN_SIZE = 4096
KV_MODEL_KV_TENSORS = 2
KV_MODEL_FP16_BYTES = 2
CATALYST_STATE_DIM = 4096
COMMON_HEADER_LIMIT_BYTES = 8_192


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


def _raw_dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / len(a)


def _flip_deterministic(vector: list[float], *, noise_pct: float, seed: int) -> list[float]:
    if noise_pct <= 0.0:
        return list(vector)
    period = 10_000
    threshold = int(noise_pct * period)
    return [
        -value
        if ((index * 1_103 + seed * 9_176 + 41) % period) < threshold
        else value
        for index, value in enumerate(vector)
    ]


def _classical_softmax_attention(
    query: list[float],
    keys: list[list[float]],
    values: list[list[float]],
    *,
    beta: float = 8.0,
) -> list[float]:
    scores = [_raw_dot(query, key) for key in keys]
    offset = max(scores)
    weights = [math.exp(beta * (score - offset)) for score in scores]
    total = sum(weights)
    if total <= 0.0:
        return [0.0] * len(query)
    out = [0.0] * len(query)
    for weight, value in zip(weights, values):
        scaled = weight / total
        for index, item in enumerate(value):
            out[index] += scaled * item
    return out


def _predict_value_index(output: list[float], values: list[list[float]]) -> tuple[int, float]:
    scores = [_raw_dot(output, value) for value in values]
    best = max(range(len(scores)), key=lambda index: scores[index])
    return best, scores[best]


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


def _make_selection_tool_specs() -> list[ToolSpec]:
    """Small MCP-style catalog with unambiguous task intents."""
    cases = (
        (
            "sandbox.execute_python",
            "Run Python code safely in a sandbox and return a deferred task status.",
            ("python", "execute", "sandbox", "code", "deferred"),
            {"code": {"type": "string"}, "timeout_s": {"type": "number"}},
        ),
        (
            "repo.search_text",
            "Find text across repository files using a search query.",
            ("repo", "search", "text", "files"),
            {"query": {"type": "string"}, "glob": {"type": "string"}},
        ),
        (
            "filesystem.read_file",
            "Read a local source file from a workspace path.",
            ("filesystem", "read", "file", "source"),
            {"path": {"type": "string"}},
        ),
        (
            "stripe.create_checkout",
            "Create a Stripe billing checkout session for paid SDK access.",
            ("stripe", "billing", "checkout", "payment"),
            {"plan": {"type": "string"}, "customer_email": {"type": "string"}},
        ),
        (
            "cloudflare.deploy_worker",
            "Deploy a Cloudflare Worker for serverless edge memory.",
            ("cloudflare", "deploy", "worker", "edge"),
            {"script": {"type": "string"}, "account_id": {"type": "string"}},
        ),
        (
            "browser.fetch_page",
            "Fetch browser page content from a web URL for inspection.",
            ("browser", "web", "fetch", "page"),
            {"url": {"type": "string"}},
        ),
        (
            "github.create_pull_request",
            "Open a GitHub pull request after code changes are ready.",
            ("github", "pull", "request", "pr"),
            {"branch": {"type": "string"}, "title": {"type": "string"}},
        ),
        (
            "docs.summarize_document",
            "Summarize a long document into concise notes.",
            ("docs", "summarize", "document", "notes"),
            {"path": {"type": "string"}, "max_words": {"type": "integer"}},
        ),
    )
    specs: list[ToolSpec] = []
    for name, description, tags, properties in cases:
        specs.append(
            ToolSpec(
                name=name,
                description=description,
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": properties,
                    "required": list(properties)[:1],
                },
                tags=tags,
            )
        )
    return specs


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


def benchmark_tool_selection_accuracy(*, mode: str) -> list[dict[str, Any]]:
    """Check that compact discovery still selects the intended tool."""
    catalog_sizes = [8, 50, 100] if mode == "quick" else [8, 50, 100, 250, 500]
    queries = (
        ("run python code safely in a sandbox", "sandbox.execute_python"),
        ("search repository files for text", "repo.search_text"),
        ("read a local source file", "filesystem.read_file"),
        ("create a stripe billing checkout", "stripe.create_checkout"),
        ("deploy a cloudflare worker", "cloudflare.deploy_worker"),
        ("fetch a browser web page", "browser.fetch_page"),
        ("open a github pull request", "github.create_pull_request"),
        ("summarize a long document", "docs.summarize_document"),
    )
    anchor_specs = _make_selection_tool_specs()
    rows: list[dict[str, Any]] = []
    for catalog_size in catalog_sizes:
        kernel = CatalystTokenKernel(dim=1024)
        specs = list(anchor_specs)
        while len(specs) < catalog_size:
            specs.append(_make_tool_spec(len(specs)))
        for spec in specs:
            kernel.register_tool(spec)
        full_manifest_bytes = _json_bytes([spec.full_descriptor() for spec in specs])
        for query, expected in queries:
            compact_results = kernel.discover(query, limit=3, include_schema=False)
            expanded_result = kernel.discover(query, limit=1, include_schema=True)
            names = [item["name"] for item in compact_results]
            rank = names.index(expected) + 1 if expected in names else 0
            rows.append(
                {
                    "tool_count": catalog_size,
                    "query": query,
                    "expected": expected,
                    "top1": names[0] if names else "",
                    "top1_ok": bool(names and names[0] == expected),
                    "top3_ok": expected in names,
                    "rank": rank,
                    "compact_results_bytes": _json_bytes(compact_results),
                    "expanded_one_schema_bytes": _json_bytes(expanded_result),
                    "full_manifest_bytes": full_manifest_bytes,
                    "compact_saved_pct": _percent_saved(
                        _json_bytes(compact_results),
                        full_manifest_bytes,
                    ),
                    "expanded_saved_pct": _percent_saved(
                        _json_bytes(expanded_result),
                        full_manifest_bytes,
                    ),
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


def benchmark_quantum_attention_heads(*, mode: str) -> list[dict[str, Any]]:
    """Benchmark quantum-inspired attention routing through the public wheel."""
    configs = (
        [(256, 4, 0.00), (512, 16, 0.05), (1024, 64, 0.10)]
        if mode == "quick"
        else [
            (256, 4, 0.00),
            (512, 16, 0.05),
            (1024, 64, 0.10),
            (2048, 128, 0.15),
        ]
    )
    trials = 12 if mode == "quick" else 32
    repeats = 60 if mode == "quick" else 140
    rows: list[dict[str, Any]] = []

    for dim, key_count, noise_pct in configs:
        keys = [hdc.hv_hash_string(f"qattn-key-{dim}-{key_count}-{i}", dim) for i in range(key_count)]
        values = [
            hdc.hv_hash_string(f"qattn-value-{dim}-{key_count}-{i}", dim)
            for i in range(key_count)
        ]
        nqubits = max(4, min(40, (key_count - 1).bit_length()))
        latency_target = key_count // 2
        latency_query = _flip_deterministic(
            keys[latency_target],
            noise_pct=noise_pct,
            seed=latency_target,
        )

        quantum_head = hdc.PyQuantumAttentionHead(dim, nqubits)

        def quantum_compute() -> list[float]:
            return quantum_head.compute(latency_query, keys, values)

        def classical_compute() -> list[float]:
            return _classical_softmax_attention(latency_query, keys, values)

        quantum_latency = _measure_us(quantum_compute, repeats=repeats)
        classical_latency = _measure_us(classical_compute, repeats=repeats)

        quantum_correct = 0
        classical_correct = 0
        quantum_confidences: list[float] = []
        classical_confidences: list[float] = []
        for trial in range(trials):
            target = (trial * 7 + key_count // 3) % key_count
            query = _flip_deterministic(
                keys[target],
                noise_pct=noise_pct,
                seed=trial + dim,
            )
            q_out = quantum_head.compute(query, keys, values)
            c_out = _classical_softmax_attention(query, keys, values)
            q_pred, q_conf = _predict_value_index(q_out, values)
            c_pred, c_conf = _predict_value_index(c_out, values)
            quantum_correct += int(q_pred == target)
            classical_correct += int(c_pred == target)
            quantum_confidences.append(q_conf)
            classical_confidences.append(c_conf)

        rows.append(
            {
                "dimension": dim,
                "key_count": key_count,
                "noise_pct": round(noise_pct * 100.0, 2),
                "method": "Catalyst PyQuantumAttentionHead",
                "nqubits": nqubits,
                "trials": trials,
                "top1_accuracy_pct": round(100.0 * quantum_correct / trials, 4),
                "mean_target_confidence": round(statistics.fmean(quantum_confidences), 6),
                "median_us": quantum_latency["median_us"],
                "p95_us": quantum_latency["p95_us"],
                "latency_vs_classical_x": round(
                    classical_latency["median_us"] / quantum_latency["median_us"],
                    4,
                )
                if quantum_latency["median_us"] > 0
                else 0.0,
                "baseline": "classical cosine softmax reference",
            }
        )
        rows.append(
            {
                "dimension": dim,
                "key_count": key_count,
                "noise_pct": round(noise_pct * 100.0, 2),
                "method": "Classical cosine softmax",
                "nqubits": 0,
                "trials": trials,
                "top1_accuracy_pct": round(100.0 * classical_correct / trials, 4),
                "mean_target_confidence": round(statistics.fmean(classical_confidences), 6),
                "median_us": classical_latency["median_us"],
                "p95_us": classical_latency["p95_us"],
                "latency_vs_classical_x": 1.0,
                "baseline": "pure Python reference implementation",
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


def benchmark_hkvc_path_breakdown(*, mode: str) -> list[dict[str, Any]]:
    """Separate exact-key indexed hits from the documented fallback path."""
    counts = [100, 1000, 5000] if mode == "quick" else [100, 1000, 5000, 10000, 25000]
    repeats = 80 if mode == "quick" else 160
    dim = 1024
    rows: list[dict[str, Any]] = []
    for count in counts:
        kv = hdc.PyHKVC(dim)
        for i in range(count):
            kv.store(f"key-{i}", f"value-{i}", i)
        target = count // 2

        def exact_query() -> tuple[str, float]:
            return kv.query(f"key-{target}")

        def fallback_query() -> tuple[str, float]:
            return kv.query(f"missing-key-{count}")

        exact_measured = _measure_us(exact_query, repeats=repeats)
        fallback_measured = _measure_us(fallback_query, repeats=repeats)
        exact_value, exact_score = exact_query()
        fallback_value, fallback_score = fallback_query()
        exact_median = exact_measured["median_us"]
        fallback_median = fallback_measured["median_us"]
        rows.append(
            {
                "entries": count,
                "dimension": dim,
                "exact_median_us": exact_median,
                "exact_p95_us": exact_measured["p95_us"],
                "exact_value_ok": exact_value == f"value-{target}",
                "exact_confidence": round(exact_score, 6),
                "miss_fallback_median_us": fallback_median,
                "miss_fallback_p95_us": fallback_measured["p95_us"],
                "miss_fallback_returned_value": bool(fallback_value),
                "miss_fallback_confidence": round(fallback_score, 6),
                "fallback_vs_exact_median_x": round(
                    fallback_median / exact_median,
                    4,
                )
                if exact_median > 0
                else 0.0,
            }
        )
    return rows


def benchmark_hkvc_recency_uniformity(*, mode: str) -> list[dict[str, Any]]:
    """Probe first/middle/last entries to detect recency-position bias."""
    counts = [1000, 5000] if mode == "quick" else [1000, 5000, 25000]
    repeats = 60 if mode == "quick" else 120
    dim = 1024
    rows: list[dict[str, Any]] = []
    for count in counts:
        kv = hdc.PyHKVC(dim)
        for i in range(count):
            kv.store(f"key-{i}", f"value-{i}", i)
        probes = (
            ("first", 0, 0.0),
            ("middle", count // 2, 0.5),
            ("last", count - 1, 1.0),
        )
        measured_rows: list[dict[str, Any]] = []
        for bucket, position, fraction in probes:
            key = f"key-{position}"

            def query(key: str = key) -> tuple[str, float]:
                return kv.query(key)

            measured = _measure_us(query, repeats=repeats)
            value, confidence = query()
            measured_rows.append(
                {
                    "entries": count,
                    "dimension": dim,
                    "bucket": bucket,
                    "position": position,
                    "position_fraction": fraction,
                    "median_us": measured["median_us"],
                    "p95_us": measured["p95_us"],
                    "value_ok": value == f"value-{position}",
                    "confidence": round(confidence, 6),
                    "position_score": round(kv.position_score(position), 6),
                }
            )
        confidence_values = [row["confidence"] for row in measured_rows]
        latency_values = [row["median_us"] for row in measured_rows]
        for row in measured_rows:
            row["confidence_spread"] = round(max(confidence_values) - min(confidence_values), 6)
            row["median_latency_spread_us"] = round(max(latency_values) - min(latency_values), 4)
            rows.append(row)
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


def benchmark_rain_state_transfer(*, mode: str) -> list[dict[str, Any]]:
    dims = [1024, 4096, 10000] if mode == "quick" else [1024, 4096, 10000, 20000]
    repeats = 40 if mode == "quick" else 100
    rows: list[dict[str, Any]] = []
    for dim in dims:
        world_vector = hdc.hv_hash_string(f"rain-state-{dim}", dim)
        payload = RainPayload(
            agent_id=f"bench-agent-{dim}",
            dim=dim,
            world_vector=world_vector,
            config={
                "mode": "stateless-agent-handoff",
                "tool_count": 50,
                "task_count": 10,
            },
        )
        rain_blob = rain_dumps(payload)
        rain_header = rain_to_header(payload)
        json_state = {
            "agent_id": payload.agent_id,
            "dim": payload.dim,
            "world_vector": world_vector,
            "config": payload.config,
        }
        json_bytes = _json_bytes(json_state)

        def roundtrip() -> Any:
            return rain_loads(rain_blob)

        measured = _measure_us(roundtrip, repeats=repeats)
        restored = roundtrip()
        rows.append(
            {
                "dimension": dim,
                "raw_world_vector_bytes": dim * 4,
                "json_state_bytes": json_bytes,
                "rain_binary_bytes": len(rain_blob),
                "rain_header_bytes": len(rain_header.encode("ascii")),
                "rain_vs_json_reduction_x": round(json_bytes / len(rain_blob), 4),
                "header_vs_json_reduction_x": round(
                    json_bytes / len(rain_header.encode("ascii")),
                    4,
                ),
                "header_under_8kb": len(rain_header.encode("ascii")) <= COMMON_HEADER_LIMIT_BYTES,
                "roundtrip_median_us": measured["median_us"],
                "roundtrip_p95_us": measured["p95_us"],
                "roundtrip_ok": restored.agent_id == payload.agent_id
                and restored.dim == payload.dim
                and len(restored.world_vector or []) == dim,
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
    selection_rows = results["tool_selection_accuracy"]
    top1_pct = 100.0 * sum(1 for row in selection_rows if row["top1_ok"]) / len(selection_rows)
    top3_pct = 100.0 * sum(1 for row in selection_rows if row["top3_ok"]) / len(selection_rows)
    path_rows = results["hkvc_path_breakdown"]
    largest_path = max(path_rows, key=lambda row: row["entries"])
    recency_rows = results["hkvc_recency_uniformity"]
    largest_recency_entries = max(row["entries"] for row in recency_rows)
    largest_recency_rows = [
        row for row in recency_rows if row["entries"] == largest_recency_entries
    ]
    recency_ok_pct = 100.0 * sum(
        1 for row in largest_recency_rows if row["value_ok"]
    ) / len(largest_recency_rows)
    largest_rain = max(results["rain_state_transfer"], key=lambda row: row["dimension"])
    quantum_rows = [
        row
        for row in results["quantum_attention_heads"]
        if row["method"] == "Catalyst PyQuantumAttentionHead"
    ]
    largest_quantum = max(quantum_rows, key=lambda row: row["key_count"])
    return {
        "best_compact_discovery_saved_pct": round(token_best, 4),
        "best_deferred_output_saved_pct": round(deferred_best, 4),
        "tool_selection_top1_pct": round(top1_pct, 4),
        "tool_selection_top3_pct": round(top3_pct, 4),
        "largest_kv_context_tokens": largest_kv["tokens"],
        "largest_kv_catalyst_memory_mb": largest_kv["memory_mb"],
        "largest_kv_reduction_vs_fp16_x": largest_kv["reduction_vs_fp16_x"],
        "hkvc_first_entries": hkvc_rows[0]["entries"],
        "hkvc_last_entries": hkvc_rows[-1]["entries"],
        "hkvc_median_latency_span_us": round(latency_span_us, 4),
        "hkvc_largest_exact_median_us": largest_path["exact_median_us"],
        "hkvc_largest_fallback_median_us": largest_path["miss_fallback_median_us"],
        "hkvc_largest_fallback_vs_exact_x": largest_path["fallback_vs_exact_median_x"],
        "hkvc_largest_recency_entries": largest_recency_entries,
        "hkvc_largest_recency_value_ok_pct": round(recency_ok_pct, 4),
        "rain_largest_dimension": largest_rain["dimension"],
        "rain_largest_header_bytes": largest_rain["rain_header_bytes"],
        "rain_largest_header_under_8kb": largest_rain["header_under_8kb"],
        "quantum_attention_largest_key_count": largest_quantum["key_count"],
        "quantum_attention_largest_accuracy_pct": largest_quantum["top1_accuracy_pct"],
        "quantum_attention_largest_median_us": largest_quantum["median_us"],
        "quantum_attention_largest_speedup_vs_reference_x": largest_quantum[
            "latency_vs_classical_x"
        ],
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
            "suite_version": "0.4.0",
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
        "tool_selection_accuracy": benchmark_tool_selection_accuracy(mode=mode),
        "deferred_outputs": benchmark_deferred_outputs(mode=mode),
        "hdc_primitives": benchmark_hdc_primitives(mode=mode),
        "quantum_attention_heads": benchmark_quantum_attention_heads(mode=mode),
        "bind_unbind_correctness": benchmark_bind_unbind_correctness(mode=mode),
        "hkvc_scaling": benchmark_hkvc_scaling(mode=mode),
        "hkvc_path_breakdown": benchmark_hkvc_path_breakdown(mode=mode),
        "hkvc_recency_uniformity": benchmark_hkvc_recency_uniformity(mode=mode),
        "rain_state_transfer": benchmark_rain_state_transfer(mode=mode),
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
            "| Tool discovery top-1 accuracy | "
            f"{summary['tool_selection_top1_pct']:.2f}% |"
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
        (
            "| HKVC largest hit vs miss-fallback latency | "
            f"{summary['hkvc_largest_exact_median_us']:.4f} us / "
            f"{summary['hkvc_largest_fallback_median_us']:.4f} us |"
        ),
        (
            "| Rain largest header under 8 KB | "
            f"`{summary['rain_largest_header_under_8kb']}` |"
        ),
        (
            "| Quantum attention largest-key top-1 accuracy | "
            f"{summary['quantum_attention_largest_accuracy_pct']:.2f}% |"
        ),
        (
            "| Quantum attention largest-key speedup vs reference | "
            f"{summary['quantum_attention_largest_speedup_vs_reference_x']:.2f}x |"
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
            "## Tool Selection Accuracy",
            "",
            "| Tools | Top-1 accuracy | Top-3 accuracy | Median compact saved |",
            "|---:|---:|---:|---:|",
        ]
    )
    for tool_count in sorted({row["tool_count"] for row in results["tool_selection_accuracy"]}):
        rows = [
            row
            for row in results["tool_selection_accuracy"]
            if row["tool_count"] == tool_count
        ]
        top1 = 100.0 * sum(1 for row in rows if row["top1_ok"]) / len(rows)
        top3 = 100.0 * sum(1 for row in rows if row["top3_ok"]) / len(rows)
        saved = statistics.median(row["compact_saved_pct"] for row in rows)
        lines.append(f"| {tool_count} | {top1:.2f}% | {top3:.2f}% | {saved:.2f}% |")
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
            "## Quantum-Inspired Attention Heads",
            "",
            "This benchmark measures the public `PyQuantumAttentionHead` routing "
            "behavior against a pure-Python cosine softmax reference. It does "
            "not claim physical quantum execution.",
            "",
            "| Dim | Keys | Noise | Method | Top-1 accuracy | Median us | p95 us | Speedup vs ref | Mean confidence |",
            "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["quantum_attention_heads"]:
        lines.append(
            "| {dimension} | {key_count} | {noise_pct:.2f}% | {method} | "
            "{top1_accuracy_pct:.2f}% | {median_us:.4f} | {p95_us:.4f} | "
            "{latency_vs_classical_x:.2f}x | {mean_target_confidence:.6f} |".format(**row)
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
    lines.extend(
        [
            "",
            "## HKVC Path Breakdown",
            "",
            "Exact-key hits use the indexed public API path. Missing-key fallback is "
            "reported separately because it performs approximate recall work.",
            "",
            "| Entries | Exact hit median us | Miss fallback median us | Fallback / exact | Exact value OK |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["hkvc_path_breakdown"]:
        lines.append(
            "| {entries} | {exact_median_us:.4f} | {miss_fallback_median_us:.4f} | "
            "{fallback_vs_exact_median_x:.2f}x | `{exact_value_ok}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "## HKVC Recency Position Probe",
            "",
            "| Entries | Bucket | Position | Median query us | Value OK | Confidence |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in results["hkvc_recency_uniformity"]:
        lines.append(
            "| {entries} | {bucket} | {position} | {median_us:.4f} | "
            "`{value_ok}` | {confidence:.6f} |".format(**row)
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
            "## Rain State Transfer",
            "",
            "| Dimension | JSON bytes | Rain binary bytes | Rain header bytes | Header under 8 KB | Roundtrip median us |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results["rain_state_transfer"]:
        lines.append(
            "| {dimension} | {json_state_bytes} | {rain_binary_bytes} | "
            "{rain_header_bytes} | `{header_under_8kb}` | {roundtrip_median_us:.4f} |".format(
                **row
            )
        )
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
