from __future__ import annotations


def test_quick_suite_shape():
    from catalyst_brain_benchmarks.benchmarks import run_suite

    results = run_suite(mode="quick")

    assert results["metadata"]["catalyst_brain_version"] == "1.3.3"
    assert results["install_smoke"]["version_ok"] is True
    assert results["install_smoke"]["holo_swarm_method_len"] == 128
    assert results["token_discovery"][0]["compact_page_saved_pct"] > 0.0
    assert results["claim_summary"]["tool_selection_top1_pct"] == 100.0
    assert results["deferred_outputs"][-1]["saved_pct"] > 90.0
    assert results["bind_unbind_correctness"]["perfect_pct"] == 100.0
    assert all(row["value_ok"] for row in results["hkvc_scaling"])
    assert all(row["exact_value_ok"] for row in results["hkvc_path_breakdown"])
    assert all(row["value_ok"] for row in results["hkvc_recency_uniformity"])
    assert all(row["roundtrip_ok"] for row in results["rain_state_transfer"])
    assert {row["method"] for row in results["kv_cache_comparison"]} == {
        "FP16 KV cache",
        "TurboQuant 3.5-bit",
        "KIVI 2-bit",
        "PyramidKV 12%",
        "Catalyst Brain HKVC",
    }
    assert results["claim_summary"]["largest_kv_reduction_vs_fp16_x"] > 1_000_000


def test_markdown_report_renders():
    from catalyst_brain_benchmarks.benchmarks import render_markdown_report, run_suite

    report = render_markdown_report(run_suite(mode="quick"))

    assert "Token Discovery Savings" in report
    assert "Tool Selection Accuracy" in report
    assert "HKVC Query Scaling" in report
    assert "HKVC Path Breakdown" in report
    assert "Rain State Transfer" in report
    assert "KV-Cache Competitor Model" in report
    assert "Memory Model" in report


def test_chart_generation(tmp_path):
    from catalyst_brain_benchmarks.benchmarks import run_suite
    from catalyst_brain_benchmarks.charts import render_all_charts

    render_all_charts(run_suite(mode="quick"), tmp_path)

    expected = {
        "token_savings.svg",
        "tool_selection_accuracy.svg",
        "deferred_output_savings.svg",
        "hkvc_query_latency.svg",
        "hkvc_path_latency.svg",
        "hkvc_recency_latency.svg",
        "hdc_primitive_latency.svg",
        "chain_correctness.svg",
        "rain_state_transfer.svg",
        "memory_model.svg",
        "kv_cache_comparison.svg",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
