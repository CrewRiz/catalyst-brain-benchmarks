from __future__ import annotations


def test_quick_suite_shape():
    from catalyst_brain_benchmarks.benchmarks import run_suite

    results = run_suite(mode="quick")

    assert results["metadata"]["catalyst_brain_version"] == "1.3.2"
    assert results["install_smoke"]["version_ok"] is True
    assert results["install_smoke"]["holo_swarm_method_len"] == 128
    assert results["token_discovery"][0]["compact_page_saved_pct"] > 0.0
    assert results["deferred_outputs"][-1]["saved_pct"] > 90.0
    assert results["bind_unbind_correctness"]["perfect_pct"] == 100.0
    assert all(row["value_ok"] for row in results["hkvc_scaling"])


def test_markdown_report_renders():
    from catalyst_brain_benchmarks.benchmarks import render_markdown_report, run_suite

    report = render_markdown_report(run_suite(mode="quick"))

    assert "Token Discovery Savings" in report
    assert "HKVC Query Scaling" in report
    assert "Memory Model" in report
