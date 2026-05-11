from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalyst_brain_benchmarks.benchmarks import render_markdown_report, run_suite, write_outputs
from catalyst_brain_benchmarks.charts import render_all_charts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="catalyst-brain-bench",
        description="Run Catalyst Brain SDK benchmarks from the public PyPI wheel.",
    )
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="quick",
        help="quick is CI-friendly; full runs larger scaling checks.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/local"),
        help="Directory for JSON, CSV, Markdown, and SVG outputs.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip SVG chart generation.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full result payload to stdout.",
    )
    args = parser.parse_args(argv)

    results = run_suite(mode=args.mode)
    write_outputs(results, args.out)
    (args.out / "README.md").write_text(render_markdown_report(results), encoding="utf-8")
    if not args.no_charts:
        chart_dir = args.out / "charts"
        chart_dir.mkdir(parents=True, exist_ok=True)
        render_all_charts(results, chart_dir)

    if args.print_json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print(f"wrote benchmark results to {args.out}")
        print(f"sdk={results['metadata']['catalyst_brain_version']} mode={args.mode}")
    return 0
