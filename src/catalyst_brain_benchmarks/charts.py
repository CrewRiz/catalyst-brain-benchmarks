from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any


WIDTH = 960
HEIGHT = 540
LEFT = 84
RIGHT = 36
TOP = 46
BOTTOM = 78


def render_all_charts(results: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _render_line_chart(
        out_dir / "token_savings.svg",
        title="Token Efficiency: Progressive Tool Discovery",
        x_label="Registered tools",
        y_label="Saved context (%)",
        series={
            "compact page": [
                (row["tool_count"], row["compact_page_saved_pct"])
                for row in results["token_discovery"]
            ],
            "compact discovery": [
                (row["tool_count"], row["compact_discovery_saved_pct"])
                for row in results["token_discovery"]
            ],
            "one schema expanded": [
                (row["tool_count"], row["expanded_one_schema_saved_pct"])
                for row in results["token_discovery"]
            ],
        },
        y_min=0.0,
        y_max=100.0,
    )
    selection_accuracy = _selection_accuracy_by_count(results["tool_selection_accuracy"])
    _render_line_chart(
        out_dir / "tool_selection_accuracy.svg",
        title="Progressive Discovery: Tool Selection Quality",
        x_label="Registered tools",
        y_label="Accuracy (%)",
        series={
            "top-1": [
                (row["tool_count"], row["top1_pct"])
                for row in selection_accuracy
            ],
            "top-3": [
                (row["tool_count"], row["top3_pct"])
                for row in selection_accuracy
            ],
        },
        y_min=0.0,
        y_max=100.0,
    )
    _render_bar_chart(
        out_dir / "deferred_output_savings.svg",
        title="Deferred Tool Output: Context Kept Out Until Fetch",
        x_label="stdout bytes",
        y_label="Saved context (%)",
        bars=[
            (str(row["stdout_bytes"]), row["saved_pct"])
            for row in results["deferred_outputs"]
        ],
        y_max=100.0,
    )
    _render_line_chart(
        out_dir / "hkvc_query_latency.svg",
        title="HKVC Query Scaling",
        x_label="Stored entries",
        y_label="Median query latency (us)",
        series={
            "median": [
                (row["entries"], row["median_us"])
                for row in results["hkvc_scaling"]
            ],
            "p95": [
                (row["entries"], row["p95_us"])
                for row in results["hkvc_scaling"]
            ],
        },
    )
    _render_line_chart(
        out_dir / "hkvc_path_latency.svg",
        title="HKVC Exact-Key Hit vs Missing-Key Fallback",
        x_label="Stored entries",
        y_label="Median query latency (us)",
        series={
            "exact indexed hit": [
                (row["entries"], row["exact_median_us"])
                for row in results["hkvc_path_breakdown"]
            ],
            "missing-key fallback": [
                (row["entries"], row["miss_fallback_median_us"])
                for row in results["hkvc_path_breakdown"]
            ],
        },
    )
    _render_line_chart(
        out_dir / "hkvc_recency_latency.svg",
        title="HKVC Query Latency by Entry Position",
        x_label="Normalized position in cache",
        y_label="Median query latency (us)",
        series={
            f"{entries} entries": [
                (row["position_fraction"], row["median_us"])
                for row in results["hkvc_recency_uniformity"]
                if row["entries"] == entries
            ]
            for entries in sorted(
                {row["entries"] for row in results["hkvc_recency_uniformity"]}
            )
        },
    )
    _render_line_chart(
        out_dir / "hdc_primitive_latency.svg",
        title="HDC Primitive Median Latency",
        x_label="Dimension",
        y_label="Median latency (us)",
        series={
            operation: [
                (row["dimension"], row["median_us"])
                for row in results["hdc_primitives"]
                if row["operation"] == operation
            ]
            for operation in ("bind", "unbind", "bundle", "resonance")
        },
    )
    _render_line_chart(
        out_dir / "chain_correctness.svg",
        title="Bind/Unbind Correctness Through Chained Composition",
        x_label="Composition depth",
        y_label="Resonance",
        series={
            "resonance": [
                (row["depth"], row["resonance"])
                for row in results["bind_unbind_correctness"]["chained_composition"]
            ]
        },
        y_min=0.999,
        y_max=1.001,
    )
    _render_line_chart(
        out_dir / "memory_model.svg",
        title="Fixed Catalyst State vs Standard FP16 KV Cache Model",
        x_label="Tokens",
        y_label="Megabytes (log scale)",
        series={
            "standard FP16 KV cache": [
                (row["tokens"], row["standard_kv_cache_mb"])
                for row in results["memory_model"]
            ],
            "Catalyst world vector": [
                (row["tokens"], row["catalyst_world_vector_mb"])
                for row in results["memory_model"]
            ],
            "compressed Rain header": [
                (row["tokens"], row["catalyst_rain_header_mb"])
                for row in results["memory_model"]
            ],
        },
        x_scale="log",
        y_scale="log",
    )
    _render_line_chart(
        out_dir / "rain_state_transfer.svg",
        title="Rain Stateless Agent State Transfer Size",
        x_label="Hypervector dimension",
        y_label="Bytes (log scale)",
        series={
            "JSON state": [
                (row["dimension"], row["json_state_bytes"])
                for row in results["rain_state_transfer"]
            ],
            "Rain binary": [
                (row["dimension"], row["rain_binary_bytes"])
                for row in results["rain_state_transfer"]
            ],
            "Rain header": [
                (row["dimension"], row["rain_header_bytes"])
                for row in results["rain_state_transfer"]
            ],
        },
        x_scale="log",
        y_scale="log",
    )
    _render_line_chart(
        out_dir / "kv_cache_comparison.svg",
        title="KV-Cache Memory Scaling: Published Methods vs Catalyst Fixed State",
        x_label="Context tokens",
        y_label="Modeled memory (MB, log scale)",
        series={
            method: [
                (row["tokens"], row["memory_mb"])
                for row in results["kv_cache_comparison"]
                if row["method"] == method
            ]
            for method in _ordered_methods(results["kv_cache_comparison"])
        },
        x_scale="log",
        y_scale="log",
    )


def _ordered_methods(rows: list[dict[str, Any]]) -> list[str]:
    methods: list[str] = []
    for row in rows:
        method = row["method"]
        if method not in methods:
            methods.append(method)
    return methods


def _selection_accuracy_by_count(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for tool_count in sorted({row["tool_count"] for row in rows}):
        subset = [row for row in rows if row["tool_count"] == tool_count]
        out.append(
            {
                "tool_count": tool_count,
                "top1_pct": 100.0
                * sum(1 for row in subset if row["top1_ok"])
                / len(subset),
                "top3_pct": 100.0
                * sum(1 for row in subset if row["top3_ok"])
                / len(subset),
            }
        )
    return out


def _transform(value: float, scale: str) -> float:
    if scale == "log":
        return math.log10(max(value, 1e-12))
    return value


def _scale_points(
    points: list[tuple[float, float]],
    y_min: float | None,
    y_max: float | None,
    *,
    x_scale: str = "linear",
    y_scale: str = "linear",
):
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min = min(x_values)
    x_max = max(x_values)
    if x_min == x_max:
        x_max = x_min + 1.0
    actual_y_min = min(y_values) if y_min is None else y_min
    actual_y_max = max(y_values) if y_max is None else y_max
    if actual_y_min == actual_y_max:
        actual_y_max = actual_y_min + 1.0
    scaled_x_min = _transform(x_min, x_scale)
    scaled_x_max = _transform(x_max, x_scale)
    scaled_y_min = _transform(actual_y_min, y_scale)
    scaled_y_max = _transform(actual_y_max, y_scale)
    if scaled_x_min == scaled_x_max:
        scaled_x_max = scaled_x_min + 1.0
    if scaled_y_min == scaled_y_max:
        scaled_y_max = scaled_y_min + 1.0
    plot_w = WIDTH - LEFT - RIGHT
    plot_h = HEIGHT - TOP - BOTTOM

    def map_point(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        scaled_x = _transform(x, x_scale)
        scaled_y = _transform(y, y_scale)
        px = LEFT + ((scaled_x - scaled_x_min) / (scaled_x_max - scaled_x_min)) * plot_w
        py = TOP + (1.0 - ((scaled_y - scaled_y_min) / (scaled_y_max - scaled_y_min))) * plot_h
        return px, py

    return map_point, (x_min, x_max, actual_y_min, actual_y_max)


def _render_line_chart(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[float, float]]],
    y_min: float | None = None,
    y_max: float | None = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
) -> None:
    all_points = [point for points in series.values() for point in points]
    map_point, bounds = _scale_points(
        all_points,
        y_min,
        y_max,
        x_scale=x_scale,
        y_scale=y_scale,
    )
    x_min, x_max, actual_y_min, actual_y_max = bounds
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#111827", "#f59e0b"]
    body = [
        _svg_frame(
            title,
            x_label,
            y_label,
            x_min,
            x_max,
            actual_y_min,
            actual_y_max,
            x_scale=x_scale,
            y_scale=y_scale,
        )
    ]
    for idx, (name, points) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        mapped = [map_point(point) for point in points]
        path_data = " ".join(
            ("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}"
            for i, (x, y) in enumerate(mapped)
        )
        body.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in mapped:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        legend_y = TOP + 24 * idx
        body.append(f'<rect x="{WIDTH - 240}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        body.append(
            f'<text x="{WIDTH - 218}" y="{legend_y + 12}" class="legend">{html.escape(name)}</text>'
        )
    path.write_text(_wrap_svg("\n".join(body)), encoding="utf-8")


def _render_bar_chart(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    bars: list[tuple[str, float]],
    y_max: float,
) -> None:
    plot_w = WIDTH - LEFT - RIGHT
    plot_h = HEIGHT - TOP - BOTTOM
    body = [_svg_frame(title, x_label, y_label, 0, max(1, len(bars) - 1), 0, y_max)]
    gap = 24
    bar_w = max(24, (plot_w - gap * (len(bars) + 1)) / max(1, len(bars)))
    for i, (label, value) in enumerate(bars):
        x = LEFT + gap + i * (bar_w + gap)
        h = (value / y_max) * plot_h
        y = TOP + plot_h - h
        body.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="#2563eb"/>')
        body.append(f'<text x="{x + bar_w / 2:.2f}" y="{HEIGHT - 48}" class="tick" text-anchor="middle">{html.escape(label)}</text>')
        body.append(f'<text x="{x + bar_w / 2:.2f}" y="{y - 8:.2f}" class="label" text-anchor="middle">{value:.1f}%</text>')
    path.write_text(_wrap_svg("\n".join(body)), encoding="utf-8")


def _svg_frame(
    title: str,
    x_label: str,
    y_label: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    x_scale: str = "linear",
    y_scale: str = "linear",
) -> str:
    plot_w = WIDTH - LEFT - RIGHT
    plot_h = HEIGHT - TOP - BOTTOM
    lines = [
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        f'<text x="{LEFT}" y="30" class="title">{html.escape(title)}</text>',
        f'<line x1="{LEFT}" y1="{TOP + plot_h}" x2="{LEFT + plot_w}" y2="{TOP + plot_h}" stroke="#111827"/>',
        f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + plot_h}" stroke="#111827"/>',
    ]
    for i, value in enumerate(_tick_values(y_min, y_max, y_scale)):
        y = TOP + plot_h - (plot_h * i / 4)
        lines.append(f'<line x1="{LEFT}" y1="{y:.2f}" x2="{LEFT + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{LEFT - 10}" y="{y + 4:.2f}" class="tick" text-anchor="end">{_format_tick(value)}</text>')
    integer_x_ticks = x_scale == "log" or abs(x_max - x_min) > 10
    for i, value in enumerate(_tick_values(x_min, x_max, x_scale)):
        x = LEFT + plot_w * i / 4
        lines.append(f'<text x="{x:.2f}" y="{HEIGHT - 48}" class="tick" text-anchor="middle">{_format_tick(value, integer=integer_x_ticks)}</text>')
    lines.append(f'<text x="{LEFT + plot_w / 2}" y="{HEIGHT - 14}" class="axis" text-anchor="middle">{html.escape(x_label)}</text>')
    lines.append(
        f'<text x="20" y="{TOP + plot_h / 2}" class="axis" text-anchor="middle" transform="rotate(-90 20 {TOP + plot_h / 2})">{html.escape(y_label)}</text>'
    )
    return "\n".join(lines)


def _tick_values(start: float, stop: float, scale: str) -> list[float]:
    if scale == "log":
        low = _transform(start, "log")
        high = _transform(stop, "log")
        return [10 ** (low + (high - low) * i / 4) for i in range(5)]
    return [start + (stop - start) * i / 4 for i in range(5)]


def _format_tick(value: float, *, integer: bool = False) -> str:
    if integer:
        return f"{value:,.0f}"
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.4g}"


def _wrap_svg(body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<style>
.title {{ font: 700 22px system-ui, -apple-system, Segoe UI, sans-serif; fill: #111827; }}
.axis {{ font: 600 14px system-ui, -apple-system, Segoe UI, sans-serif; fill: #374151; }}
.tick {{ font: 12px system-ui, -apple-system, Segoe UI, sans-serif; fill: #4b5563; }}
.legend {{ font: 13px system-ui, -apple-system, Segoe UI, sans-serif; fill: #111827; }}
.label {{ font: 700 12px system-ui, -apple-system, Segoe UI, sans-serif; fill: #111827; }}
</style>
{body}
</svg>
'''
