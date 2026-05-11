from __future__ import annotations

import html
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
        out_dir / "memory_model.svg",
        title="Fixed Catalyst State vs Standard FP16 KV Cache Model",
        x_label="Tokens",
        y_label="Megabytes",
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
    )


def _scale_points(points: list[tuple[float, float]], y_min: float | None, y_max: float | None):
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
    plot_w = WIDTH - LEFT - RIGHT
    plot_h = HEIGHT - TOP - BOTTOM

    def map_point(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        px = LEFT + ((x - x_min) / (x_max - x_min)) * plot_w
        py = TOP + (1.0 - ((y - actual_y_min) / (actual_y_max - actual_y_min))) * plot_h
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
) -> None:
    all_points = [point for points in series.values() for point in points]
    map_point, bounds = _scale_points(all_points, y_min, y_max)
    x_min, x_max, actual_y_min, actual_y_max = bounds
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea"]
    body = [_svg_frame(title, x_label, y_label, x_min, x_max, actual_y_min, actual_y_max)]
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
) -> str:
    plot_w = WIDTH - LEFT - RIGHT
    plot_h = HEIGHT - TOP - BOTTOM
    lines = [
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        f'<text x="{LEFT}" y="30" class="title">{html.escape(title)}</text>',
        f'<line x1="{LEFT}" y1="{TOP + plot_h}" x2="{LEFT + plot_w}" y2="{TOP + plot_h}" stroke="#111827"/>',
        f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + plot_h}" stroke="#111827"/>',
    ]
    for i in range(5):
        y = TOP + plot_h - (plot_h * i / 4)
        value = y_min + (y_max - y_min) * i / 4
        lines.append(f'<line x1="{LEFT}" y1="{y:.2f}" x2="{LEFT + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{LEFT - 10}" y="{y + 4:.2f}" class="tick" text-anchor="end">{value:.2f}</text>')
    for i in range(5):
        x = LEFT + plot_w * i / 4
        value = x_min + (x_max - x_min) * i / 4
        lines.append(f'<text x="{x:.2f}" y="{HEIGHT - 48}" class="tick" text-anchor="middle">{value:.0f}</text>')
    lines.append(f'<text x="{LEFT + plot_w / 2}" y="{HEIGHT - 14}" class="axis" text-anchor="middle">{html.escape(x_label)}</text>')
    lines.append(
        f'<text x="20" y="{TOP + plot_h / 2}" class="axis" text-anchor="middle" transform="rotate(-90 20 {TOP + plot_h / 2})">{html.escape(y_label)}</text>'
    )
    return "\n".join(lines)


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
