from __future__ import annotations

import html
import json
from typing import Any

from ..export import format_content_html
from .artifacts import normalize_sample_payload


def _render_json(value: Any) -> str:
    return f"<pre><code>{html.escape(json.dumps(value, indent=2, ensure_ascii=False))}</code></pre>"


def _render_value(value: Any) -> str:
    if value is None:
        return '<span class="muted">null</span>'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return html.escape(str(value))
    if isinstance(value, str):
        return html.escape(value)
    return _render_json(value)


def _render_message_content(content: Any) -> str:
    if isinstance(content, str):
        return html.escape(content)
    if isinstance(content, list) and all(isinstance(block, dict) for block in content):
        return format_content_html(content)
    return _render_json(content)


def _sample_metrics(sample: dict[str, Any]) -> list[tuple[str, str]]:
    metadata = sample.get("metadata")
    metrics: list[tuple[str, str]] = []

    reward = sample.get("reward")
    if isinstance(reward, (int, float)):
        metrics.append(("Reward", f"{reward:.3f}"))

    status = sample.get("status")
    if not isinstance(status, str) and isinstance(metadata, dict):
        status = metadata.get("status") or metadata.get("stop_reason")
    if isinstance(status, str) and status:
        metrics.append(("Status", status))

    if isinstance(metadata, dict):
        turns_used = metadata.get("turns_used")
        if isinstance(turns_used, int):
            metrics.append(("Turns", str(turns_used)))
        total_tokens = metadata.get("total_tokens")
        if isinstance(total_tokens, int):
            metrics.append(("Tokens", f"{total_tokens:,}"))
        duration_seconds = metadata.get("duration_seconds")
        if isinstance(duration_seconds, (int, float)):
            metrics.append(("Duration", f"{duration_seconds:.2f}s"))

    return metrics


def _message_role_label(message: dict[str, Any]) -> str:
    role = message.get("role")
    if role == "tool":
        tool_call_id = message.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id:
            return f"Tool Result ({tool_call_id})"
        return "Tool Result"
    if isinstance(role, str) and role:
        return role.capitalize()
    return "Message"


def _sample_body(sample: dict[str, Any], *, compact_header: bool = False) -> str:
    sample_id = html.escape(str(sample.get("id", "")))
    prompt = sample.get("prompt")
    ground_truth = sample.get("ground_truth")
    metadata = sample.get("metadata")
    environment_state = sample.get("environment_state")
    trajectory = sample.get("trajectory")
    messages = trajectory.get("messages", []) if isinstance(trajectory, dict) else []

    parts = ['<div class="sample-body">']
    if not compact_header:
        parts.append(f'<h1 class="page-title">Sample {sample_id}</h1>')

    metrics = _sample_metrics(sample)
    if metrics:
        parts.append('<section class="metric-grid">')
        for label, value in metrics:
            parts.append(
                f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
                f'<div class="metric-value">{html.escape(value)}</div></div>'
            )
        parts.append("</section>")

    if isinstance(prompt, str) and prompt:
        parts.append(
            f'<section><h2>Prompt</h2><div class="card prose">{html.escape(prompt)}</div></section>'
        )

    if ground_truth is not None:
        parts.append(
            "<section><h2>Ground Truth</h2>"
            f'<div class="card prose">{_render_value(ground_truth)}</div></section>'
        )

    if sample.get("input") is not None:
        parts.append("<section><h2>Input</h2>")
        parts.append(f'<div class="card">{_render_json(sample.get("input"))}</div></section>')

    if isinstance(metadata, dict) and metadata:
        parts.append("<section><h2>Metadata</h2>")
        parts.append(f'<div class="card">{_render_json(metadata)}</div></section>')

    if environment_state is not None:
        parts.append("<section><h2>Environment State</h2>")
        parts.append(f'<div class="card">{_render_json(environment_state)}</div></section>')

    parts.append("<section><h2>Conversation</h2>")
    if isinstance(messages, list) and messages:
        for index, message in enumerate(messages, start=1):
            if not isinstance(message, dict):
                parts.append(
                    f'<div class="message"><pre><code>{html.escape(str(message))}</code></pre></div>'
                )
                continue
            role = html.escape(str(message.get("role", "message")))
            role_label = html.escape(_message_role_label(message))
            content_html = _render_message_content(message.get("content"))
            parts.append(
                f'<article class="message message-{role}">'
                f'<div class="message-header"><span>{role_label}</span><span class="muted">#{index}</span></div>'
                f'<div class="message-content prose">{content_html}</div>'
                "</article>"
            )
    else:
        parts.append('<div class="card muted">No recorded messages.</div>')
    parts.append("</section>")
    parts.append("</div>")
    return "".join(parts)


def _html_shell(title: str, body: str) -> str:
    safe_title = html.escape(title)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{safe_title}</title>
  <style>
    :root {{
      --bg: #0f1115;
      --surface: #171a21;
      --surface-2: #20242d;
      --border: #313744;
      --text: #eef1f7;
      --muted: #9aa4b2;
      --accent: #7dd3fc;
      --good: #22c55e;
      --warn: #f59e0b;
      --mono: "SFMono-Regular", "SF Mono", Consolas, "Liberation Mono", Menlo, monospace;
      --sans: "SF Pro Display", "Segoe UI", system-ui, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #0b0d12 0%, var(--bg) 100%);
      color: var(--text);
      font-family: var(--sans);
      line-height: 1.5;
    }}
    main {{
      width: min(1200px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 64px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1.page-title {{
      font-family: var(--mono);
      font-size: 28px;
      margin-bottom: 20px;
    }}
    h2 {{
      margin-top: 28px;
      font-size: 14px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    .muted {{ color: var(--muted); }}
    .card {{
      background: color-mix(in srgb, var(--surface) 92%, black);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 16px;
    }}
    .prose {{
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .metric-grid {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      margin-bottom: 20px;
    }}
    .metric-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 16px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .metric-value {{
      font-family: var(--mono);
      font-size: 18px;
      font-weight: 600;
    }}
    pre {{
      margin: 0;
      padding: 14px;
      overflow-x: auto;
      border-radius: 10px;
      background: #0a0c10;
      border: 1px solid #20242c;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.45;
    }}
    code {{ font-family: var(--mono); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }}
    th, td {{
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: var(--surface-2);
    }}
    tbody tr:last-child td {{ border-bottom: none; }}
    .message {{
      margin-bottom: 12px;
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      background: var(--surface);
    }}
    .message-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 14px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: var(--surface-2);
      color: var(--muted);
    }}
    .message-content {{ padding: 14px; }}
    .message-user {{ border-left: 4px solid var(--accent); }}
    .message-assistant {{ border-left: 4px solid var(--good); }}
    .message-tool {{ border-left: 4px solid var(--warn); }}
    .run-sections {{
      display: grid;
      gap: 16px;
    }}
    details.sample-detail {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }}
    details.sample-detail > summary {{
      cursor: pointer;
      padding: 14px 16px;
      font-family: var(--mono);
      font-size: 14px;
      background: var(--surface-2);
    }}
    details.sample-detail .sample-body {{
      padding: 0 16px 20px;
    }}
  </style>
</head>
<body>
  <main>
    {body}
  </main>
</body>
</html>"""


def sample_to_html(trace_id: str, sample_id: str, sample: dict[str, Any]) -> str:
    sample = normalize_sample_payload(sample)
    body = (
        f'<div class="muted" style="margin-bottom: 12px;">Run {html.escape(trace_id)}</div>'
        + _sample_body(sample, compact_header=False)
    )
    return _html_shell(f"{trace_id}:{sample_id}", body)


def run_to_html(
    trace_id: str,
    report: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    sample_link_prefix: str | None = None,
) -> str:
    samples = [normalize_sample_payload(sample) for sample in samples]
    metrics = report.get("summary_metrics", {}) if isinstance(report, dict) else {}
    report_config = report.get("config", {}) if isinstance(report, dict) else {}
    endpoint = report_config.get("endpoint", {}) if isinstance(report_config, dict) else {}
    title = report.get("eval_name") if isinstance(report, dict) else None
    title = title if isinstance(title, str) and title else trace_id

    metric_cards: list[str] = []
    for label, value in [
        ("Total Samples", metrics.get("total_samples")),
        ("Mean Reward", metrics.get("mean_reward")),
        ("Avg Turns", metrics.get("avg_turns")),
        ("Avg Tokens", metrics.get("avg_tokens")),
        ("Success Rate", metrics.get("success_rate")),
        ("Provider Errors", metrics.get("provider_errors")),
    ]:
        if value is None:
            display = "—"
        elif isinstance(value, float):
            display = f"{value:.3f}" if abs(value) < 1000 else f"{value:,.0f}"
        else:
            display = str(value)
        metric_cards.append(
            f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
            f'<div class="metric-value">{html.escape(display)}</div></div>'
        )

    sample_rows: list[str] = []
    sample_details: list[str] = []
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        sample_href = (
            f"{sample_link_prefix.rstrip('/')}/{sample_id}.html"
            if isinstance(sample_link_prefix, str) and sample_link_prefix
            else f"#sample-{sample_id}"
        )
        metadata = sample.get("metadata")
        turns = metadata.get("turns_used") if isinstance(metadata, dict) else None
        tokens = metadata.get("total_tokens") if isinstance(metadata, dict) else None
        status = sample.get("status") or (
            metadata.get("status") if isinstance(metadata, dict) else None
        )
        reward = sample.get("reward")
        reward_text = f"{reward:.3f}" if isinstance(reward, (int, float)) else "—"
        sample_rows.append(
            "<tr>"
            f'<td><a href="{html.escape(sample_href)}">{html.escape(sample_id)}</a></td>'
            f"<td>{html.escape(reward_text)}</td>"
            f"<td>{html.escape(str(turns if turns is not None else '—'))}</td>"
            f"<td>{html.escape(str(f'{tokens:,}' if isinstance(tokens, int) else '—'))}</td>"
            f"<td>{html.escape(str(status if status is not None else '—'))}</td>"
            "</tr>"
        )
        sample_details.append(
            f'<details class="sample-detail" id="sample-{html.escape(sample_id)}">'
            f"<summary>{html.escape(sample_id)} · reward {html.escape(reward_text)}</summary>"
            f"{_sample_body(sample, compact_header=True)}"
            "</details>"
        )

    body = (
        f'<div class="muted" style="margin-bottom: 12px;">Run {html.escape(trace_id)}</div>'
        f'<h1 class="page-title">{html.escape(title)}</h1>'
        '<section class="metric-grid">' + "".join(metric_cards) + "</section>"
        '<section class="card" style="margin-bottom: 24px;">'
        f"<div><strong>Model:</strong> {html.escape(str(endpoint.get('model', '—')))}</div>"
        f"<div><strong>Provider:</strong> {html.escape(str(endpoint.get('provider', '—')))}</div>"
        f"<div><strong>Dataset:</strong> {html.escape(str(report.get('dataset_path', '—')))}</div>"
        f"<div><strong>Timestamp:</strong> {html.escape(str(report.get('timestamp', '—')))}</div>"
        "</section>"
        "<section><h2>Samples</h2><table><thead><tr><th>Sample</th><th>Reward</th><th>Turns</th><th>Tokens</th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(sample_rows)}</tbody></table></section>"
        '<section><h2>Sample Details</h2><div class="run-sections">'
        + "".join(sample_details)
        + "</div></section>"
    )
    return _html_shell(trace_id, body)
