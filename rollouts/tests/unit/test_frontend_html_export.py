from __future__ import annotations

from rollouts.frontend.html_export import run_to_html, sample_to_html


def _sample() -> dict:
    return {
        "id": "sample_0001",
        "prompt": "Optimize this kernel.",
        "ground_truth": "A faster kernel patch",
        "reward": 0.75,
        "status": "completed",
        "input": {"prompt": "Optimize this kernel."},
        "metadata": {
            "turns_used": 3,
            "total_tokens": 1024,
            "duration_seconds": 12.5,
        },
        "environment_state": {"kind": "workspace", "files": ["kernel.cu"]},
        "trajectory": {
            "messages": [
                {"role": "user", "content": "Optimize this kernel."},
                {"role": "assistant", "content": "I will inspect the profiler output first."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Profiler shows memory bottleneck."},
                        {
                            "type": "toolCall",
                            "name": "read_file",
                            "arguments": {"path": "kernel.cu"},
                        },
                    ],
                },
            ],
            "completions": [],
        },
    }


def test_sample_to_html_renders_standalone_document() -> None:
    document = sample_to_html("kernel_eval_001", "sample_0001", _sample())

    assert "<!DOCTYPE html>" in document
    assert "Sample sample_0001" in document
    assert "Optimize this kernel." in document
    assert "Tool Call: read_file" in document
    assert "Profiler shows memory bottleneck." in document


def test_run_to_html_embeds_sample_summaries_and_details() -> None:
    report = {
        "eval_name": "kernel_eval",
        "dataset_path": "datasets/kernelbench",
        "timestamp": "2026-03-19T10:00:00",
        "config": {"endpoint": {"provider": "anthropic", "model": "claude-sonnet"}},
        "summary_metrics": {
            "total_samples": 1,
            "mean_reward": 0.75,
            "avg_turns": 3.0,
            "avg_tokens": 1024.0,
            "success_rate": 1.0,
            "provider_errors": 0,
        },
    }

    document = run_to_html("kernel_eval_001", report, [_sample()], sample_link_prefix="samples")

    assert "<!DOCTYPE html>" in document
    assert "kernel_eval" in document
    assert "datasets/kernelbench" in document
    assert 'href="samples/sample_0001.html"' in document
    assert "Sample Details" in document
    assert "Optimize this kernel." in document
