"""Formatting helpers for displaying annotation results."""

from __future__ import annotations

from itertools import groupby

import numpy as np
from annotation import AnnotatedOutput


def format_annotations(annotated_output: AnnotatedOutput, show_chunks: bool = False) -> str:
    """Render annotations in a verbose "chess-engine" style."""
    lines: list[str] = []
    lines.append("")
    lines.append("Model Output:")
    lines.append(f'"{annotated_output.text}"')
    lines.append("")

    if not annotated_output.annotations:
        lines.append("📍 Source Analysis:")
        lines.append("  (no annotations)")
        return "\n".join(lines)

    lines.append("📍 Source Analysis:")
    annotations_sorted = sorted(
        annotated_output.annotations,
        key=lambda ann: (ann.text_span, ann.rank),
    )

    for phrase, group in groupby(annotations_sorted, key=lambda ann: ann.text_span):
        phrase_annotations = list(group)
        lines.append(f'  ├─ "{phrase}"')

        for ann in phrase_annotations:
            stage = f", {ann.corpus_stage}" if ann.corpus_stage else ""
            lines.append(
                f"     [{ann.rank}] {ann.cluster_name} (d={ann.distance:.3f}{stage})"
            )

        if show_chunks and phrase_annotations:
            chunk_text = phrase_annotations[0].nearest_chunk_text
            lines.append(f'     Corpus: "{chunk_text}..."')

        lines.append("")

    top_annotations = [ann for ann in annotated_output.annotations if ann.rank == 1]
    if top_annotations:
        distances = [ann.distance for ann in top_annotations]
        avg_distance = float(np.mean(distances))

        if avg_distance < 0.15:
            confidence = "HIGH"
            interpretation = "Model output closely matches training corpus - likely grounded response."
        elif avg_distance < 0.30:
            confidence = "MEDIUM"
            interpretation = "Model output moderately close to training corpus - possible extrapolation."
        else:
            confidence = "LOW"
            interpretation = "Model output far from training corpus - likely hallucination or novel generation."

        lines.append("🎯 Interpretation:")
        lines.append(f"  Average distance: {avg_distance:.3f} ({confidence} confidence)")
        lines.append(f"  {interpretation}")
    else:
        lines.append("🎯 Interpretation:")
        lines.append("  No top-ranked annotations available.")

    lines.append("")
    return "\n".join(lines)


def format_annotations_compact(annotated_output: AnnotatedOutput) -> str:
    """Render annotations in a single-line summary form."""
    if not annotated_output.annotations:
        preview = annotated_output.text[:60]
        if len(annotated_output.text) > 60:
            preview += "..."
        return f'"{preview}" → (no annotations)'

    top_annotations = [ann for ann in annotated_output.annotations if ann.rank == 1]
    clusters_str = ", ".join(
        f"{ann.cluster_name} ({ann.distance:.2f})" for ann in top_annotations[:3]
    )

    text_preview = annotated_output.text
    if len(text_preview) > 60:
        text_preview = text_preview[:60] + "..."

    return f'"{text_preview}" → {clusters_str}'

