"""Command line interface for corpus-proximity workflows."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from annotation import AnnotatedOutput, annotate_text
from corpus_index import CorpusIndex
from formatting import format_annotations, format_annotations_compact

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def run_python_script(script: str, *args: str) -> int:
    cmd = [sys.executable, str(SCRIPT_DIR / script), *args]
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def run_deploy_pipeline(
    config: str,
    keep_running: bool,
    provider: str | None,
    use_existing: str | None,
    name: str | None,
) -> int:
    cmd = [sys.executable, str(SCRIPT_DIR / "deploy.py"), "--config", config]
    if keep_running:
        cmd.append("--keep-running")
    if provider:
        cmd.extend(["--provider", provider])
    if use_existing:
        cmd.extend(["--use-existing", use_existing])
    if name:
        cmd.extend(["--name", name])
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def run_local_pipeline(config_path: Path, skip_naming: bool) -> int:
    steps = [
        ("prepare_data.py", (str(config_path),)),
        ("embed_chunks.py", (str(config_path),)),
        ("cluster_corpus.py", (str(config_path),)),
    ]
    if not skip_naming:
        steps.append(("name_clusters.py", (str(config_path), "--name")))

    for script, script_args in steps:
        code = run_python_script(script, *script_args)
        if code != 0:
            return code
    return 0


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            data = line.strip()
            if not data:
                continue
            yield json.loads(data)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def handle_index(args: argparse.Namespace) -> int:
    config_path = SCRIPT_DIR / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    if args.deploy_gpu:
        return run_deploy_pipeline(
            config=args.config,
            keep_running=args.keep_running,
            provider=args.provider,
            use_existing=args.use_existing,
            name=args.name,
        )

    return run_local_pipeline(config_path, skip_naming=args.skip_naming)


def handle_annotate(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO)
    index = CorpusIndex.load(args.corpus_index)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(input_path.stem + "_annotated.jsonl")
    )

    records = []
    for row in read_jsonl(input_path):
        prompt = row.get("prompt")
        text = row.get("output") or row.get("text", "")
        if not text:
            continue

        annotated = annotate_text(
            index,
            text,
            k=args.k,
            phrase_level=not args.no_phrase_level,
            prompt=prompt,
        )
        records.append(annotated.to_dict())

    if not records:
        print("No annotations generated; check input file contents.")
        return 1

    write_jsonl(output_path, records)
    print(f"Saved annotations to {output_path}")
    return 0


def handle_show(args: argparse.Namespace) -> int:
    input_path = Path(args.annotated_file)
    if not input_path.exists():
        print(f"Annotated file not found: {input_path}")
        return 1

    entries = list(read_jsonl(input_path))
    if not entries:
        print("Annotated file is empty.")
        return 1

    if args.index < 0 or args.index >= len(entries):
        print(f"Index {args.index} out of range (0 <= index < {len(entries)})")
        return 1

    entry = AnnotatedOutput.from_dict(entries[args.index])
    if args.compact:
        print(format_annotations_compact(entry))
    else:
        print(format_annotations(entry, show_chunks=args.show_chunks))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="corpus-proximity")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Build corpus index")
    index_parser.add_argument(
        "--config",
        default="configs/clustering_01_tiny.py",
        help="Config file relative to corpus-proximity directory",
    )
    index_parser.add_argument(
        "--deploy-gpu", action="store_true", help="Run pipeline on remote GPU"
    )
    index_parser.add_argument(
        "--keep-running", action="store_true", help="Keep remote GPU running after completion"
    )
    index_parser.add_argument(
        "--provider", choices=["runpod", "primeintellect"], help="GPU provider (deploy mode)"
    )
    index_parser.add_argument("--use-existing", help="Existing instance name or SSH (deploy mode)")
    index_parser.add_argument("--name", help="Custom instance name (deploy mode)")
    index_parser.add_argument(
        "--skip-naming", action="store_true", help="Skip cluster naming step (local mode)"
    )
    index_parser.set_defaults(func=handle_index)

    annotate_parser = subparsers.add_parser("annotate", help="Annotate model outputs")
    annotate_parser.add_argument(
        "--corpus-index", required=True, help="Path to corpus index directory"
    )
    annotate_parser.add_argument("--input", required=True, help="Input JSONL with model outputs")
    annotate_parser.add_argument("--output", help="Destination JSONL for annotations")
    annotate_parser.add_argument("--k", type=int, default=3, help="Top-k nearest clusters per span")
    annotate_parser.add_argument(
        "--no-phrase-level", action="store_true", help="Disable sentence-level splitting"
    )
    annotate_parser.set_defaults(func=handle_annotate)

    show_parser = subparsers.add_parser("show", help="Pretty-print annotations")
    show_parser.add_argument("--annotated-file", required=True, help="Annotated JSONL file")
    show_parser.add_argument(
        "--index", type=int, default=0, help="Entry index to display (0-based)"
    )
    show_parser.add_argument(
        "--show-chunks", action="store_true", help="Display nearest corpus chunks"
    )
    show_parser.add_argument("--compact", action="store_true", help="Use compact output format")
    show_parser.set_defaults(func=handle_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
