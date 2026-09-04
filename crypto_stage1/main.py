"""Command line entry point for the full crypto_stage1 pipeline.

Reads newline-delimited JSON from ``--input`` (or stdin) and writes Stage 2
output as JSON lines to ``--output`` (or stdout).

Examples:
    python3 main.py --input feed.jsonl --output out.jsonl
    cat feed.jsonl | python3 main.py
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from pipeline import run_pipeline


def load_records(path: str | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    handle = open(path, "r", encoding="utf-8") if path else sys.stdin
    for line in handle:
        line = line.strip()
        if line:
            records.append(json.loads(line))
    if path:
        handle.close()
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 0 -> 1 -> 2 pipeline.")
    parser.add_argument("--input", help="JSONL input file (defaults to stdin).")
    parser.add_argument("--output", help="JSONL output file (defaults to stdout).")
    parser.add_argument("--no-model-voter", action="store_true", help="Disable LLM voter in Stage 1.")
    args = parser.parse_args(argv)

    records = load_records(args.input)
    outputs = run_pipeline(records, use_model_voter=not args.no_model_voter)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            for output in outputs:
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
    else:
        for output in outputs:
            sys.stdout.write(json.dumps(output, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
