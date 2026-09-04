"""Command line entry point for Stage 1.

Streams JSON records in from stdin (one JSON object per line) and emits JSON
lines out.

Example:

    echo '{"record_id":"r1","normalised_title":"BTC ETF","deduplicated":false,"canonical_id":"r1","event_id":"e1","event_label":"BTC ETF","facts":[],"vector":[],"similarity_score":0.8}' \\
      | python -m stage1.main
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import chain

from .schemas import Stage0Output
from .runner import Stage1Runner


def _load_records(lines) -> list[Stage0Output]:
    records: list[Stage0Output] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        records.append(Stage0Output(**payload))
    return records


def _dump_results(results) -> None:
    for result in results:
        sys.stdout.write(json.dumps(result.__dict__, ensure_ascii=False, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 1 consensus.")
    parser.add_argument("--input", help="Path to newline-delimited JSON Stage 0 output.")
    parser.add_argument("--jsonl", action="store_true", help="Read from stdin if no --input is given.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.input:
        with open(args.input, "r", encoding="utf-8") as handle:
            lines = iter(handle)
    elif args.jsonl or not sys.stdin.isatty():
        lines = iter(chain.from_iterable([sys.stdin]))
    else:
        parser.error("provide --input FILE or pipe JSON lines on stdin")
        return 1

    records = _load_records(lines)
    results = Stage1Runner().run(records)
    _dump_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
