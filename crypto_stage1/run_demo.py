"""Run a small end-to-end demo through Stage 0, Stage 1 and Stage 2.

Usage:
    cd ~/crypto_stage1
    python3 run_demo.py
"""

from __future__ import annotations

import json

from pipeline import run_pipeline


def main() -> int:
    records = [
        {
            "record_id": "r1",
            "title": "Bitcoin ETF approved",
            "body": "SEC approves spot bitcoin ETF.",
            "source": "demo",
        },
        {
            "record_id": "r2",
            "title": "Bitcoin ETF approved",
            "body": "SEC approves spot bitcoin ETF.",
            "source": "demo",
        },
        {
            "record_id": "r3",
            "title": "Ethereum upgrade activated",
            "body": "Ethereum activates next major upgrade.",
            "source": "demo",
        },
    ]

    outputs = run_pipeline(records, use_model_voter=True)
    for item in outputs:
        print(json.dumps(item, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
