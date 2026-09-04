import asyncio
import json
import os
import time

import httpx

from services.config import PROVIDER_BASE_URL, read_key_pool
from services.logging_setup import redact
from services.models import FINAL_PANEL

BASE = PROVIDER_BASE_URL["nvidia"]
KEY = read_key_pool("NVIDIA")[0]


async def main() -> None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        for spec in FINAL_PANEL:
            payload = {
                "model": spec.id,
                "messages": [{"role": "user", "content": "Reply with the single word OK."}],
                "max_tokens": 64,
                "temperature": 0.2,
            }
            started = time.monotonic()
            try:
                r = await client.post(
                    f"{BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                    json=payload,
                )
                elapsed = time.monotonic() - started
                body = redact(r.text)[:400].replace("\n", " ")
                print(f"{spec.id:<45} HTTP {r.status_code} in {elapsed:6.1f}s  {body!r}", flush=True)
            except Exception as exc:  # noqa: BLE001
                elapsed = time.monotonic() - started
                print(
                    f"{spec.id:<45} EXC  {type(exc).__name__} in {elapsed:6.1f}s "
                    f"{redact(str(exc))[:200]!r}",
                    flush=True,
                )


asyncio.run(main())
