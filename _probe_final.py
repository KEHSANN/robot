import asyncio

from services.keypool import KeyPoolRegistry
from services.llm import LLMClient
from services.logging_setup import redact, setup_logging
from services.models import FINAL_PANEL

setup_logging("WARNING")


async def main() -> None:
    registry = KeyPoolRegistry()
    async with LLMClient(registry) as client:
        for spec in FINAL_PANEL:
            result = await client.complete_text(
                spec, "Reply with the single word OK.", "OK", max_output_tokens=64
            )
            mark = "ok    " if result.ok else "FAILED"
            body = redact(str(result.text or result.error))[:80]
            print(f"{mark} {spec.id:<45} {body!r}", flush=True)


asyncio.run(main())
