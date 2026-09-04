# crypto_stage1

Crypto event intelligence pipeline:

```text
stage0/      normalise, deduplicate, embed, cluster
stage1/      consensus over event assignments (rule + optional LLM voter)
stage2/      generate narrative / risk output (Gemini primary, OpenAI fallback)
stage3/      final market summary/panel (NVIDIA)
services/    config, LLM client, embedding provider
database/    durable schema
tests/       dependency-free unit and end-to-end tests
```

The implementation is intentionally dependency-free for the core path, so it
can be checked and tested on a bare Linux server without provider SDKs.

Provider mapping:
- Stage 0: Gemini Embedding 2 (`gemini-embedding-002`) when a Gemini key exists.
- Stage 2: Gemini primary, OpenAI fallback.
- Final stage: NVIDIA.

## Quick checks

```bash
python3 -m py_compile \
  run_demo.py main.py pipeline.py \
  stage0/*.py stage1/*.py stage2/*.py stage3/*.py services/*.py tests/*.py

python3 -m unittest discover -s tests -p '*_test.py' -v
```

Expected:

```text
Ran 13 tests ... OK
```

## Run the full pipeline

From this directory:

```bash
python3 run_demo.py
```

Run over your own JSONL feed:

```bash
python3 main.py --input feed.jsonl
python3 main.py --input feed.jsonl --output out.jsonl
cat feed.jsonl | python3 main.py
```

Each input line is a JSON object:

```json
{"record_id":"r1","title":"Bitcoin ETF approved","body":"SEC approves spot bitcoin ETF.","source":"rss"}
```

Output is one JSON object per line with `record_id`, `event_id`, `narrative`,
`risk_label`, `confidence`, and `metadata`, plus a trailing object with
`stage: "final"` for the NVIDIA-generated panel.

## Use a real model

Copy `.env.example` to `.env` and fill the providers you need:

```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash
```

or

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

When no key is configured, the pipeline uses deterministic rule-based
embeddings, voting and narratives. The status flag is recorded as
`used_model` in Stage 2 metadata.

## Layout

```text
crypto_stage1/
├── main.py                 # full JSONL CLI
├── pipeline.py             # Stage 0 -> 1 -> 2 -> Final orchestrator
├── run_demo.py             # small in-memory demo
├── stage0/
├── stage1/
├── stage2/
├── stage3/
├── services/
├── database/
└── tests/
```

## Notes

- `.env` should not be committed; the root repository `.gitignore` already
  excludes it.
- Keep any `.organization_backup_*/` directory until all imports and tests are
  verified.
