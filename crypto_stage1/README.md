# crypto_stage1

Crypto event intelligence pipeline under a clean package layout:

```text
stage0/      normalise, deduplicate, embed, cluster
stage1/      consensus over event assignments
stage2/      narrative/output pipeline (placeholder)
services/    embedding provider, shared services
database/    durable schema
tests/       dependency-free unit and end-to-end shape tests
```

The implementation is intentionally dependency-light right now, so it can be
checked and tested on a fresh server without installing provider SDKs.

## Quick checks

```bash
python -m py_compile \
  stage0/*.py \
  stage1/*.py \
  stage2/*.py \
  services/*.py \
  tests/*.py

python -m unittest discover -s tests -p '*_test.py' -v
```

## Layout

```text
crypto_stage1/
├── stage0/
│   └── __init__.py, normalizer.py, dedup.py, similarity.py,
│        event_assignment.py, fact_engine.py, embedding_store.py, pipeline.py
├── stage1/
│   └── __init__.py, schemas.py, consensus.py, runner.py, main.py
├── stage2/
│   └── __init__.py, pipeline.py
├── services/
│   └── __init__.py, embedding_service.py
├── database/
│   └── stage0_schema.sql
├── tests/
│   └── __init__.py, stage0_similarity_test.py, test_stage0_pipeline.py,
│        test_stage1_consensus.py
├── .env
├── .gitignore
├── pyproject.toml
└── organize_project.sh
```

## Notes

- `services/embedding_service.py` is a hashed fallback by default; swap in a
  real provider when keys are configured.
- `.env` is intentionally not committed (root `.gitignore` excludes it).
- The `.organization_backup_*/` directory, if present, should be kept until all
  imports and tests pass.
