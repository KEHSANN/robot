#!/usr/bin/env bash
# Organise crypto_stage1 files into the package layout.
#
# Idempotent: already-moved files are skipped. The backup directory
# (.organization_backup_*) is intentionally never deleted by this script.
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p stage0 stage1 stage2 services database tests
touch stage0/__init__.py stage1/__init__.py stage2/__init__.py \
      services/__init__.py tests/__init__.py

move_if_present() {
  local src="$1" dst="$2"
  if [[ -f "$src" && ! -f "$dst" ]]; then
    mv "$src" "$dst"
    echo "moved: $src -> $dst"
  fi
}

move_if_present stage0_embedding_store.py stage0/embedding_store.py
move_if_present stage0_schema.sql database/stage0_schema.sql
move_if_present stage0_similarity_test.py tests/stage0_similarity_test.py
move_if_present stage2.py stage2/pipeline.py
move_if_present embedding_service.py services/embedding_service.py

echo "Done. Verify imports with:"
echo "  python -m py_compile stage0/*.py stage1/*.py stage2/*.py services/*.py tests/*.py"
echo "  python -m unittest discover -s tests -p '*_test.py' -v"
