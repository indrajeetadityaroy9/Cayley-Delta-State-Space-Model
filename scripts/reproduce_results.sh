#!/bin/bash
set -euo pipefail

# Reproducibility settings
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

SEED=42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python3 -m cdssm.train --config "${REPO_ROOT}/configs/default.yaml" --seed "$SEED"
