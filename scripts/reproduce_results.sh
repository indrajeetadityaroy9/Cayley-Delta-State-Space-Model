#!/bin/bash
set -euo pipefail

# Reproducibility settings
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

SEED=42

python scripts/train.py --config configs/default.yaml --seed "$SEED"
