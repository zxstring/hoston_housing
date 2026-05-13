#!/usr/bin/env bash
# Single run.  Usage: SEED=42 bash run.sh   (defaults to 42)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "aiInd" ]]; then
    eval "$(conda shell.bash hook)" && conda activate aiInd
fi

export SEED="${SEED:-42}"
echo "SEED=$SEED"
timeout --signal=KILL 600s python -u train.py
echo "Done. Output → output_s${SEED}/"
