#!/usr/bin/env bash
# Smoke test: end-to-end pipeline (collect → train → test) on a tiny
# bundled URL list. Intended to validate that the PhishXGraph reference
# code runs end-to-end on a new machine.
#
# Run with:
#   cd PhishXGraph
#   bash scripts/smoke_test.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

FEATURES_DIR="${FEATURES_DIR:-$ROOT/data/smoke/features}"
MODEL_DIR="${MODEL_DIR:-$ROOT/data/smoke/models}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/data/smoke/results}"
INPUT_TRAIN="${INPUT_TRAIN:-$ROOT/data/samples/input/smoke_train.csv}"
INPUT_TEST="${INPUT_TEST:-$ROOT/data/samples/input/smoke_test.csv}"

mkdir -p "$FEATURES_DIR" "$MODEL_DIR" "$RESULTS_DIR"

export PYTHONPATH="$ROOT"

echo "== [1/4] collect train URLs =="
python -m phishxgraph.cli collect \
    --input "$INPUT_TRAIN" \
    --features-dir "$FEATURES_DIR" \
    --timeout 30

echo "== [2/4] collect test URLs =="
python -m phishxgraph.cli collect \
    --input "$INPUT_TEST" \
    --features-dir "$FEATURES_DIR" \
    --timeout 30

echo "== [3/4] train (XGBoost) =="
python -m phishxgraph.cli train \
    --input "$INPUT_TRAIN" \
    --features-dir "$FEATURES_DIR" \
    --model-dir "$MODEL_DIR" \
    --classifier xgboost

echo "== [4/4] test =="
python -m phishxgraph.cli test \
    --input "$INPUT_TEST" \
    --features-dir "$FEATURES_DIR" \
    --model-dir "$MODEL_DIR" \
    --output "$RESULTS_DIR/predictions.csv"

echo
echo "== smoke test done =="
echo "  predictions: $RESULTS_DIR/predictions.csv"
column -t -s, "$RESULTS_DIR/predictions.csv" | head
