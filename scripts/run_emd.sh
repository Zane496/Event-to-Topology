#!/usr/bin/env bash
set -Eeuo pipefail
DATASET="emd_all"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${PROJECT_ROOT}"
if [[ -n "${PYTHON_BIN:-}" ]]; then PYTHON_CMD="${PYTHON_BIN}";
elif command -v python3 >/dev/null 2>&1; then PYTHON_CMD="python3";
elif command -v python >/dev/null 2>&1; then PYTHON_CMD="python";
else echo "Python not found" >&2; exit 127; fi
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
GRAPH="data/${DATASET}/downstream_inputs_${DATASET}/dynamic_4rel_graphs_${DATASET}.npy"
DATES="data/${DATASET}/downstream_inputs_${DATASET}/graph_dates_${DATASET}.csv"
if [[ ! -s "$GRAPH" || ! -s "$DATES" ]]; then
  PRED_ROOT="data/${DATASET}"
  MISSING=0
  for prediction in \
    "test_results_raw_train_OOF_Bstyle_compact_${DATASET}.csv" \
    "test_results_raw_val_B_compact_${DATASET}.csv" \
    "test_results_raw_test_B_compact_${DATASET}.csv"; do
    if [[ ! -s "${PRED_ROOT}/${prediction}" ]]; then
      echo "[MISSING] ${PRED_ROOT}/${prediction}" >&2
      MISSING=1
    fi
  done
  if [[ "$MISSING" -ne 0 ]]; then
    echo "[STOP] Baseline node/feature/prior CSVs alone are insufficient for Full Model training." >&2
    echo "Copy the two Step7 graph/date files or the three Student/OOF predictions for ${DATASET} first." >&2
    exit 2
  fi
  echo "Step7 outputs absent for $DATASET; building from existing Student/OOF predictions."
  "${PYTHON_CMD}" -u upstream/build_topology.py --dataset "$DATASET"
fi
"${PYTHON_CMD}" -u check_data.py --dataset "$DATASET" --stage downstream
"${PYTHON_CMD}" -u main.py --dataset "$DATASET" "$@"
