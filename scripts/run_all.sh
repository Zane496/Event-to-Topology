#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
for dataset in cev inu emd brn mro; do
  echo "== E2T Full Model: ${dataset}_all =="
  bash "${SCRIPT_DIR}/run_${dataset}.sh" "$@"
done
