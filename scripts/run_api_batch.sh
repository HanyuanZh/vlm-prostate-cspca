#!/usr/bin/env bash
# Batch API run for gemini or gpt.  `build` is offline; `submit` costs money and
# asks for confirmation; `fetch` works once every job has finished (usually
# within 24 h).  Keys come from the environment: GEMINI_API_KEY / OPENAI_API_KEY.
#
#   scripts/run_api_batch.sh <experiment> <gemini|gpt> <data_root> <build|submit|status|fetch>
set -euo pipefail
exp=$1; model=$2; data=$3; action=$4
extra=()
if [[ $action == submit ]]; then
  read -r -p "Submit $exp to $model? This incurs API charges [y/N] " ok
  [[ $ok == y ]] || exit 1
  extra=(--yes)
fi
python -m vlmbench batch "$action" -e "$exp" -m "$model" --data-root "$data" "${extra[@]}"
