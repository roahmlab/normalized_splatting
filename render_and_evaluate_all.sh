#!/bin/bash
# Render every trained model under one or more parent directories and compute
# metrics, fanning the work out across the available GPUs.
#
# Usage:
#   ./render_and_evaluate_all.sh [-i ITERATION] [-g NUM_GPUS] PARENT_DIR [PARENT_DIR ...]
#
# Each PARENT_DIR is expected to contain one subdirectory per trained model.

set -euo pipefail

ITERATION=30000
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)

while getopts "i:g:h" opt; do
    case $opt in
        i) ITERATION=$OPTARG ;;
        g) NUM_GPUS=$OPTARG ;;
        h) sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "Unknown option. Use -h for usage." >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

if [ "$#" -eq 0 ]; then
    echo "error: no parent directories given. Use -h for usage." >&2
    exit 1
fi

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "error: need at least one GPU (got $NUM_GPUS)." >&2
    exit 1
fi

run_on_gpu() {
    local gpu_id=$1
    local folder=$2
    CUDA_VISIBLE_DEVICES=$gpu_id python3 render.py \
        --model_path "$folder" --iteration "$ITERATION" --compute_metrics
}

distribute_and_run() {
    local model_path=$1
    local folders=()
    while IFS= read -r -d '' d; do folders+=("$d"); done \
        < <(find "$model_path" -mindepth 1 -maxdepth 1 -type d -print0)

    if [ "${#folders[@]}" -eq 0 ]; then
        echo "warning: no model directories under $model_path, skipping." >&2
        return
    fi

    local i=0
    for folder in "${folders[@]}"; do
        run_on_gpu $((i % NUM_GPUS)) "$folder" &
        i=$((i + 1))
    done

    wait
    echo "All tasks for $model_path have been completed."
}

for model_path in "$@"; do
    distribute_and_run "$model_path"
done
