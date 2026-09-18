#!/usr/bin/env bash
set -euo pipefail

if ! command -v hf >/dev/null 2>&1; then
    echo "error: install the Hugging Face CLI first (the Cosmos uv environment provides it)" >&2
    exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target_dir="${script_dir}/lehome"

tasks=(
    record_pant_long_release_10
    record_pant_short_release_10
    record_top_long_release_10
    record_top_short_release_10
)
variants=(001 002 003 004 005)
include_args=()

for task in "${tasks[@]}"; do
    for variant in "${variants[@]}"; do
        include_args+=(--include "${task}/${variant}/**")
    done
done

mkdir -p "${target_dir}"
HF_XET_HIGH_PERFORMANCE=1 hf download lehome/dataset_challenge \
    --repo-type dataset \
    --revision main \
    --local-dir "${target_dir}" \
    --max-workers 4 \
    "${include_args[@]}"

echo "LeHome source variants downloaded to ${target_dir}"
