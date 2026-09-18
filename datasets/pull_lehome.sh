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
includes=()

for task in "${tasks[@]}"; do
    for variant in "${variants[@]}"; do
        includes+=("${task}/${variant}/**")
    done
done

mkdir -p "${target_dir}"
hf download lehome/dataset_challenge \
    --repo-type dataset \
    --revision main \
    --local-dir "${target_dir}" \
    --include "${includes[@]}"

echo "LeHome source variants downloaded to ${target_dir}"
