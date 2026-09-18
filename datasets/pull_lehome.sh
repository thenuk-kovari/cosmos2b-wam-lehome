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
        include_args+=("${task}/${variant}/**")
    done
done

mkdir -p "${target_dir}"

max_attempts="${HF_DOWNLOAD_ATTEMPTS:-12}"
max_workers="${HF_DOWNLOAD_WORKERS:-8}"
for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    if HF_XET_HIGH_PERFORMANCE=1 hf download lehome/dataset_challenge \
        --repo-type dataset \
        --revision main \
        --local-dir "${target_dir}" \
        --max-workers "${max_workers}" \
        --include \
        "${include_args[@]}"; then
        echo "LeHome source variants downloaded to ${target_dir}"
        exit 0
    fi

    if ((attempt == max_attempts)); then
        break
    fi
    delay=$((attempt * 30))
    ((delay > 300)) && delay=300
    echo "Hugging Face download attempt ${attempt}/${max_attempts} failed; retrying in ${delay}s" >&2
    sleep "${delay}"
done

echo "error: LeHome download failed after ${max_attempts} attempts" >&2
exit 1
