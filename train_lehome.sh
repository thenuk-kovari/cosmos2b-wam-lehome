#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image_name="${COSMOS_DOCKER_IMAGE:-cosmos-policy-lehome}"
output_dir="${COSMOS_OUTPUT_DIR:-${HOME}/cosmos-policy-output}"
cache_dir="${COSMOS_CACHE_DIR:-${HOME}/.cache/cosmos}"
hf_cache_dir="${HF_HOME:-${HOME}/.cache/huggingface}"
master_port="${MASTER_PORT:-12341}"

required_paths=(
    "${repo_dir}/datasets/lehome"
    "${repo_dir}/datasets/lehome/t5_embeddings.pkl"
    "${repo_dir}/cosmos-policy/.venv/pyvenv.cfg"
    "${HOME}/.aws"
    "${HOME}/.netrc"
)
for path in "${required_paths[@]}"; do
    if [[ ! -e "${path}" ]]; then
        echo "error: required training input is missing: ${path}" >&2
        exit 1
    fi
done

mkdir -p "${output_dir}" "${cache_dir}" "${hf_cache_dir}"

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
    docker_cmd=(sudo docker)
fi

exec "${docker_cmd[@]}" run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e HOST_USER_NAME="$(id -un)" \
    -e HOST_USER_ID="$(id -u)" \
    -e HOST_GROUP_ID="$(id -g)" \
    -e UV_PYTHON_INSTALL_DIR=/home/ubuntu/.cache/uv/python \
    -e BASE_DATASETS_DIR=/workspace/datasets \
    -e IMAGINAIRE_OUTPUT_ROOT=/outputs \
    -e WANDB_DIR=/outputs/wandb \
    -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    -v "${repo_dir}:/workspace" \
    -v "${output_dir}:/outputs" \
    -v "${cache_dir}:/home/ubuntu/.cache" \
    -v "${hf_cache_dir}:/home/ubuntu/.cache/huggingface" \
    -v "${HOME}/.aws:/home/ubuntu/.aws:ro" \
    -v "${HOME}/.netrc:/home/ubuntu/.netrc:ro" \
    -w /workspace/cosmos-policy \
    "${image_name}" \
    .venv/bin/torchrun --nproc_per_node=8 --master_port="${master_port}" \
    -m cosmos_policy.scripts.train \
    --config=cosmos_policy/config/config.py -- \
    experiment=cosmos_predict2_2b_480p_lehome_100_demos_no_value \
    "$@"
