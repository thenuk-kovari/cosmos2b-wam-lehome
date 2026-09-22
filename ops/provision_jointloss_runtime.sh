#!/usr/bin/env bash
# Run on the training VM after building cosmos-policy-lehome. No secrets needed.
set -euo pipefail
repo_dir="${COSMOS_REPO_DIR:-/home/ubuntu/cosmos2b-wam}"
mkdir -p /home/ubuntu/.cache/cosmos /home/ubuntu/.cache/huggingface
sudo -n docker run --rm --gpus all --ipc=host \
    -e HOST_USER_NAME=ubuntu -e HOST_USER_ID=1000 -e HOST_GROUP_ID=1000 \
    -e UV_PYTHON_INSTALL_DIR=/home/ubuntu/.cache/uv/python \
    -v "${repo_dir}:/workspace" \
    -v /home/ubuntu/.cache/cosmos:/home/ubuntu/.cache \
    -v /home/ubuntu/.cache/huggingface:/home/ubuntu/.cache/huggingface \
    -w /workspace/cosmos-policy cosmos-policy-lehome \
    uv sync --locked --extra cu128 --python 3.10
sudo -n docker run --rm --ipc=host \
    -e HOST_USER_NAME=ubuntu -e HOST_USER_ID=1000 -e HOST_GROUP_ID=1000 \
    -e UV_PYTHON_INSTALL_DIR=/home/ubuntu/.cache/uv/python \
    -v "${repo_dir}:/workspace" \
    -v /home/ubuntu/.cache/cosmos:/home/ubuntu/.cache \
    -v /home/ubuntu/.cache/huggingface:/home/ubuntu/.cache/huggingface \
    -w /workspace/cosmos-policy cosmos-policy-lehome \
    bash -c 'export PATH="/workspace/cosmos-policy/.venv/bin:$PATH"; bash /workspace/datasets/pull_lehome.sh'
