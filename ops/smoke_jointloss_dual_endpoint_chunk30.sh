#!/usr/bin/env bash
# Three-step eight-GPU gate for the t+15/t+30 dual-endpoint run.
set -euo pipefail

repo_dir="${COSMOS_REPO_DIR:-/home/ubuntu/cosmos2b-wam}"
run_name="cosmos_predict2_2b_480p_lehome_100_demos_dual_endpoint_jointloss_dropout50_chunk30_prod_v1"
mkdir -p /home/ubuntu/cosmos-policy-dual-endpoint-preflight

exec sudo -n docker run --rm --gpus all --ipc=host \
    --name cosmos-dual-endpoint-preflight \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HOST_USER_NAME=ubuntu -e HOST_USER_ID=1000 -e HOST_GROUP_ID=1000 \
    -e UV_PYTHON_INSTALL_DIR=/home/ubuntu/.cache/uv/python \
    -e BASE_DATASETS_DIR=/workspace/datasets \
    -e LEHOME_T5_EMBEDDINGS_PATH=/runtime/t5_embeddings.pkl \
    -e COSMOS_POLICY_CHECKPOINT_OVERRIDE=/workspace/pretrained/model-480p-16fps.pt \
    -e IMAGINAIRE_OUTPUT_ROOT=/outputs -e WANDB_MODE=disabled \
    -e JOINTLOSS_PREFLIGHT_MAX_ITER=3 \
    -e JOINTLOSS_PREFLIGHT_CHUNK_SIZE=30 \
    -e JOINTLOSS_PREFLIGHT_STATE_T=13 \
    -e JOINTLOSS_PREFLIGHT_IMAGE_START=7 \
    -e JOINTLOSS_PREFLIGHT_IMAGE_END=12 \
    -e JOINTLOSS_PREFLIGHT_IMAGE_MULTIPLIER=0.5 \
    -e JOINTLOSS_PREFLIGHT_LOSS_SCALE=13 \
    -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    -v "${repo_dir}:/workspace" \
    -v /home/ubuntu/.cache/cosmos:/home/ubuntu/.cache \
    -v /home/ubuntu/.cache/huggingface:/home/ubuntu/.cache/huggingface \
    -v /home/ubuntu/t5_embeddings.pkl:/runtime/t5_embeddings.pkl:ro \
    -v /home/ubuntu/.aws:/home/ubuntu/.aws:ro \
    -v /home/ubuntu/.netrc:/home/ubuntu/.netrc:ro \
    -v /home/ubuntu/cosmos-policy-dual-endpoint-preflight:/outputs \
    -w /workspace/cosmos-policy cosmos-policy-lehome \
    .venv/bin/torchrun --nproc_per_node=8 --master_port=12343 \
    -m cosmos_policy.scripts.preflight_lehome_jointloss \
    --config=cosmos_policy/config/config.py -- \
    experiment="${run_name}" \
    job.name=lehome_dual_endpoint_preflight job.wandb_mode=disabled \
    model.config.tokenizer.vae_pth=/workspace/pretrained/tokenizer/tokenizer.pth \
    trainer.max_iter=3 trainer.logging_iter=1 trainer.run_validation=false \
    checkpoint.save_to_object_store.enabled=false \
    checkpoint.load_from_object_store.enabled=false \
    trainer.callbacks.wandb.save_s3=false trainer.callbacks.wandb_10x.save_s3=false
