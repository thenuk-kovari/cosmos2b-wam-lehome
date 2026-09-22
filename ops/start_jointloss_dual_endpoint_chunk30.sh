#!/usr/bin/env bash
# Fresh one-second LeHome run with three-camera targets at t+15 and t+30.
set -euo pipefail

repo_dir="${COSMOS_REPO_DIR:-/home/ubuntu/cosmos2b-wam}"
run_name="cosmos_predict2_2b_480p_lehome_100_demos_dual_endpoint_jointloss_dropout50_chunk30_prod_v1"
export COSMOS_EXPERIMENT="${run_name}"
export COSMOS_POLICY_CHECKPOINT_OVERRIDE=/workspace/pretrained/model-480p-16fps.pt

for required in \
    "${repo_dir}/pretrained/model-480p-16fps.pt" \
    "${repo_dir}/pretrained/tokenizer/tokenizer.pth" \
    "${repo_dir}/datasets/lehome/lehome_cosmos_stats_chunk30.json" \
    "${repo_dir}/cosmos-policy/.venv/pyvenv.cfg" \
    /home/ubuntu/t5_embeddings.pkl \
    /home/ubuntu/.aws/credentials \
    /home/ubuntu/.netrc; do
    test -s "${required}" || { echo "Missing required training input: ${required}" >&2; exit 1; }
done

exec bash "${repo_dir}/train_lehome.sh" \
    model.config.tokenizer.vae_pth=/workspace/pretrained/tokenizer/tokenizer.pth
