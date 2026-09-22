# LeHome 1-second dual-endpoint training

This fresh-run configuration predicts 30 q0-anchored action deltas and all
three cameras at t+15 (0.5 seconds) and t+30 (1.0 second).

## Layout and objective

The VAE receives a 49-frame pseudo-video and emits 13 temporal latent slots:

| Slot | Contents |
| ---: | --- |
| 0 | blank |
| 1 | q0 (50% conditioning dropout in training) |
| 2-4 | current left wrist, right wrist, and head images |
| 5 | complete 30x12 q0-anchored action chunk |
| 6 | future proprioception at t+30 |
| 7-9 | left wrist, right wrist, and head images at t+15 |
| 10-12 | left wrist, right wrist, and head images at t+30 |

The resolved foundation config uses rectified-flow weighting. For a temporal
slot X, define `ell_X = ((1 + sigma)^2 / sigma^2) * MSE(X, X_hat)`, where MSE is
the mean over latent channels and spatial positions. Policy samples are 75%
(slots 0-4 clean; 5-12 denoised), while state-prediction samples are 25%
(slots 0-5 clean; 6-12 denoised).

The old 10-slot endpoint run used `loss_scale=10`, so its exact objectives were:

```text
L_policy_old = ell_A + ell_q30 + ell_L30 + ell_R30 + ell_H30
L_state_old  =         ell_q30 + ell_L30 + ell_R30 + ell_H30
```

This 13-slot run uses `loss_scale=13` and weights each of slots 7-12 by 0.5:

```text
L_policy_new = ell_A + ell_q30
             + 0.5 * (ell_L15 + ell_R15 + ell_H15
                    + ell_L30 + ell_R30 + ell_H30)
L_state_new  =         ell_q30
             + 0.5 * (ell_L15 + ell_R15 + ell_H15
                    + ell_L30 + ell_R30 + ell_H30)
```

Thus action and future-proprio coefficients remain 1, and total image
coefficient remains 3. The L1 and unweighted component MSE values logged to
W&B are diagnostics only and are not added as separate objectives.

## Fresh VM setup

These commands assume eight NVIDIA GPUs and a checkout at
`/home/ubuntu/cosmos2b-wam`:

```bash
git clone git@github.com:thenuk-kovari/cosmos2b-wam-lehome.git /home/ubuntu/cosmos2b-wam
cd /home/ubuntu/cosmos2b-wam/cosmos-policy
sudo docker build -t cosmos-policy-lehome docker
cd /home/ubuntu/cosmos2b-wam
```

Install these external files; credentials are never stored in Git:

```text
/home/ubuntu/.aws/credentials
/home/ubuntu/.netrc
/home/ubuntu/t5_embeddings.pkl
/home/ubuntu/cosmos2b-wam/pretrained/model-480p-16fps.pt
/home/ubuntu/cosmos2b-wam/pretrained/tokenizer/tokenizer.pth
```

Use mode `0600` for `.netrc` and AWS credentials. The required chunk-30
normalization statistics are committed to the repository. Provision the Python
environment and download the selected 100 training and 12 validation demos:

```bash
cd /home/ubuntu/cosmos2b-wam
bash ops/provision_jointloss_runtime.sh
```

## Validate and launch

First run the local focused suite:

```bash
cd /home/ubuntu/cosmos2b-wam/cosmos-policy
.venv/bin/pytest -q tests/test_lehome_dataset.py tests/test_lehome_joint_loss.py tests/test_lehome_inference.py
```

Then run the three-step eight-GPU smoke gate. It disables W&B and S3 and checks
the actual foundation-weight load, 13-slot layout, six image-loss slots, finite
loss, and nonzero gradients:

```bash
cd /home/ubuntu/cosmos2b-wam
bash ops/smoke_jointloss_dual_endpoint_chunk30.sh 2>&1 | tee /home/ubuntu/dual-endpoint-smoke.log
nohup bash ops/start_jointloss_dual_endpoint_chunk30.sh > /home/ubuntu/dual-endpoint-training.log 2>&1 < /dev/null &
```

Production logs online to W&B and saves every 1,000 optimizer steps to:

```text
s3://policy-training/cosmos2b-wam-lehome/cosmos_v2_finetune/cosmos_predict2_2b_480p_lehome_100_demos_dual_endpoint_jointloss_dropout50_chunk30_prod_v1/checkpoints/
```

This is a fresh foundation-model run; do not set a resume checkpoint override.
Follow the startup log and verify all eight GPUs are active before leaving it.
