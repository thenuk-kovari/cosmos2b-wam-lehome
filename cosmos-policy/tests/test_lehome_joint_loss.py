"""Exercise production denoising and loss code, without loading the 2B DiT."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from cosmos_policy._src.imaginaire.config import load_config
from cosmos_policy.models.policy_text2world_model import CosmosPolicyDiffusionModel
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel


EXPERIMENT = "cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_prod_v2"
CHUNK30_EXPERIMENT = (
    "cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_chunk30_prod_v3"
)
DUAL_ENDPOINT_EXPERIMENT = (
    "cosmos_predict2_2b_480p_lehome_100_demos_dual_endpoint_jointloss_dropout50_chunk30_prod_v1"
)


@pytest.fixture(scope="module")
def config():
    return load_config("cosmos_policy/config/config.py", ["--", f"experiment={EXPERIMENT}"])


@pytest.fixture(scope="module")
def chunk30_config():
    return load_config(
        "cosmos_policy/config/config.py", ["--", f"experiment={CHUNK30_EXPERIMENT}"]
    )

@pytest.fixture(scope="module")
def dual_endpoint_config():
    return load_config(
        "cosmos_policy/config/config.py",
        ["--", f"experiment={DUAL_ENDPOINT_EXPERIMENT}"],
    )



def _dataloader_without_horizon(config, name):
    result = OmegaConf.to_container(getattr(config, name), resolve=False)

    def canonicalize(value):
        if isinstance(value, dict):
            value.pop("chunk_size", None)
            value.pop("stats_path", None)
            return {key: canonicalize(child) for key, child in value.items()}
        if isinstance(value, list):
            return [canonicalize(child) for child in value]
        if isinstance(value, tuple):
            return tuple(canonicalize(child) for child in value)
        if isinstance(value, partial):
            return (
                "partial", value.func, canonicalize(value.args),
                canonicalize(value.keywords or {}),
            )
        return value

    return canonicalize(result)


def test_chunk30_changes_only_horizon_stats_and_run_name(config, chunk30_config):
    # Exhaustively compare every non-loader field in the resolved configs. The
    # 1-second experiment must not drift from the proven joint-loss setup.
    for field in (
        "model", "optimizer", "scheduler", "trainer", "model_parallel",
        "checkpoint", "upload_reproducible_setup",
    ):
        assert getattr(chunk30_config, field) == getattr(config, field), field

    assert _dataloader_without_horizon(chunk30_config, "dataloader_train") == (
        _dataloader_without_horizon(config, "dataloader_train")
    )
    assert _dataloader_without_horizon(chunk30_config, "dataloader_val") == (
        _dataloader_without_horizon(config, "dataloader_val")
    )
    assert chunk30_config.dataloader_train.dataset.chunk_size == 30
    assert chunk30_config.dataloader_val.dataset.chunk_size == 30
    assert str(chunk30_config.dataloader_train.dataset.stats_path).endswith(
        "lehome_cosmos_stats_chunk30.json"
    )
    assert str(chunk30_config.dataloader_val.dataset.stats_path).endswith(
        "lehome_cosmos_stats_chunk30.json"
    )

    old_job = vars(config.job).copy()
    new_job = vars(chunk30_config.job).copy()
    assert old_job.pop("name") == EXPERIMENT
    assert new_job.pop("name") == CHUNK30_EXPERIMENT
    assert new_job == old_job

def test_dual_endpoint_changes_layout_and_preserves_old_modality_weights(
    chunk30_config, dual_endpoint_config
):
    old_model = chunk30_config.model.config
    new_model = dual_endpoint_config.model.config
    unchanged_loss_fields = (
        "mask_loss_for_action_future_state_prediction",
        "mask_value_prediction_loss_for_policy_prediction",
        "mask_current_state_action_for_value_prediction",
        "mask_future_state_for_qvalue_prediction",
        "action_loss_multiplier",
        "denoise_replace_gt_frames",
        "sigma_data",
    )
    for field in unchanged_loss_fields:
        assert getattr(new_model, field) == getattr(old_model, field), field
    assert old_model.scaling == new_model.scaling == "rectified_flow"
    assert old_model.loss_scale == 10.0
    assert new_model.loss_scale == 13.0
    assert old_model.future_image_loss_multiplier == 1.0
    assert new_model.future_image_loss_multiplier == 0.5

    assert new_model.state_t == 13
    assert new_model.min_num_conditional_frames == 5
    assert new_model.max_num_conditional_frames == 5
    assert new_model.tokenizer.chunk_duration == 49
    assert dual_endpoint_config.optimizer == chunk30_config.optimizer
    assert dual_endpoint_config.scheduler == chunk30_config.scheduler
    assert dual_endpoint_config.checkpoint == chunk30_config.checkpoint
    unchanged_trainer_fields = (
        "max_iter",
        "grad_accum_iter",
        "validation_iter",
        "run_validation",
        "logging_iter",
    )
    for field in unchanged_trainer_fields:
        assert getattr(dual_endpoint_config.trainer, field) == getattr(
            chunk30_config.trainer, field
        )

    train = dual_endpoint_config.dataloader_train.dataset
    val = dual_endpoint_config.dataloader_val.dataset
    assert train.chunk_size == val.chunk_size == 30
    assert train.predict_midpoint_images is True
    assert val.predict_midpoint_images is True
    assert train.demonstration_sampling_prob == 0.75
    assert train.proprio_conditioning_dropout_prob == 0.5
    assert val.proprio_conditioning_dropout_prob == 0.0
    assert dual_endpoint_config.job.name == DUAL_ENDPOINT_EXPERIMENT


def test_resolved_training_contract(config):
    model = config.model.config
    assert model.mask_loss_for_action_future_state_prediction is False
    assert model.mask_value_prediction_loss_for_policy_prediction is False
    assert model.mask_current_state_action_for_value_prediction is False
    assert model.mask_future_state_for_qvalue_prediction is False
    assert model.action_loss_multiplier == 1
    assert model.future_image_loss_multiplier == 1.0
    assert model.scaling == "rectified_flow"
    assert model.sigma_data == 1.0
    assert model.loss_scale == 10.0
    assert model.denoise_replace_gt_frames is True
    assert model.state_t == 10
    assert model.min_num_conditional_frames == model.max_num_conditional_frames == 5
    assert model.tokenizer.chunk_duration == 37
    train = config.dataloader_train.dataset
    assert train.chunk_size == 60
    assert train.demonstration_sampling_prob == 0.75
    assert train.proprio_conditioning_dropout_prob == 0.5
    assert train.return_value_function_returns is False
    assert config.dataloader_val.dataset.proprio_conditioning_dropout_prob == 0.0
    assert config.checkpoint.save_iter == 1000
    assert config.checkpoint.load_training_state is False
    assert str(config.checkpoint.load_path).endswith("model-480p-16fps.pt")


def test_3k_resume_overrides_restore_full_training_state():
    checkpoint_key = (
        "cosmos2b-wam-lehome/cosmos_v2_finetune/"
        "cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_prod_v2/"
        "checkpoints/iter_000003000"
    )
    resume = load_config(
        "cosmos_policy/config/config.py",
        [
            "--", f"experiment={EXPERIMENT}", f"checkpoint.load_path={checkpoint_key}",
            "checkpoint.load_training_state=true",
            "checkpoint.load_from_object_store.enabled=true",
            "checkpoint.load_from_object_store.bucket=policy-training",
            "checkpoint.load_from_object_store.credentials=credentials/aws_default_chain.json",
        ],
    )
    assert resume.job.name == EXPERIMENT
    assert resume.checkpoint.load_path == checkpoint_key
    assert resume.checkpoint.load_training_state is True
    assert resume.checkpoint.load_from_object_store.enabled is True
    assert resume.checkpoint.load_from_object_store.bucket == "policy-training"
    assert resume.checkpoint.save_to_object_store.enabled is True
    assert resume.checkpoint.save_iter == 1000
    assert resume.trainer.max_iter == 50000
    assert resume.model.config.mask_loss_for_action_future_state_prediction is False
    assert resume.model.config.action_loss_multiplier == 1


@pytest.mark.parametrize("sigma_value", [1.0, 4.0, 80.0])
def test_production_loss_and_gradients_cover_all_future_modalities(config, sigma_value):
    # Exact 75/25 batch. Full-size action chunks fit in these reduced spatial latents.
    shape = (4, 16, 10, 8, 8)
    clean = torch.zeros(shape)
    network_output = torch.ones(shape, requires_grad=True)
    condition_mask = torch.zeros((4, 1, 10, 8, 8))
    condition_mask[:, :, :5] = 1
    condition_mask[3, :, 5] = 1  # Only world-model samples condition on true actions.
    condition = SimpleNamespace(
        gt_frames=clean.clone(),
        is_video=True,
        use_video_condition=True,
        condition_video_input_mask_B_C_T_H_W=condition_mask,
        to_dict=lambda: {},
    )
    sigma = torch.full((4, 1), sigma_value)
    sigma_data = float(config.model.config.sigma_data)
    weight = (sigma.square() + sigma_data**2) / (sigma * sigma_data).square()

    # Replace only the expensive network and SDE scaling; use the production
    # denoiser's conditioning/GT replacement and the entire production loss.
    fake = SimpleNamespace(
        config=config.model.config,
        tensor_kwargs={"dtype": torch.float32, "device": "cpu"},
        net=lambda **kwargs: network_output,
        scaling=lambda sigma: (torch.zeros_like(sigma), torch.ones_like(sigma),
                               torch.ones_like(sigma), torch.zeros_like(sigma)),
        sde=SimpleNamespace(marginal_prob=lambda x, s: (x, s)),
        get_per_sigma_loss_weights=lambda sigma: weight,
    )
    fake.denoise = lambda xt, s, c: CosmosPolicyVideo2WorldModel.denoise(fake, xt, s, c)
    idx = lambda value: torch.full((4,), value, dtype=torch.long)
    _, loss, mse, edm = CosmosPolicyDiffusionModel.compute_loss_with_epsilon_and_sigma(
        fake, clean, condition, torch.zeros_like(clean), sigma,
        action_chunk=torch.zeros((4, 60, 12)), action_indices=idx(5),
        proprio=torch.zeros((4, 12)), current_proprio_indices=idx(1),
        future_proprio=torch.zeros((4, 12)), future_proprio_indices=idx(6),
        future_wrist_image_indices=idx(7), future_wrist_image2_indices=idx(8),
        future_image_indices=idx(9), future_image2_indices=idx(-1),
        future_video_start_indices=idx(-1), future_video_end_indices=idx(-1),
        rollout_data_mask=torch.tensor([0, 0, 0, 1]),
        world_model_sample_mask=torch.tensor([0, 0, 0, 1]),
        value_function_sample_mask=idx(0), value_function_return=torch.zeros(4),
        value_indices=idx(-1),
    )
    expected = (1 - condition_mask).expand_as(clean)
    torch.testing.assert_close(mse, expected)
    torch.testing.assert_close(loss, expected * weight[:, None, :, None, None])
    torch.testing.assert_close(loss, edm)
    loss.mean().backward()
    expected_grad = 2 * expected * weight[:, None, :, None, None] / clean.numel()
    torch.testing.assert_close(network_output.grad, expected_grad)
    # Explicitly verify action, future proprio, left, right and head slots.
    per_slot = network_output.grad.abs().sum(dim=(1, 3, 4))
    assert torch.all(per_slot[:3, 5:10] > 0)
    assert torch.all(per_slot[3, 6:10] > 0)
    assert torch.all(per_slot[:, :5] == 0)
    assert per_slot[3, 5] == 0


@pytest.mark.parametrize("sigma_value", [1.0, 4.0, 80.0])
def test_dual_endpoint_loss_and_gradients_cover_t15_and_t30_images(
    dual_endpoint_config, sigma_value
):
    shape = (4, 16, 13, 8, 8)
    clean = torch.zeros(shape)
    network_output = torch.ones(shape, requires_grad=True)
    condition_mask = torch.zeros((4, 1, 13, 8, 8))
    condition_mask[:, :, :5] = 1
    condition_mask[3, :, 5] = 1
    condition = SimpleNamespace(
        gt_frames=clean.clone(),
        is_video=True,
        use_video_condition=True,
        condition_video_input_mask_B_C_T_H_W=condition_mask,
        to_dict=lambda: {},
    )
    sigma = torch.full((4, 1), sigma_value)
    weight = (1 + sigma).square() / sigma.square()
    fake = SimpleNamespace(
        config=dual_endpoint_config.model.config,
        tensor_kwargs={"dtype": torch.float32, "device": "cpu"},
        net=lambda **kwargs: network_output,
        scaling=lambda sigma: (
            torch.zeros_like(sigma),
            torch.ones_like(sigma),
            torch.ones_like(sigma),
            torch.zeros_like(sigma),
        ),
        sde=SimpleNamespace(marginal_prob=lambda x, s: (x, s)),
        get_per_sigma_loss_weights=lambda sigma: weight,
    )
    fake.denoise = lambda xt, s, c: CosmosPolicyVideo2WorldModel.denoise(
        fake, xt, s, c
    )
    idx = lambda value: torch.full((4,), value, dtype=torch.long)
    output, loss, mse, edm = (
        CosmosPolicyDiffusionModel.compute_loss_with_epsilon_and_sigma(
            fake,
            clean,
            condition,
            torch.zeros_like(clean),
            sigma,
            action_chunk=torch.zeros((4, 30, 12)),
            action_indices=idx(5),
            proprio=torch.zeros((4, 12)),
            current_proprio_indices=idx(1),
            future_proprio=torch.zeros((4, 12)),
            future_proprio_indices=idx(6),
            future_wrist_image_indices=idx(10),
            future_wrist_image2_indices=idx(11),
            future_image_indices=idx(12),
            future_image2_indices=idx(-1),
            future_video_start_indices=idx(7),
            future_video_end_indices=idx(12),
            rollout_data_mask=torch.tensor([0, 0, 0, 1]),
            world_model_sample_mask=torch.tensor([0, 0, 0, 1]),
            value_function_sample_mask=idx(0),
            value_function_return=torch.zeros(4),
            value_indices=idx(-1),
        )
    )
    expected_mse = (1 - condition_mask).expand_as(clean)
    expected_edm = expected_mse * weight[:, None, :, None, None]
    expected_loss = expected_edm.clone()
    expected_loss[:, :, 7:13] *= 0.5
    torch.testing.assert_close(mse, expected_mse)
    torch.testing.assert_close(edm, expected_edm)
    torch.testing.assert_close(loss, expected_loss)
    expected_image_metric = expected_mse[:3, :, 7:13].mean()
    torch.testing.assert_close(
        output["demo_sample_future_image_mse_loss"], expected_image_metric
    )

    # Three policy plus one state sample, all slot MSEs equal to one:
    # ((3 * (action 1 + proprio 1 + images 6 * 0.5))
    #   + (proprio 1 + images 6 * 0.5)) / 4 = 4.75.
    final_training_loss = loss.mean() * dual_endpoint_config.model.config.loss_scale
    torch.testing.assert_close(final_training_loss, weight.squeeze().mean() * 4.75)

    loss.mean().backward()
    per_slot = network_output.grad.abs().sum(dim=(1, 3, 4))
    assert torch.all(per_slot[:3, 5:13] > 0)
    assert torch.all(per_slot[3, 6:13] > 0)
    torch.testing.assert_close(
        per_slot[:3, 7:13],
        (per_slot[:3, 6:7] * 0.5).expand(-1, 6),
    )
    torch.testing.assert_close(
        per_slot[3:, 7:13],
        (per_slot[3:, 6:7] * 0.5).expand(-1, 6),
    )
    assert torch.all(per_slot[:, 7:10] > 0)
    assert torch.all(per_slot[:, 10:13] > 0)
    assert torch.all(per_slot[:, :5] == 0)
    assert per_slot[3, 5] == 0
