# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.utils import log
from cosmos_policy._src.imaginaire.utils.checkpoint_db import get_checkpoint_path  # noqa: F401
from cosmos_policy.datasets.aloha_dataset import ALOHADataset
from cosmos_policy.datasets.lehome_dataset import LeHomeDataset
from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.datasets.robocasa_dataset import RoboCasaDataset
from cosmos_policy.lehome_dual_endpoint_layout import (
    ACTION_CHUNK_SIZE as LEHOME_DUAL_ENDPOINT_ACTION_CHUNK_SIZE,
    NUM_CONDITIONAL_LATENTS as LEHOME_DUAL_ENDPOINT_NUM_CONDITIONAL_LATENTS,
    RAW_SEQUENCE_FRAMES as LEHOME_DUAL_ENDPOINT_RAW_SEQUENCE_FRAMES,
    STATE_T as LEHOME_DUAL_ENDPOINT_STATE_T,
)
from cosmos_policy.lehome_layout import (
    ACTION_CHUNK_SIZE as LEHOME_ACTION_CHUNK_SIZE,
    NUM_CONDITIONAL_LATENTS as LEHOME_NUM_CONDITIONAL_LATENTS,
    RAW_SEQUENCE_FRAMES as LEHOME_RAW_SEQUENCE_FRAMES,
    STATE_T as LEHOME_STATE_T,
    VIDEO_FPS as LEHOME_VIDEO_FPS,
)
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE

cs = ConfigStore.instance()
val_sampling_size_override = dict(
    video_length=121,
    video_height=704,
    video_width=1280,
)
BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")


# *** Main checkpoint ***
libero_all_4_suites_dataset = L(LIBERODataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=16,
    use_image_aug=True,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "all_episodes"
    ),  # All demo rollouts (successes + failures)
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.99,
)
cosmos_predict2_2b_480p_libero = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-102-Size-2B-Res-480-Fps-16-Note-HQ_V5_from_26",
            {"override /data_train": "mock"},
            {"override /model": "policy_fsdp"},
            {"override /tokenizer": "policy_wan2pt1_tokenizer"},
            {
                "override /callbacks": [
                    "basic",
                    "long",
                    "cluster_speed",
                    "wandb",
                    "wandb_callback_actions",
                ]
            },
            "_self_",
        ],
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=100000,
                    save_s3=False,
                    use_negative_prompt=False,
                    guidance=[0],
                    num_sampling_step=9,
                ),
            ),
            run_validation=False,
            logging_iter=5,
            max_iter=1000000,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            # LR decay for 30K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[30000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                conditioner=dict(
                    text=dict(
                        # IMPORTANT: We don't want any text dropout; otherwise, the model may fail to follow language
                        dropout_rate=0.0,
                    ),
                ),
                state_t=9,  # Latent temporal dim (blank, proprio, wrist, primary, action, future proprio, future wrist, future primary, value)
                min_num_conditional_frames=4,  # 1 blank, 3 conditioning (proprio, wrist, primary)
                max_num_conditional_frames=4,  # 1 blank, 3 conditioning (proprio, wrist, primary)
                sigma_conditional=0.0,  # No noise on conditional latents
                conditioning_strategy="frame_replace",
                denoise_replace_gt_frames=True,
                tokenizer=dict(
                    chunk_duration=33,  # 1 blank + 32 images (4 proprio, 4 wrist image, 4 primary image, 4 action, 4 future proprio, 4 future wrist, 4 future primary, 4 value)
                ),
                ema=dict(
                    enabled=False,
                ),
                input_data_key="video",
                sde=L(HybridEDMSDE)(
                    hybrid_sigma_distribution=True,
                    p_mean=1.3862943611198906,  # Copied from base model config
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                    uniform_lower=1.0,
                    uniform_upper=85.0,
                ),
                adjust_video_noise=True,
                resize_online=True,
                resolution="224",
                high_sigma_strategy="none",
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        checkpoint=dict(
            load_path=get_checkpoint_path("hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"),
            load_training_state=False,  # This means do not load train state from the base checkpoint above (load_path); but when resuming this job, will load train state
            strict_resume=False,
            save_iter=1000,
            load_ema_to_reg=True,
            load_from_object_store=dict(
                enabled=False,
            ),
            save_to_object_store=dict(
                enabled=False,
            ),
        ),
        dataloader_train=L(DataLoader)(
            # Each worker decodes three AV1 streams. Four workers per rank,
            # paired with the dataset's bounded reader cache, avoids retaining
            # hundreds of libdav1d decoder contexts across shuffled episodes.
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero_all_4_suites_dataset,
            sampler=L(DistributedSampler)(
                dataset=libero_all_4_suites_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=30,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_libero",
        ),
        upload_reproducible_setup=False,
    )
)
# Inference version
cosmos_predict2_2b_480p_libero__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_libero__inference_only",
        ),
    )
)


# *** Main checkpoint ***
robocasa_50_demos_per_task_dataset = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "all_episodes"
    ),  # All demo rollouts (successes + failures)
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.99,
)
cosmos_predict2_2b_480p_robocasa_50_demos_per_task = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,  # Latent temporal dim (blank, proprio, wrist image, primary image, secondary image, action, future proprio, future wrist image, future primary image, future secondary image, value)
                min_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, wrist image, primary image, secondary image)
                max_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, wrist image, primary image, secondary image)
                tokenizer=dict(
                    chunk_duration=41,  # 1 blank + 40 images (4 proprio, 4 wrist image, 4 primary image, 4 secondary image, 4 action, 4 future proprio, 4 future wrist, 4 future primary, 4 future secondary, 4 value)
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_50_demos_per_task_dataset,
            sampler=L(DistributedSampler)(
                dataset=robocasa_50_demos_per_task_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robocasa_50_demos_per_task",
        ),
    )
)
# Inference version
cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robocasa_50_demos_per_task",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference",
        ),
    )
)


# *** Main checkpoint ***
aloha_cosmos_policy_dataset_185_demos = L(ALOHADataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed"),
    t5_text_embeddings_path=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed", "t5_embeddings.pkl"),
    chunk_size=50,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    treat_demos_as_success_rollouts=True,  # Include demos as success rollouts
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.998,  # Higher gamma for ALOHA because episodes can have up to 1.5-2.0K steps  # (s, a, s', v)
)
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        scheduler=dict(
            # LR decay for 20K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[2000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,  # Latent temporal dim (blank, proprio, left wrist, right wrist, primary, action, future proprio, future left wrist, future right wrist, future primary, value)
                min_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, left wrist, right wrist, primary)
                max_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, left wrist, right wrist, primary)
                tokenizer=dict(
                    chunk_duration=41,  # 1 blank + 40 images (4 proprio, 4 left wrist image, 4 right wrist image, 4 primary image, 4 action, 4 future proprio, 4 future left wrist, 4 future right wrist, 4 future primary, 4 value)
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=aloha_cosmos_policy_dataset_185_demos,
            sampler=L(DistributedSampler)(
                dataset=aloha_cosmos_policy_dataset_185_demos,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
        ),
    )
)
# Inference version
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only",
        ),
    )
)


# LeHome folding policy: 100 train demonstrations, 12 held-out validation episodes.
lehome_data_dir = os.path.join(BASE_DATASETS_DIR, "lehome")
lehome_t5_embeddings_path = os.environ.get(
    "LEHOME_T5_EMBEDDINGS_PATH",
    os.path.join(lehome_data_dir, "t5_embeddings.pkl"),
)
lehome_stats_path = os.path.join(lehome_data_dir, "lehome_cosmos_stats.json")

lehome_train_dataset = L(LeHomeDataset)(
    data_dir=lehome_data_dir,
    split="train",
    t5_text_embeddings_path=lehome_t5_embeddings_path,
    stats_path=lehome_stats_path,
    chunk_size=60,
    use_image_aug=True,
    use_stronger_image_aug=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    demonstration_sampling_prob=0.75,
    proprio_conditioning_dropout_prob=0.5,
    use_wrist_images=False,
    use_third_person_images=True,
    use_proprio=True,
    return_value_function_returns=False,
)
lehome_validation_dataset = L(LeHomeDataset)(
    data_dir=lehome_data_dir,
    split="validation",
    t5_text_embeddings_path=lehome_t5_embeddings_path,
    stats_path=lehome_stats_path,
    chunk_size=60,
    use_image_aug=False,
    use_stronger_image_aug=False,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    demonstration_sampling_prob=0.75,
    proprio_conditioning_dropout_prob=0.0,
)

cosmos_predict2_2b_480p_lehome_100_demos_no_value = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            run_validation=True,
            run_validation_on_start=False,
            validation_iter=5000,
            max_val_iter=None,
            max_iter=50000,
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[2000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        checkpoint=dict(
            save_iter=1000,
            save_to_object_store=dict(
                enabled=True,
                bucket="policy-training",
                credentials="credentials/aws_default_chain.json",
            ),
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=10,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(
                    chunk_duration=37,
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=lehome_train_dataset,
            sampler=L(DistributedSampler)(
                dataset=lehome_train_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        dataloader_val=L(DataLoader)(
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            dataset=lehome_validation_dataset,
            sampler=L(DistributedSampler)(
                dataset=lehome_validation_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=False,
                seed=0,
            ),
            batch_size=25,
            drop_last=False,
        ),
        job=dict(
            project="cosmos2b-wam-lehome",
            wandb_mode="online",
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_lehome_100_demos_no_value",
        ),
    )
)


# Value-free LeHome training in the exact Cosmos Policy ALOHA latent order:
# blank, q, left, right, head, action, future q, future left, future right,
# future head. Demo samples optimize only action; world-model samples optimize
# only the four future-state latents. The action multiplier compensates for the
# four state slots, preserving the requested 75% action / 25% state weighting.
cosmos_predict2_2b_480p_lehome_100_demos_endpoint_dropout50 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_lehome_100_demos_no_value",
            "_self_",
        ],
        trainer=dict(
            callbacks=dict(
                compile_tokenizer=dict(enabled=False),
                device_monitor=dict(log_wandb_table=False),
                wandb=dict(synchronize_ranks_after_step=True),
                wandb_10x=dict(synchronize_ranks_after_step=True),
                every_n_sample_reg=dict(
                    every_n=5000,
                    save_s3=False,
                    use_negative_prompt=False,
                    guidance=[0],
                    num_sampling_step=9,
                    fps=16,
                ),
            ),
            run_validation=True,
            run_validation_on_start=False,
            validation_iter=5000,
            max_val_iter=None,
            max_iter=50000,
            # 6 x 2 accumulation x 8 GPUs = effective global batch 96.
            grad_accum_iter=2,
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=10,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(
                    chunk_duration=37,
                ),
                mask_loss_for_action_future_state_prediction=True,
                # A state sample has four supervised latent frames (6..9)
                # versus one action latent (5). Compensate before applying
                # the 75/25 sample mix so it is also the effective loss mix.
                action_loss_multiplier=4,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=lehome_train_dataset,
            sampler=L(DistributedSampler)(
                dataset=lehome_train_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=6,
            drop_last=True,
        ),
        dataloader_val=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=lehome_validation_dataset,
            sampler=L(DistributedSampler)(
                dataset=lehome_validation_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=False,
                seed=0,
            ),
            batch_size=6,
            drop_last=False,
        ),
        job=dict(
            project="cosmos2b-wam-lehome",
            wandb_mode="online",
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_lehome_100_demos_endpoint_dropout50_prod_v1",
        ),
    )
)


# Paper joint objective, with the requested no-value 75/25 sampling split.
# Policy: p(actions, future state | current state).
# World model: p(future state | current state, ground-truth actions).
# The 75/25 ratio selects conditioning patterns, NOT scalar loss weights.
# All unconditioned latent elements use the original sigma-weighted EDM MSE.
cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_lehome_100_demos_endpoint_dropout50_prod_v1",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                mask_loss_for_action_future_state_prediction=False,
                mask_value_prediction_loss_for_policy_prediction=False,
                mask_current_state_action_for_value_prediction=False,
                mask_future_state_for_qvalue_prediction=False,
                action_loss_multiplier=1,
            ),
        ),
        job=dict(
            name="cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_prod_v2",
        ),
    )
)


# One-second action horizon at the dataset's 30 Hz control rate. This changes
# only the q0-anchored action target length and its corresponding statistics;
# the 10-slot latent layout and model/loss configuration are inherited intact.
lehome_chunk30_stats_path = os.path.join(lehome_data_dir, "lehome_cosmos_stats_chunk30.json")
lehome_chunk30_train_dataset = L(LeHomeDataset)(
    data_dir=lehome_data_dir,
    split="train",
    t5_text_embeddings_path=lehome_t5_embeddings_path,
    stats_path=lehome_chunk30_stats_path,
    chunk_size=30,
    use_image_aug=True,
    use_stronger_image_aug=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    demonstration_sampling_prob=0.75,
    proprio_conditioning_dropout_prob=0.5,
    use_wrist_images=False,
    use_third_person_images=True,
    use_proprio=True,
    return_value_function_returns=False,
)
lehome_chunk30_validation_dataset = L(LeHomeDataset)(
    data_dir=lehome_data_dir,
    split="validation",
    t5_text_embeddings_path=lehome_t5_embeddings_path,
    stats_path=lehome_chunk30_stats_path,
    chunk_size=30,
    use_image_aug=False,
    use_stronger_image_aug=False,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    demonstration_sampling_prob=0.75,
    proprio_conditioning_dropout_prob=0.0,
)

cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_chunk30 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_prod_v2",
            "_self_",
        ],
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=lehome_chunk30_train_dataset,
            sampler=L(DistributedSampler)(
                dataset=lehome_chunk30_train_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=6,
            drop_last=True,
        ),
        dataloader_val=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=lehome_chunk30_validation_dataset,
            sampler=L(DistributedSampler)(
                dataset=lehome_chunk30_validation_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=False,
                seed=0,
            ),
            batch_size=6,
            drop_last=False,
        ),
        job=dict(
            name="cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_chunk30_prod_v3",
        ),
    )
)

cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_chunk30_inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_chunk30_prod_v3",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(sigma_max=80, sigma_min=4),
            ),
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_chunk30_prod_v3_inference",
        ),
    )
)

# One-second action horizon with visual supervision at both 0.5 s and 1.0 s.
# This changes only the input/target layout. It inherits the exact joint EDM
# objective, 75/25 conditioning-pattern sampler, optimizer, scheduler, and
# 50% current-proprio dropout from the proven one-second endpoint experiment.
lehome_dual_endpoint_chunk30_train_dataset = L(LeHomeDataset)(
    data_dir=lehome_data_dir,
    split="train",
    t5_text_embeddings_path=lehome_t5_embeddings_path,
    stats_path=lehome_chunk30_stats_path,
    chunk_size=LEHOME_DUAL_ENDPOINT_ACTION_CHUNK_SIZE,
    use_image_aug=True,
    use_stronger_image_aug=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    demonstration_sampling_prob=0.75,
    proprio_conditioning_dropout_prob=0.5,
    predict_midpoint_images=True,
    use_wrist_images=False,
    use_third_person_images=True,
    use_proprio=True,
    return_value_function_returns=False,
)
lehome_dual_endpoint_chunk30_validation_dataset = L(LeHomeDataset)(
    data_dir=lehome_data_dir,
    split="validation",
    t5_text_embeddings_path=lehome_t5_embeddings_path,
    stats_path=lehome_chunk30_stats_path,
    chunk_size=LEHOME_DUAL_ENDPOINT_ACTION_CHUNK_SIZE,
    use_image_aug=False,
    use_stronger_image_aug=False,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    demonstration_sampling_prob=0.75,
    proprio_conditioning_dropout_prob=0.0,
    predict_midpoint_images=True,
)

cosmos_predict2_2b_480p_lehome_100_demos_dual_endpoint_jointloss_dropout50_chunk30 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_chunk30_prod_v3",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=LEHOME_DUAL_ENDPOINT_STATE_T,
                min_num_conditional_frames=LEHOME_DUAL_ENDPOINT_NUM_CONDITIONAL_LATENTS,
                max_num_conditional_frames=LEHOME_DUAL_ENDPOINT_NUM_CONDITIONAL_LATENTS,
                tokenizer=dict(
                    chunk_duration=LEHOME_DUAL_ENDPOINT_RAW_SEQUENCE_FRAMES,
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=lehome_dual_endpoint_chunk30_train_dataset,
            sampler=L(DistributedSampler)(
                dataset=lehome_dual_endpoint_chunk30_train_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=6,
            drop_last=True,
        ),
        dataloader_val=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=lehome_dual_endpoint_chunk30_validation_dataset,
            sampler=L(DistributedSampler)(
                dataset=lehome_dual_endpoint_chunk30_validation_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=False,
                seed=0,
            ),
            batch_size=6,
            drop_last=False,
        ),
        job=dict(
            name="cosmos_predict2_2b_480p_lehome_100_demos_dual_endpoint_jointloss_dropout50_chunk30_prod_v1",
        ),
    )
)

cosmos_predict2_2b_480p_lehome_100_demos_dual_endpoint_jointloss_dropout50_chunk30_inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_lehome_100_demos_dual_endpoint_jointloss_dropout50_chunk30_prod_v1",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(sde=L(HybridEDMSDE)(sigma_max=80, sigma_min=4)),
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_lehome_100_demos_dual_endpoint_jointloss_dropout50_chunk30_prod_v1_inference",
        ),
    )
)

cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_prod_v2",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(sigma_max=80, sigma_min=4),
            ),
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_prod_v2_inference",
        ),
    )
)


# Corrected LeHome world-action model. Unlike the legacy experiment above,
# this model predicts every one of the 60 future top-camera frames, followed
# by a separate q0-anchored 60x12 action latent. Keep a distinct job name so a
# legacy endpoint-only checkpoint can never be resumed into this layout.
lehome_dense_video_train_dataset = L(LeHomeDataset)(
    data_dir=lehome_data_dir,
    split="train",
    t5_text_embeddings_path=lehome_t5_embeddings_path,
    stats_path=lehome_stats_path,
    chunk_size=LEHOME_ACTION_CHUNK_SIZE,
    use_image_aug=True,
    use_stronger_image_aug=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    demonstration_sampling_prob=0.75,
    proprio_conditioning_dropout_prob=0.5,
    predict_dense_video=True,
    use_wrist_images=False,
    use_third_person_images=True,
    use_proprio=True,
    return_value_function_returns=False,
)
lehome_dense_video_validation_dataset = L(LeHomeDataset)(
    data_dir=lehome_data_dir,
    split="validation",
    t5_text_embeddings_path=lehome_t5_embeddings_path,
    stats_path=lehome_stats_path,
    chunk_size=LEHOME_ACTION_CHUNK_SIZE,
    use_image_aug=False,
    use_stronger_image_aug=False,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    demonstration_sampling_prob=0.75,
    proprio_conditioning_dropout_prob=0.0,
    predict_dense_video=True,
)

cosmos_predict2_2b_480p_lehome_100_demos_dense_video_dropout50 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_lehome_100_demos_no_value",
            "_self_",
        ],
        trainer=dict(
            callbacks=dict(
                # The 81-frame causal VAE repeatedly recompiles under
                # torch.compile on A100, pinning all GPUs for minutes. Eager
                # encoding is stable and already runs at normal throughput.
                compile_tokenizer=dict(enabled=False),
                # W&B 0.23.1 rejects the DeviceMonitor table artifact and then
                # queues subsequent scalar history behind it. Keep scalar GPU
                # telemetry, but do not emit the optional table artifact.
                device_monitor=dict(log_wandb_table=False),
                # Both loss loggers inspect CUDA tensors every step. Keep all
                # ranks out of the next FSDP forward until each logger is done.
                wandb=dict(synchronize_ranks_after_step=True),
                wandb_10x=dict(synchronize_ranks_after_step=True),
                every_n_sample_reg=dict(
                    every_n=5000,
                    save_s3=False,
                    use_negative_prompt=False,
                    guidance=[0],
                    num_sampling_step=9,
                    fps=LEHOME_VIDEO_FPS,
                ),
            ),
            run_validation=True,
            run_validation_on_start=False,
            validation_iter=5000,
            max_val_iter=None,
            max_iter=50000,
            # A microbatch of 12 exceeds a 40 GB A100 during the DiT MLP, and
            # 6 leaves insufficient headroom for a distributed checkpoint.
            # 4 samples x 3 accumulation steps x 8 GPUs preserves the
            # intended effective global batch of 96.
            grad_accum_iter=3,
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=LEHOME_STATE_T,
                min_num_conditional_frames=LEHOME_NUM_CONDITIONAL_LATENTS,
                max_num_conditional_frames=LEHOME_NUM_CONDITIONAL_LATENTS,
                # Fifteen video latents versus one action latent: this keeps
                # the action objective comparable to the full-video objective.
                action_loss_multiplier=15,
                # The 75% demo / 25% world-model sampler below now means
                # action-only versus full-future-video-only optimization.
                mask_loss_for_action_future_state_prediction=True,
                tokenizer=dict(
                    chunk_duration=LEHOME_RAW_SEQUENCE_FRAMES,
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=lehome_dense_video_train_dataset,
            sampler=L(DistributedSampler)(
                dataset=lehome_dense_video_train_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=4,
            drop_last=True,
        ),
        dataloader_val=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=lehome_dense_video_validation_dataset,
            sampler=L(DistributedSampler)(
                dataset=lehome_dense_video_validation_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=False,
                seed=0,
            ),
            batch_size=4,
            drop_last=False,
        ),
        job=dict(
            project="cosmos2b-wam-lehome",
            wandb_mode="online",
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_lehome_100_demos_dense_video_dropout50_prod_v3",
        ),
    )
)


# ALOHA planning model
# Dataset: 648 rollouts from evaluations with Cosmos Policy, pi05, pi0, OpenVLA-OFT+, Diffusion Policy
# NOTE: This rollouts dataset is not released; you will need to replace `rollout_data_dir` below with your own rollouts dataset
aloha_2025_09_18__648_rollouts__cosmos_policy__pi05__pi0__openvla_oft__diffusion_policy__dataset = L(
    ALOHADataset
)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed"),
    t5_text_embeddings_path=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed", "t5_embeddings.pkl"),
    chunk_size=50,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    treat_demos_as_success_rollouts=False,  # Don't include demos as success rollouts because they have a fixed episode length + we want to focus on real policy rollouts
    demonstration_sampling_prob=0.1,  # Smaller demonstration sampling prob - more emphasis on rollouts
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.998,  # Higher gamma for ALOHA because episodes can have up to 1.5-2.0K steps  # (s, a, s', v)
    rollout_data_dir=os.path.join(BASE_DATASETS_DIR, "PATH/TO/YOUR/ROLLOUTS/DATASET"),  # JPEG images
    use_jpeg_for_rollouts=True,  # JPEG images
)
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
            "_self_",
        ],
        checkpoint=dict(
            # Resume from 50K checkpoint of base Cosmos Policy run
            load_path=get_checkpoint_path(
                "hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt"
            ),
        ),
        scheduler=dict(
            # LR decay for 15K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[15000, 100000000000000],
            warm_up_steps=[1500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=aloha_2025_09_18__648_rollouts__cosmos_policy__pi05__pi0__openvla_oft__diffusion_policy__dataset,
            sampler=L(DistributedSampler)(
                dataset=aloha_2025_09_18__648_rollouts__cosmos_policy__pi05__pi0__openvla_oft__diffusion_policy__dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                mask_current_state_action_for_value_prediction=True,  # Use input masking to mask out irrelevant inputs (current state and action) during value prediction
            ),
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func",
        ),
    )
)
# Inference version
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only",
        ),
    )
)


def register_configs():
    cs = ConfigStore.instance()
    # Register the experiments
    for _item in [
        # LIBERO
        cosmos_predict2_2b_480p_libero,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_libero__inference_only,
        # RoboCasa
        cosmos_predict2_2b_480p_robocasa_50_demos_per_task,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference,
        # LeHome
        cosmos_predict2_2b_480p_lehome_100_demos_no_value,
        cosmos_predict2_2b_480p_lehome_100_demos_endpoint_dropout50,
        cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50,
        cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_chunk30,
        cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_chunk30_inference,
        cosmos_predict2_2b_480p_lehome_100_demos_dual_endpoint_jointloss_dropout50_chunk30,
        cosmos_predict2_2b_480p_lehome_100_demos_dual_endpoint_jointloss_dropout50_chunk30_inference,
        cosmos_predict2_2b_480p_lehome_100_demos_endpoint_jointloss_dropout50_inference,
        cosmos_predict2_2b_480p_lehome_100_demos_dense_video_dropout50,
        # ALOHA
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only,
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func,  # ALOHA planning model
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only,
    ]:
        experiment_name = _item["job"]["name"]
        log.info(f"Registering experiment: {experiment_name}")
        cs.store(
            group="experiment",
            package="_global_",
            name=experiment_name,
            node=_item,
        )
