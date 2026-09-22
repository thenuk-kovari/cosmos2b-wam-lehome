"""Validate the real selected dataset before endpoint joint-loss training."""

import argparse
import json
import os

import numpy as np
import torch

from cosmos_policy.datasets.lehome_dataset import LeHomeDataset, _apply_conditioning_dropout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=60)
    parser.add_argument("--stats-path", default="")
    parser.add_argument("--predict-midpoint-images", action="store_true")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    torch.manual_seed(42)
    data_dir = os.path.join(os.environ["BASE_DATASETS_DIR"], "lehome")
    common = dict(
        data_dir=data_dir,
        t5_text_embeddings_path=os.environ["LEHOME_T5_EMBEDDINGS_PATH"],
        stats_path=args.stats_path or os.path.join(data_dir, "lehome_cosmos_stats.json"),
        chunk_size=args.chunk_size, demonstration_sampling_prob=0.75,
        use_image_aug=False, use_stronger_image_aug=False,
        predict_midpoint_images=args.predict_midpoint_images,
    )
    train = LeHomeDataset(split="train", proprio_conditioning_dropout_prob=0.5, **common)
    val = LeHomeDataset(split="validation", proprio_conditioning_dropout_prob=0.0, **common)
    assert len(train.episodes) == 100
    assert len(val.episodes) == 12
    ratio = train.adjusted_demo_count / len(train)
    assert abs(ratio - 0.75) < 1e-5
    # Decode both task types, including all three current/future camera views.
    for dataset in (train, val):
        for index, world in ((10, 0), (dataset.adjusted_demo_count + 10, 1)):
            sample = dataset[index]
            expected_raw_frames = 49 if args.predict_midpoint_images else 37
            assert tuple(sample["video"].shape) == (
                3, expected_raw_frames, 224, 224
            )
            assert tuple(sample["actions"].shape) == (args.chunk_size, 12)
            assert sample["world_model_sample_mask"] == world
            assert sample["value_function_sample_mask"] == 0
            assert sample["value_latent_idx"] == -1
            if args.predict_midpoint_images:
                assert sample["midpoint_wrist_image_latent_idx"] == 7
                assert sample["midpoint_wrist_image2_latent_idx"] == 8
                assert sample["midpoint_image_latent_idx"] == 9
                assert sample["future_wrist_image_latent_idx"] == 10
                assert sample["future_wrist_image2_latent_idx"] == 11
                assert sample["future_image_latent_idx"] == 12
                assert sample["future_video_start_latent_idx"] == 7
                assert sample["future_video_end_latent_idx"] == 12
            else:
                assert sample["future_wrist_image_latent_idx"] == 7
                assert sample["future_wrist_image2_latent_idx"] == 8
                assert sample["future_image_latent_idx"] == 9
            assert np.isfinite(sample["actions"]).all()
            if dataset is val:
                assert not sample["proprio_conditioning_dropped"]
            # Independently invert normalization and check the fixed-q0 target.
            states = dataset.episodes[0]["states"]
            stats = dataset.dataset_stats
            decoded_delta = (sample["actions"] + 1) / 2 * (
                stats["actions_max"] - stats["actions_min"]
            ) + stats["actions_min"]
            expected_indices = np.minimum(
                np.arange(11, 11 + args.chunk_size), len(states) - 1
            )
            expected = states[expected_indices] - states[10]
            np.testing.assert_allclose(decoded_delta, expected, atol=1e-6)
    dropped = sum(_apply_conditioning_dropout(np.ones(12), 0.5)[1] for _ in range(10000))
    assert 4700 < dropped < 5300
    print("ENDPOINT_DATA_PREFLIGHT " + json.dumps({
        "status": "passed", "training_episodes": len(train.episodes),
        "validation_episodes": len(val.episodes), "policy_sampling_fraction": ratio,
        "proprio_dropout_empirical_fraction": dropped / 10000,
        "raw_frames": 49 if args.predict_midpoint_images else 37,
        "latent_slots": 13 if args.predict_midpoint_images else 10,
        "action_shape": [args.chunk_size, 12],
        "q0_target_roundtrip": "passed", "cameras": ["left", "right", "head"],
        "visual_target_steps": [15, 30] if args.predict_midpoint_images else [args.chunk_size],
    }), flush=True)


if __name__ == "__main__":
    main()
