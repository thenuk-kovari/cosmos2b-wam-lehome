"""GPU preflight for the LeHome dense-video training contract.

This intentionally uses a real dataset sample and the production Wan VAE.  It
is a launch gate for the long training run, not a synthetic unit test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cosmos_policy.datasets.lehome_dataset import LeHomeDataset
from cosmos_policy.lehome_layout import (
    ACTION_CHUNK_SIZE,
    ACTION_LATENT_IDX,
    FUTURE_VIDEO_END_LATENT_IDX,
    FUTURE_VIDEO_START_LATENT_IDX,
    FUTURE_VIDEO_START_RAW_IDX,
    FUTURE_VIDEO_STOP_RAW_IDX,
    RAW_SEQUENCE_FRAMES,
    STATE_T,
)
from cosmos_policy.models.policy_text2world_model import add_latent_ranges_to_mask
from cosmos_policy.tokenizers.wan2pt1 import Wan2pt1VAEInterface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--t5-embeddings-path", required=True)
    parser.add_argument(
        "--vae-path",
        default="hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.data_dir, args.stats_path, args.t5_embeddings_path):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    dataset = LeHomeDataset(
        data_dir=args.data_dir,
        split="validation",
        chunk_size=ACTION_CHUNK_SIZE,
        final_image_size=224,
        t5_text_embeddings_path=args.t5_embeddings_path,
        stats_path=args.stats_path,
        normalize_images=False,
        normalize_actions=True,
        normalize_proprio=True,
        use_image_aug=False,
        use_stronger_image_aug=False,
        num_duplicates_per_image=4,
        demonstration_sampling_prob=0.75,
        proprio_conditioning_dropout_prob=0.0,
        predict_dense_video=True,
    )
    sample = dataset[0]
    video = sample["video"]
    assert tuple(video.shape) == (3, RAW_SEQUENCE_FRAMES, 224, 224), video.shape
    assert tuple(sample["actions"].shape) == (ACTION_CHUNK_SIZE, 12)
    assert sample["action_latent_idx"] == ACTION_LATENT_IDX
    assert sample["future_video_start_latent_idx"] == FUTURE_VIDEO_START_LATENT_IDX
    assert sample["future_video_end_latent_idx"] == FUTURE_VIDEO_END_LATENT_IDX

    future_video = video[:, FUTURE_VIDEO_START_RAW_IDX:FUTURE_VIDEO_STOP_RAW_IDX]
    assert future_video.shape[1] == ACTION_CHUNK_SIZE
    future_video_float = future_video.float()
    adjacent_motion = (
        future_video_float[:, 1:] - future_video_float[:, :-1]
    ).abs().mean().item()
    endpoint_motion = (
        future_video_float[:, -1] - future_video_float[:, 0]
    ).abs().mean().item()
    if adjacent_motion <= 0.0 or endpoint_motion <= 0.0:
        raise AssertionError(
            "Real future-video target is static; refusing to launch training: "
            f"adjacent_motion={adjacent_motion}, endpoint_motion={endpoint_motion}"
        )

    tokenizer = Wan2pt1VAEInterface(
        chunk_duration=RAW_SEQUENCE_FRAMES,
        vae_pth=args.vae_path,
        temporal_window=16,
        is_parallel=False,
    )
    latent = tokenizer.encode(video.unsqueeze(0).cuda(non_blocking=True))
    assert latent.ndim == 5
    assert tuple(latent.shape[:3]) == (1, 16, STATE_T), latent.shape
    future_latent = latent[
        :, :, FUTURE_VIDEO_START_LATENT_IDX : FUTURE_VIDEO_END_LATENT_IDX + 1
    ]
    assert future_latent.shape[2] == 15
    latent_temporal_change = (future_latent[:, :, 1:] - future_latent[:, :, :-1]).abs().mean().item()
    if latent_temporal_change <= 0.0:
        raise AssertionError("VAE produced a static future-video latent target")

    loss_mask = torch.zeros((2, STATE_T), dtype=torch.long, device=latent.device)
    loss_mask = add_latent_ranges_to_mask(
        loss_mask,
        batch_indices=torch.tensor([1], device=latent.device),
        start_indices=torch.full((2,), FUTURE_VIDEO_START_LATENT_IDX, device=latent.device),
        end_indices=torch.full((2,), FUTURE_VIDEO_END_LATENT_IDX, device=latent.device),
    )
    assert int(loss_mask[0].sum()) == 0
    assert int(loss_mask[1].sum()) == 15
    assert int(loss_mask[1, ACTION_LATENT_IDX]) == 0

    print(
        json.dumps(
            {
                "status": "ok",
                "raw_video_shape": list(video.shape),
                "future_raw_frame_range": [
                    FUTURE_VIDEO_START_RAW_IDX,
                    FUTURE_VIDEO_STOP_RAW_IDX - 1,
                ],
                "future_raw_frame_count": int(future_video.shape[1]),
                "vae_latent_shape": list(latent.shape),
                "future_latent_range": [
                    FUTURE_VIDEO_START_LATENT_IDX,
                    FUTURE_VIDEO_END_LATENT_IDX,
                ],
                "future_latent_count": int(future_latent.shape[2]),
                "world_model_loss_latent_count": int(loss_mask[1].sum()),
                "raw_adjacent_motion": adjacent_motion,
                "raw_endpoint_motion": endpoint_motion,
                "latent_temporal_change": latent_temporal_change,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
