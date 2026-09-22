"""Shared constants and validation for LeHome dense-video policy samples.

The WAN tokenizer maps ``1 + 4 * (T - 1)`` pixel frames to ``T`` latent
frames. LeHome uses the first five latent frames as conditioning, the next
fifteen to represent 60 consecutive future top-camera frames, and the final
latent frame for a 60x12 q0-anchored action chunk.
"""

from __future__ import annotations

import numpy as np

ACTION_DIM = 12
ACTION_CHUNK_SIZE = 60
VIDEO_FPS = 30
TEMPORAL_COMPRESSION_FACTOR = 4

NUM_CONDITIONAL_LATENTS = 5
FUTURE_VIDEO_START_LATENT_IDX = NUM_CONDITIONAL_LATENTS
FUTURE_VIDEO_LATENT_FRAMES = ACTION_CHUNK_SIZE // TEMPORAL_COMPRESSION_FACTOR
FUTURE_VIDEO_END_LATENT_IDX = (
    FUTURE_VIDEO_START_LATENT_IDX + FUTURE_VIDEO_LATENT_FRAMES - 1
)
ACTION_LATENT_IDX = FUTURE_VIDEO_END_LATENT_IDX + 1
STATE_T = ACTION_LATENT_IDX + 1
RAW_SEQUENCE_FRAMES = 1 + TEMPORAL_COMPRESSION_FACTOR * (STATE_T - 1)

# The tokenizer's first latent consumes raw frame zero. Every later latent
# consumes one group of four raw frames.
FUTURE_VIDEO_START_RAW_IDX = (
    1
    + (FUTURE_VIDEO_START_LATENT_IDX - 1) * TEMPORAL_COMPRESSION_FACTOR
)
FUTURE_VIDEO_STOP_RAW_IDX = FUTURE_VIDEO_START_RAW_IDX + ACTION_CHUNK_SIZE


def validate_dense_layout_parameters(
    chunk_size: int,
    temporal_compression_factor: int,
) -> None:
    """Reject settings that would break the fixed 60-frame LeHome contract."""

    if chunk_size != ACTION_CHUNK_SIZE:
        raise ValueError(
            f"Dense LeHome video uses chunk_size={ACTION_CHUNK_SIZE}, got {chunk_size}"
        )
    if temporal_compression_factor != TEMPORAL_COMPRESSION_FACTOR:
        raise ValueError(
            "Dense LeHome video requires temporal compression factor "
            f"{TEMPORAL_COMPRESSION_FACTOR}, got {temporal_compression_factor}"
        )


def build_dense_video_raw_sequence(
    left_current: np.ndarray,
    right_current: np.ndarray,
    top_current: np.ndarray,
    future_top_video: np.ndarray,
) -> np.ndarray:
    """Build the exact 81-frame sequence consumed by the WAN tokenizer.

    Layout in raw frames:

    - 0: tokenizer bootstrap blank
    - 1..4: current-proprio placeholder
    - 5..8: current left camera
    - 9..12: current right camera
    - 13..16: current top camera
    - 17..76: real top-camera frames t+1 through t+60
    - 77..80: action-chunk placeholder
    """

    current_images = (left_current, right_current, top_current)
    expected_shape = np.asarray(top_current).shape
    if len(expected_shape) != 3 or expected_shape[-1] != 3:
        raise ValueError(f"Current RGB images must have shape (H, W, 3), got {expected_shape}")
    for name, image in zip(("left", "right", "top"), current_images):
        array = np.asarray(image)
        if array.shape != expected_shape or array.dtype != np.uint8:
            raise ValueError(
                f"{name} current image must be uint8 with shape {expected_shape}, "
                f"got dtype={array.dtype}, shape={array.shape}"
            )

    future = np.asarray(future_top_video)
    expected_future_shape = (ACTION_CHUNK_SIZE, *expected_shape)
    if future.shape != expected_future_shape or future.dtype != np.uint8:
        raise ValueError(
            "Future top video must contain every t+1..t+60 frame as uint8: "
            f"expected {expected_future_shape}, got dtype={future.dtype}, shape={future.shape}"
        )

    blank = np.zeros_like(top_current)
    repeat = TEMPORAL_COMPRESSION_FACTOR
    sequence = np.concatenate(
        [
            blank[None],
            np.repeat(blank[None], repeat, axis=0),
            np.repeat(left_current[None], repeat, axis=0),
            np.repeat(right_current[None], repeat, axis=0),
            np.repeat(top_current[None], repeat, axis=0),
            future,
            np.repeat(blank[None], repeat, axis=0),
        ],
        axis=0,
    )
    if sequence.shape != (RAW_SEQUENCE_FRAMES, *expected_shape):
        raise AssertionError(
            f"Dense LeHome sequence contract produced {sequence.shape}, "
            f"expected {(RAW_SEQUENCE_FRAMES, *expected_shape)}"
        )
    return np.ascontiguousarray(sequence, dtype=np.uint8)
