"""Shared layout for the one-second LeHome policy with two visual endpoints.

The WAN 2.1 tokenizer maps ``1 + 4 * (T - 1)`` pixel frames to ``T``
latent frames.  This layout keeps the existing action and future-proprio
positions, then predicts all three cameras at t+15 and t+30.
"""

from __future__ import annotations

import numpy as np

ACTION_CHUNK_SIZE = 30
MIDPOINT_STEP = 15
ENDPOINT_STEP = 30
TEMPORAL_COMPRESSION_FACTOR = 4

NUM_CONDITIONAL_LATENTS = 5
ACTION_LATENT_IDX = 5
FUTURE_PROPRIO_LATENT_IDX = 6
MIDPOINT_LEFT_LATENT_IDX = 7
MIDPOINT_RIGHT_LATENT_IDX = 8
MIDPOINT_HEAD_LATENT_IDX = 9
ENDPOINT_LEFT_LATENT_IDX = 10
ENDPOINT_RIGHT_LATENT_IDX = 11
ENDPOINT_HEAD_LATENT_IDX = 12

FUTURE_IMAGE_START_LATENT_IDX = MIDPOINT_LEFT_LATENT_IDX
FUTURE_IMAGE_END_LATENT_IDX = ENDPOINT_HEAD_LATENT_IDX
STATE_T = ENDPOINT_HEAD_LATENT_IDX + 1
RAW_SEQUENCE_FRAMES = 1 + TEMPORAL_COMPRESSION_FACTOR * (STATE_T - 1)


def validate_dual_endpoint_layout_parameters(
    chunk_size: int,
    temporal_compression_factor: int,
) -> None:
    """Reject settings that violate the fixed 30-step/two-endpoint contract."""

    if chunk_size != ACTION_CHUNK_SIZE:
        raise ValueError(
            f"Dual-endpoint LeHome uses chunk_size={ACTION_CHUNK_SIZE}, got {chunk_size}"
        )
    if temporal_compression_factor != TEMPORAL_COMPRESSION_FACTOR:
        raise ValueError(
            "Dual-endpoint LeHome requires temporal compression factor "
            f"{TEMPORAL_COMPRESSION_FACTOR}, got {temporal_compression_factor}"
        )


def build_dual_endpoint_raw_sequence(
    left_current: np.ndarray,
    right_current: np.ndarray,
    head_current: np.ndarray,
    left_midpoint: np.ndarray,
    right_midpoint: np.ndarray,
    head_midpoint: np.ndarray,
    left_endpoint: np.ndarray,
    right_endpoint: np.ndarray,
    head_endpoint: np.ndarray,
) -> np.ndarray:
    """Build the exact 49-frame pseudo-video consumed by the WAN tokenizer.

    Logical latent order:

    0 blank, 1 current q, 2/3/4 current left/right/head, 5 action,
    6 endpoint q, 7/8/9 midpoint left/right/head, and 10/11/12 endpoint
    left/right/head. Numeric slots are blank VAE placeholders and are replaced
    with normalized action/proprio tensors after encoding.
    """

    images = (
        left_current,
        right_current,
        head_current,
        left_midpoint,
        right_midpoint,
        head_midpoint,
        left_endpoint,
        right_endpoint,
        head_endpoint,
    )
    expected_shape = np.asarray(head_current).shape
    if len(expected_shape) != 3 or expected_shape[-1] != 3:
        raise ValueError(
            f"LeHome RGB images must have shape (H, W, 3), got {expected_shape}"
        )
    for image in images:
        array = np.asarray(image)
        if array.shape != expected_shape or array.dtype != np.uint8:
            raise ValueError(
                "All dual-endpoint images must be uint8 with shape "
                f"{expected_shape}, got dtype={array.dtype}, shape={array.shape}"
            )

    blank = np.zeros_like(head_current)
    unique_frames = (
        blank,
        blank,
        left_current,
        right_current,
        head_current,
        blank,
        blank,
        left_midpoint,
        right_midpoint,
        head_midpoint,
        left_endpoint,
        right_endpoint,
        head_endpoint,
    )
    sequence = np.concatenate(
        [
            np.asarray(unique_frames[0])[None],
            *[
                np.repeat(np.asarray(image)[None], TEMPORAL_COMPRESSION_FACTOR, axis=0)
                for image in unique_frames[1:]
            ],
        ],
        axis=0,
    )
    if sequence.shape != (RAW_SEQUENCE_FRAMES, *expected_shape):
        raise AssertionError(
            f"Dual-endpoint sequence produced {sequence.shape}, expected "
            f"{(RAW_SEQUENCE_FRAMES, *expected_shape)}"
        )
    return np.ascontiguousarray(sequence, dtype=np.uint8)
