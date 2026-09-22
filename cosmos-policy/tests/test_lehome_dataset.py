from collections import OrderedDict

import numpy as np
import torch

from cosmos_policy.datasets import lehome_dataset
from cosmos_policy.datasets.dataset_common import calculate_epoch_structure
from cosmos_policy.datasets.lehome_dataset import (
    LeHomeDataset,
    _anchored_delta_chunk,
    _apply_conditioning_dropout,
    _scale_to_unit_range,
)
from cosmos_policy.lehome_dual_endpoint_layout import (
    ACTION_LATENT_IDX as DUAL_ENDPOINT_ACTION_LATENT_IDX,
    ENDPOINT_HEAD_LATENT_IDX,
    ENDPOINT_LEFT_LATENT_IDX,
    ENDPOINT_RIGHT_LATENT_IDX,
    FUTURE_IMAGE_END_LATENT_IDX,
    FUTURE_IMAGE_START_LATENT_IDX,
    FUTURE_PROPRIO_LATENT_IDX,
    MIDPOINT_HEAD_LATENT_IDX,
    MIDPOINT_LEFT_LATENT_IDX,
    MIDPOINT_RIGHT_LATENT_IDX,
    RAW_SEQUENCE_FRAMES as DUAL_ENDPOINT_RAW_SEQUENCE_FRAMES,
    STATE_T as DUAL_ENDPOINT_STATE_T,
    build_dual_endpoint_raw_sequence,
)
from cosmos_policy.lehome_layout import (
    ACTION_CHUNK_SIZE,
    ACTION_LATENT_IDX,
    FUTURE_VIDEO_END_LATENT_IDX,
    FUTURE_VIDEO_START_LATENT_IDX,
    FUTURE_VIDEO_START_RAW_IDX,
    FUTURE_VIDEO_STOP_RAW_IDX,
    NUM_CONDITIONAL_LATENTS,
    RAW_SEQUENCE_FRAMES,
    STATE_T,
    TEMPORAL_COMPRESSION_FACTOR,
    build_dense_video_raw_sequence,
    validate_dense_layout_parameters,
)
from cosmos_policy.models.policy_text2world_model import add_latent_ranges_to_mask


def test_anchored_delta_chunk_starts_at_executable_next_state():
    states = np.arange(6, dtype=np.float32)[:, None]

    chunk = _anchored_delta_chunk(states, anchor=1, chunk_size=3)

    np.testing.assert_array_equal(chunk[:, 0], np.array([1.0, 2.0, 3.0]))


def test_anchored_delta_chunk_clamps_to_final_state():
    states = np.arange(4, dtype=np.float32)[:, None]

    chunk = _anchored_delta_chunk(states, anchor=2, chunk_size=4)

    np.testing.assert_array_equal(chunk[:, 0], np.array([1.0, 1.0, 1.0, 1.0]))


def test_scale_to_unit_range_handles_constant_joints():
    values = np.array([[0.0, 3.0], [5.0, 3.0], [10.0, 3.0]], dtype=np.float32)
    low = np.array([0.0, 3.0], dtype=np.float32)
    high = np.array([10.0, 3.0], dtype=np.float32)

    scaled = _scale_to_unit_range(values, low, high)

    np.testing.assert_allclose(scaled[:, 0], np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(scaled[:, 1], np.zeros(3, dtype=np.float32))


def test_proprio_conditioning_dropout_preserves_or_zeros_vector():
    proprio = np.array([0.25, -0.75], dtype=np.float32)

    kept, kept_was_dropped = _apply_conditioning_dropout(proprio, 0.0)
    dropped, dropped_was_dropped = _apply_conditioning_dropout(proprio, 1.0)

    np.testing.assert_array_equal(kept, proprio)
    assert not kept_was_dropped
    np.testing.assert_array_equal(dropped, np.zeros_like(proprio))
    assert dropped_was_dropped


def test_video_reader_cache_is_bounded_and_closes_lru_reader(monkeypatch):
    class FakeReader:
        def __init__(self, path):
            self.path = path
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(lehome_dataset, "_PyAVVideoReader", FakeReader)
    dataset = object.__new__(LeHomeDataset)
    dataset.max_open_video_readers = 3
    dataset._video_readers = OrderedDict()

    readers = {path: dataset._get_video_reader(path) for path in ("a", "b", "c")}
    assert dataset._get_video_reader("a") is readers["a"]
    dataset._get_video_reader("d")

    assert list(dataset._video_readers) == ["c", "a", "d"]
    assert readers["b"].closed
    assert not readers["a"].closed


def test_dual_endpoint_layout_has_all_three_cameras_at_t15_and_t30():
    shape = (3, 4, 3)
    values = {
        "left_current": 11,
        "right_current": 12,
        "head_current": 13,
        "left_midpoint": 21,
        "right_midpoint": 22,
        "head_midpoint": 23,
        "left_endpoint": 31,
        "right_endpoint": 32,
        "head_endpoint": 33,
    }
    frames = {
        name: np.full(shape, value, dtype=np.uint8)
        for name, value in values.items()
    }

    sequence = build_dual_endpoint_raw_sequence(
        frames["left_current"],
        frames["right_current"],
        frames["head_current"],
        frames["left_midpoint"],
        frames["right_midpoint"],
        frames["head_midpoint"],
        frames["left_endpoint"],
        frames["right_endpoint"],
        frames["head_endpoint"],
    )

    assert sequence.shape == (49, *shape)
    assert DUAL_ENDPOINT_RAW_SEQUENCE_FRAMES == 49
    assert DUAL_ENDPOINT_STATE_T == 13
    assert DUAL_ENDPOINT_ACTION_LATENT_IDX == 5
    assert FUTURE_PROPRIO_LATENT_IDX == 6
    assert (
        MIDPOINT_LEFT_LATENT_IDX,
        MIDPOINT_RIGHT_LATENT_IDX,
        MIDPOINT_HEAD_LATENT_IDX,
    ) == (7, 8, 9)
    assert (
        ENDPOINT_LEFT_LATENT_IDX,
        ENDPOINT_RIGHT_LATENT_IDX,
        ENDPOINT_HEAD_LATENT_IDX,
    ) == (10, 11, 12)
    assert (FUTURE_IMAGE_START_LATENT_IDX, FUTURE_IMAGE_END_LATENT_IDX) == (
        7,
        12,
    )
    expected_groups = [
        (slice(5, 9), frames["left_current"]),
        (slice(9, 13), frames["right_current"]),
        (slice(13, 17), frames["head_current"]),
        (slice(25, 29), frames["left_midpoint"]),
        (slice(29, 33), frames["right_midpoint"]),
        (slice(33, 37), frames["head_midpoint"]),
        (slice(37, 41), frames["left_endpoint"]),
        (slice(41, 45), frames["right_endpoint"]),
        (slice(45, 49), frames["head_endpoint"]),
    ]
    for raw_slice, expected in expected_groups:
        np.testing.assert_array_equal(
            sequence[raw_slice], np.repeat(expected[None], 4, axis=0)
        )
    np.testing.assert_array_equal(sequence[0:5], np.zeros((5, *shape), dtype=np.uint8))
    np.testing.assert_array_equal(sequence[17:25], np.zeros((8, *shape), dtype=np.uint8))


def test_dense_video_layout_has_60_contiguous_future_frames_and_terminal_action():
    shape = (3, 4, 3)
    left = np.full(shape, 11, dtype=np.uint8)
    right = np.full(shape, 22, dtype=np.uint8)
    top = np.full(shape, 33, dtype=np.uint8)
    future = np.stack(
        [np.full(shape, step + 40, dtype=np.uint8) for step in range(ACTION_CHUNK_SIZE)]
    )

    sequence = build_dense_video_raw_sequence(left, right, top, future)

    assert sequence.shape == (RAW_SEQUENCE_FRAMES, *shape)
    assert RAW_SEQUENCE_FRAMES == 81
    assert STATE_T == 21
    assert NUM_CONDITIONAL_LATENTS == 5
    assert FUTURE_VIDEO_START_LATENT_IDX == 5
    assert FUTURE_VIDEO_END_LATENT_IDX == 19
    assert ACTION_LATENT_IDX == 20
    np.testing.assert_array_equal(
        sequence[FUTURE_VIDEO_START_RAW_IDX:FUTURE_VIDEO_STOP_RAW_IDX], future
    )
    np.testing.assert_array_equal(sequence[13:17], np.repeat(top[None], 4, axis=0))
    np.testing.assert_array_equal(sequence[77:81], np.zeros((4, *shape), dtype=np.uint8))


def test_dense_video_layout_rejects_non_native_horizons_and_compression():
    validate_dense_layout_parameters(ACTION_CHUNK_SIZE, TEMPORAL_COMPRESSION_FACTOR)

    for chunk_size, factor in ((59, 4), (60, 8)):
        try:
            validate_dense_layout_parameters(chunk_size, factor)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid dense-video layout was accepted")


def test_epoch_sampler_is_exactly_75_percent_action_and_25_percent_state():
    epoch = calculate_epoch_structure(
        num_steps=100,
        rollout_success_total_steps=100,
        rollout_failure_total_steps=0,
        demonstration_sampling_prob=0.75,
        success_rollout_sampling_prob=1.0,
    )

    assert epoch["adjusted_demo_count"] == 300
    assert epoch["adjusted_success_rollout_count"] == 100
    assert epoch["epoch_length"] == 400


def test_state_loss_mask_covers_all_15_future_video_latents_only():
    mask = torch.zeros((4, STATE_T), dtype=torch.long)
    state_batch_indices = torch.tensor([1, 3], dtype=torch.long)
    starts = torch.full((4,), FUTURE_VIDEO_START_LATENT_IDX, dtype=torch.long)
    ends = torch.full((4,), FUTURE_VIDEO_END_LATENT_IDX, dtype=torch.long)

    result = add_latent_ranges_to_mask(mask, state_batch_indices, starts, ends)

    expected = torch.zeros_like(mask)
    expected[1, FUTURE_VIDEO_START_LATENT_IDX : FUTURE_VIDEO_END_LATENT_IDX + 1] = 1
    expected[3, FUTURE_VIDEO_START_LATENT_IDX : FUTURE_VIDEO_END_LATENT_IDX + 1] = 1
    torch.testing.assert_close(result, expected)
    assert int(result[1].sum()) == 15
    assert result[1, ACTION_LATENT_IDX] == 0
