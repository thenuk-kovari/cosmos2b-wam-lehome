import numpy as np

from cosmos_policy.datasets.lehome_dataset import (
    _anchored_delta_chunk,
    _apply_conditioning_dropout,
    _scale_to_unit_range,
)


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
