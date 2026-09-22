import base64

import numpy as np
import pytest

from cosmos_policy.experiments.robot.lehome.inference import (
    ACTION_DIM,
    CHUNK_SIZE,
    LeHomeInferenceConfig,
    anchor_action_chunk,
    prepare_model_observation,
)
from cosmos_policy.experiments.robot.lehome.server import deserialize_observation

def test_anchor_action_chunk_adds_the_same_q0_to_every_timestep():
    q0 = np.arange(ACTION_DIM, dtype=np.float32)
    deltas = np.stack(
        [np.full(ACTION_DIM, step, dtype=np.float32) for step in range(CHUNK_SIZE)]
    )

    actions = anchor_action_chunk(q0, deltas)

    np.testing.assert_allclose(actions[0], q0)
    np.testing.assert_allclose(actions[10], q0 + 10)
    np.testing.assert_allclose(actions[-1], q0 + CHUNK_SIZE - 1)


def test_anchor_action_chunk_can_expose_a_shorter_prediction_horizon():
    q0 = np.zeros(ACTION_DIM, dtype=np.float32)
    deltas = np.ones((CHUNK_SIZE, ACTION_DIM), dtype=np.float32)

    actions = anchor_action_chunk(q0, deltas, prediction_steps=50)

    assert actions.shape == (50, ACTION_DIM)
    assert actions.dtype == np.float32


def test_anchor_action_chunk_supports_native_30_step_checkpoint():
    q0 = np.arange(ACTION_DIM, dtype=np.float32)
    deltas = np.ones((30, ACTION_DIM), dtype=np.float32)

    actions = anchor_action_chunk(
        q0, deltas, prediction_steps=30, chunk_size=30
    )

    assert actions.shape == (30, ACTION_DIM)
    expected = np.broadcast_to(q0[None, :] + 1, actions.shape)
    np.testing.assert_allclose(actions, expected)


def test_native_30_step_inference_config_is_valid():
    config = LeHomeInferenceConfig(
        ckpt_path="checkpoint",
        dataset_stats_path="stats.json",
        t5_text_embeddings_path="embeddings.pkl",
        task_label="fold the shirt",
        chunk_size=30,
        prediction_steps=30,
        execute_steps=30,
    )

    config.validate()


def test_dual_endpoint_30_step_inference_config_is_valid():
    config = LeHomeInferenceConfig(
        ckpt_path="checkpoint",
        dataset_stats_path="stats.json",
        t5_text_embeddings_path="embeddings.pkl",
        task_label="fold the shirt",
        config=(
            "cosmos_predict2_2b_480p_lehome_100_demos_"
            "dual_endpoint_jointloss_dropout50_chunk30_prod_v1_inference"
        ),
        chunk_size=30,
        prediction_steps=30,
        execute_steps=30,
        predict_midpoint_images=True,
    )

    config.validate()
    assert config.predict_midpoint_images


def test_anchor_action_chunk_rejects_a_wrong_model_output_shape():
    with pytest.raises(ValueError, match="predicted delta chunk"):
        anchor_action_chunk(
            np.zeros(ACTION_DIM, dtype=np.float32),
            np.zeros((CHUNK_SIZE - 1, ACTION_DIM), dtype=np.float32),
        )


def test_receding_horizon_config_accepts_50_predictions_and_8_executions():
    config = LeHomeInferenceConfig(
        ckpt_path="checkpoint",
        dataset_stats_path="stats.json",
        t5_text_embeddings_path="embeddings.pkl",
        task_label="fold the shirt",
        prediction_steps=50,
        execute_steps=8,
    )

    config.validate()


def test_inference_defaults_to_endpoint_three_camera_contract():
    config = LeHomeInferenceConfig(
        ckpt_path="checkpoint",
        dataset_stats_path="stats.json",
        t5_text_embeddings_path="embeddings.pkl",
        task_label="fold the shirt",
    )

    config.validate()

    assert not config.predict_dense_video
    assert config.config == (
        "cosmos_predict2_2b_480p_lehome_100_demos_endpoint_dropout50_prod_v1"
    )


def test_receding_horizon_config_rejects_execution_beyond_prediction():
    config = LeHomeInferenceConfig(
        ckpt_path="checkpoint",
        dataset_stats_path="stats.json",
        t5_text_embeddings_path="embeddings.pkl",
        task_label="fold the shirt",
        prediction_steps=50,
        execute_steps=51,
    )

    with pytest.raises(ValueError, match="execute_steps"):
        config.validate()


def test_prepare_model_observation_maps_the_three_simulator_cameras():
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    observation = {
        "observation.state": np.arange(ACTION_DIM, dtype=np.float32),
        "observation.images.left_rgb": image,
        "observation.images.right_rgb": image,
        "observation.images.top_rgb": image,
    }

    model_observation = prepare_model_observation(observation)

    assert set(model_observation) == {
        "proprio",
        "left_wrist_image",
        "right_wrist_image",
        "primary_image",
    }
    np.testing.assert_array_equal(
        model_observation["proprio"], observation["observation.state"]
    )


def test_deserialize_observation_decodes_docker_policy_arrays():
    array = np.arange(ACTION_DIM, dtype=np.float32)
    wire_value = {
        "base64": base64.b64encode(array.tobytes()).decode("ascii"),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }

    observation = deserialize_observation({"observation.state": wire_value})

    np.testing.assert_array_equal(observation["observation.state"], array)


def test_deserialize_observation_rejects_inconsistent_byte_length():
    wire_value = {
        "base64": base64.b64encode(b"short").decode("ascii"),
        "shape": [ACTION_DIM],
        "dtype": "float32",
    }

    with pytest.raises(ValueError, match="Byte length mismatch"):
        deserialize_observation({"observation.state": wire_value})
