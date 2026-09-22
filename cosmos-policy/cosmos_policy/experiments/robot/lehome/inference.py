"""Checkpoint inference for the LeHome Cosmos Policy model.

The model predicts normalized q0-relative joint deltas. This adapter uses the
same preprocessing and diffusion sampler as the upstream Cosmos evaluators,
then converts the unnormalized deltas to absolute LeHome joint targets.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cosmos_policy.lehome_dual_endpoint_layout import (
    ACTION_LATENT_IDX as DUAL_ENDPOINT_ACTION_LATENT_IDX,
    NUM_CONDITIONAL_LATENTS as DUAL_ENDPOINT_NUM_CONDITIONAL_LATENTS,
    STATE_T as DUAL_ENDPOINT_STATE_T,
)
from cosmos_policy.lehome_layout import (
    ACTION_CHUNK_SIZE,
    ACTION_DIM,
    VIDEO_FPS,
)

CHUNK_SIZE = ACTION_CHUNK_SIZE
ENDPOINT_ACTION_LATENT_IDX = 5
ENDPOINT_STATE_T = 10
ENDPOINT_NUM_CONDITIONAL_LATENTS = 5

_STATE_KEY = "observation.state"
_CAMERA_KEYS = {
    "left_wrist_image": "observation.images.left_rgb",
    "right_wrist_image": "observation.images.right_rgb",
    "primary_image": "observation.images.top_rgb",
}


@dataclass
class LeHomeInferenceConfig:
    """Runtime options consumed by the shared Cosmos Policy sampler."""

    ckpt_path: str
    dataset_stats_path: str
    t5_text_embeddings_path: str
    task_label: str

    config: str = "cosmos_predict2_2b_480p_lehome_100_demos_endpoint_dropout50_prod_v1"
    config_file: str = "cosmos_policy/config/config.py"
    chunk_size: int = CHUNK_SIZE
    action_dim: int = ACTION_DIM
    prediction_steps: int = CHUNK_SIZE
    execute_steps: int = CHUNK_SIZE
    num_denoising_steps_action: int = 5
    seed: int = 7
    randomize_seed: bool = False
    generated_video_output_dir: str | None = None
    # Deprecated compatibility alias used by the earlier endpoint exporter.
    future_image_output_dir: str | None = None
    attention_output_dir: str | None = None
    attention_layers: tuple[int, ...] = (0, 7, 14, 21, 27)
    attention_query_samples: int = 32
    run_attention_ablations: bool = True

    # These fields describe the exact modality layout used by LeHomeDataset.
    suite: str = "lehome"
    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = True
    num_wrist_images: int = 2
    use_proprio: bool = True
    normalize_proprio: bool = True
    unnormalize_actions: bool = True
    use_value_prediction: bool = False
    use_variance_scale: bool = False
    use_jpeg_compression: bool = False
    trained_with_image_aug: bool = True
    predict_dense_video: bool = False
    predict_midpoint_images: bool = False

    def validate(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.action_dim != ACTION_DIM:
            raise ValueError(f"LeHome checkpoints use action_dim={ACTION_DIM}, got {self.action_dim}")
        if not 1 <= self.prediction_steps <= self.chunk_size:
            raise ValueError(
                f"prediction_steps must be in [1, {self.chunk_size}], "
                f"got {self.prediction_steps}"
            )
        if not 1 <= self.execute_steps <= self.prediction_steps:
            raise ValueError(
                f"execute_steps must be in [1, {self.prediction_steps}], "
                f"got {self.execute_steps}"
            )
        if self.num_denoising_steps_action < 1:
            raise ValueError("num_denoising_steps_action must be positive")
        if self.attention_query_samples < 1:
            raise ValueError("attention_query_samples must be positive")
        if not self.task_label.strip():
            raise ValueError("task_label must not be empty")
        if self.generated_video_output_dir and self.future_image_output_dir:
            raise ValueError(
                "Set only generated_video_output_dir; future_image_output_dir is a deprecated alias"
            )


def prepare_model_observation(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Validate a LeHome simulator observation and map it to Cosmos names."""

    if _STATE_KEY not in observation:
        raise KeyError(f"Missing required observation key: {_STATE_KEY}")

    state = np.asarray(observation[_STATE_KEY], dtype=np.float32).reshape(-1)
    if state.shape != (ACTION_DIM,):
        raise ValueError(f"{_STATE_KEY} must have shape ({ACTION_DIM},), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError(f"{_STATE_KEY} contains NaN or infinity")

    model_observation: dict[str, np.ndarray] = {"proprio": state}
    image_shapes = set()
    for model_key, simulator_key in _CAMERA_KEYS.items():
        if simulator_key not in observation:
            raise KeyError(f"Missing required observation key: {simulator_key}")
        image = np.asarray(observation[simulator_key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{simulator_key} must have shape (H, W, 3), got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"{simulator_key} must have dtype uint8, got {image.dtype}")
        image_shapes.add(image.shape)
        model_observation[model_key] = image

    if len(image_shapes) != 1:
        raise ValueError(f"All RGB cameras must have the same shape, got {sorted(image_shapes)}")
    return model_observation


def anchor_action_chunk(
    q0: np.ndarray,
    predicted_delta_chunk: np.ndarray,
    prediction_steps: int = CHUNK_SIZE,
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """Convert a prefix of one q0-relative model chunk to an absolute plan.

    The same q0 is used for every row. In particular, this is not a cumulative
    sum of deltas and it does not re-anchor each row at the previous action.
    ``prediction_steps`` may shorten the exposed plan, but the checkpoint is
    still sampled at its native ``CHUNK_SIZE`` model horizon.
    """

    anchor = np.asarray(q0, dtype=np.float32).reshape(-1)
    deltas = np.asarray(predicted_delta_chunk, dtype=np.float32)
    if anchor.shape != (ACTION_DIM,):
        raise ValueError(f"q0 must have shape ({ACTION_DIM},), got {anchor.shape}")
    if deltas.shape != (chunk_size, ACTION_DIM):
        raise ValueError(
            f"predicted delta chunk must have shape ({chunk_size}, {ACTION_DIM}), "
            f"got {deltas.shape}"
        )
    if not 1 <= prediction_steps <= chunk_size:
        raise ValueError(
            f"prediction_steps must be in [1, {chunk_size}], got {prediction_steps}"
        )
    if not np.isfinite(anchor).all() or not np.isfinite(deltas).all():
        raise ValueError("q0 and predicted deltas must contain only finite values")

    absolute = anchor[None, :] + deltas
    return np.ascontiguousarray(absolute[:prediction_steps], dtype=np.float32)


class LeHomeCosmosPolicy:
    """Load one Cosmos checkpoint and answer LeHome action-chunk queries."""

    def __init__(self, config: LeHomeInferenceConfig) -> None:
        config.validate()
        checkpoint_path = Path(config.ckpt_path)
        if checkpoint_path.is_dir() and (checkpoint_path / "model" / ".metadata").is_file():
            config.ckpt_path = str(checkpoint_path / "model")
        self.config = config

        # Keep heavyweight Cosmos imports lazy so pure protocol/unit tests do
        # not need a CUDA environment.
        import torch
        from cosmos_policy.experiments.robot import cosmos_utils

        if not torch.cuda.is_available():
            raise RuntimeError("Cosmos Policy inference requires a CUDA GPU")
        torch.cuda.set_device(0)
        torch.set_grad_enabled(False)
        self._torch = torch

        self.dataset_stats = cosmos_utils.load_dataset_stats(config.dataset_stats_path)
        self.task_embedding = self._load_task_embedding(torch)
        self._validate_stats()

        self.model, cosmos_config = cosmos_utils.get_model(config)
        training_chunk_size = int(cosmos_config.dataloader_train.dataset.chunk_size)
        if training_chunk_size != config.chunk_size:
            raise ValueError(
                "Checkpoint/config chunk-size mismatch: "
                f"training config uses {training_chunk_size}, inference uses {config.chunk_size}"
            )
        expected_state_t = (
            DUAL_ENDPOINT_STATE_T
            if config.predict_midpoint_images
            else ENDPOINT_STATE_T
        )
        expected_conditional_latents = (
            DUAL_ENDPOINT_NUM_CONDITIONAL_LATENTS
            if config.predict_midpoint_images
            else ENDPOINT_NUM_CONDITIONAL_LATENTS
        )
        if (
            self.model.config.state_t != expected_state_t
            or self.model.config.min_num_conditional_frames
            != expected_conditional_latents
        ):
            raise ValueError(
                "Expected the configured LeHome latent layout "
                f"(state_t={expected_state_t}, min_num_conditional_frames="
                f"{expected_conditional_latents}), got "
                f"state_t={self.model.config.state_t}, "
                f"min_num_conditional_frames={self.model.config.min_num_conditional_frames}"
            )
        if config.predict_dense_video:
            raise ValueError("Endpoint checkpoints require predict_dense_video=False")

        self._get_action = cosmos_utils.get_action
        self._cosmos_utils = cosmos_utils
        output_dir = config.generated_video_output_dir or config.future_image_output_dir
        self.generated_video_output_dir = Path(output_dir) if output_dir else None
        if self.generated_video_output_dir is not None:
            self.generated_video_output_dir.mkdir(parents=True, exist_ok=True)
        self.attention_diagnostics = None
        if config.attention_output_dir:
            from .attention_diagnostics import LeHomeAttentionDiagnostics

            self.attention_diagnostics = LeHomeAttentionDiagnostics(
                self.model,
                output_dir=config.attention_output_dir,
                layers=config.attention_layers,
                query_samples=config.attention_query_samples,
            )
            print(
                "[LeHomeCosmosPolicy] attention diagnostics enabled: "
                f"output={config.attention_output_dir} layers={config.attention_layers}",
                flush=True,
            )
        self._query_count = 0
        self.last_planned_actions = np.empty((0, ACTION_DIM), dtype=np.float32)

    def _load_task_embedding(self, torch_module: Any) -> np.ndarray:
        embeddings_path = Path(self.config.t5_text_embeddings_path)
        if embeddings_path.suffix == ".npz":
            with np.load(embeddings_path, allow_pickle=False) as archive:
                stored_label = str(archive["task_label"].item())
                if stored_label != self.config.task_label:
                    raise KeyError(
                        f"Task {self.config.task_label!r} does not match the NPZ task label "
                        f"{stored_label!r}"
                    )
                embedding = archive["embedding"]
        else:
            with embeddings_path.open("rb") as file:
                embeddings = pickle.load(file)
            if self.config.task_label not in embeddings:
                raise KeyError(
                    f"Task {self.config.task_label!r} is not present in "
                    f"{self.config.t5_text_embeddings_path}"
                )
            embedding = embeddings[self.config.task_label]

        if isinstance(embedding, torch_module.Tensor):
            embedding = embedding.detach().to(torch_module.float32).cpu().numpy()
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.shape == (512, 1024):
            embedding = embedding[None, ...]
        if embedding.shape != (1, 512, 1024):
            raise ValueError(
                "Expected a T5 embedding with shape (1, 512, 1024), "
                f"got {embedding.shape}"
            )
        return embedding

    def _validate_stats(self) -> None:
        for key in ("proprio_min", "proprio_max", "actions_min", "actions_max"):
            if key not in self.dataset_stats:
                raise KeyError(f"Dataset statistics are missing {key!r}")
            value = np.asarray(self.dataset_stats[key])
            if value.shape != (ACTION_DIM,):
                raise ValueError(
                    f"Dataset statistic {key!r} must have shape ({ACTION_DIM},), got {value.shape}"
                )

        if "chunk_size" in self.dataset_stats:
            stats_chunk_size = int(np.asarray(self.dataset_stats["chunk_size"]).item())
            if stats_chunk_size != self.config.chunk_size:
                raise ValueError(
                    f"Statistics use chunk_size={stats_chunk_size}, "
                    f"but inference uses {self.config.chunk_size}"
                )

    def reset(self) -> None:
        self._query_count = 0
        self.last_planned_actions = np.empty((0, ACTION_DIM), dtype=np.float32)
        if self.attention_diagnostics is not None:
            self.attention_diagnostics.start_run()

    def _save_generated_future_endpoints(
        self,
        result: Mapping[str, Any],
        model_observation: Mapping[str, Any],
    ) -> None:
        """Save generated t+15/t+30 cameras, or only t+30 for legacy runs."""

        if self.generated_video_output_dir is None:
            return

        indices = result["latent_indices"]
        expected_action_idx = (
            DUAL_ENDPOINT_ACTION_LATENT_IDX
            if self.config.predict_midpoint_images
            else ENDPOINT_ACTION_LATENT_IDX
        )
        if indices["action_latent_idx"] != expected_action_idx:
            raise ValueError(
                f"Generated action is at latent {indices['action_latent_idx']}, "
                f"expected {expected_action_idx}"
            )
        predictions = result.get("future_image_predictions")
        if not isinstance(predictions, Mapping):
            raise ValueError("Sampler did not return future endpoint images")
        prediction_keys = {
            "future_wrist_image": "predicted_future_left_rgb",
            "future_wrist_image2": "predicted_future_right_rgb",
            "future_image": "predicted_future_head_rgb",
        }
        if self.config.predict_midpoint_images:
            prediction_keys.update(
                {
                    "midpoint_wrist_image": "predicted_midpoint_left_rgb",
                    "midpoint_wrist_image2": "predicted_midpoint_right_rgb",
                    "midpoint_image": "predicted_midpoint_head_rgb",
                }
            )
        output_images: dict[str, np.ndarray] = {}
        for source_key, output_key in prediction_keys.items():
            if source_key not in predictions:
                raise ValueError(f"Sampler omitted required endpoint image {source_key!r}")
            image = np.asarray(predictions[source_key], dtype=np.uint8)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"Generated endpoint {source_key!r} must have shape (H, W, 3), "
                    f"got {image.shape}"
                )
            output_images[output_key] = image

        output_path = (
            self.generated_video_output_dir
            / f"chunk_{self._query_count + 1:03d}_visuals.npz"
        )
        np.savez_compressed(
            output_path,
            current_left_rgb=np.asarray(
                model_observation["left_wrist_image"], dtype=np.uint8
            ),
            current_right_rgb=np.asarray(
                model_observation["right_wrist_image"], dtype=np.uint8
            ),
            current_head_rgb=np.asarray(
                model_observation["primary_image"], dtype=np.uint8
            ),
            **output_images,
            horizon_steps=np.asarray(self.config.chunk_size, dtype=np.int64),
            video_fps=np.asarray(VIDEO_FPS, dtype=np.int64),
        )
        print(
            "[LeHomeCosmosPolicy] saved generated left/right/head "
            f"{'midpoint and endpoint' if self.config.predict_midpoint_images else 'endpoint'} images: "
            f"{output_path}",
            flush=True,
        )

    def _sample_deltas(self, model_observation: Mapping[str, Any]) -> np.ndarray:
        result = self._get_action(
            self.config,
            model=self.model,
            dataset_stats=self.dataset_stats,
            obs=dict(model_observation),
            task_label_or_embedding=self.task_embedding,
            seed=self.config.seed,
            randomize_seed=self.config.randomize_seed,
            num_denoising_steps_action=self.config.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=True,
            batch_size=1,
        )
        self._save_generated_future_endpoints(result, model_observation)
        return np.asarray(result["actions"], dtype=np.float32)

    @staticmethod
    def _ablation_metrics(baseline: np.ndarray, ablated: np.ndarray) -> dict[str, float]:
        baseline_flat = baseline.reshape(-1).astype(np.float64)
        ablated_flat = ablated.reshape(-1).astype(np.float64)
        difference = ablated_flat - baseline_flat
        baseline_norm = np.linalg.norm(baseline_flat)
        denominator = max(baseline_norm * np.linalg.norm(ablated_flat), 1e-12)
        return {
            "delta_rmse": float(np.sqrt(np.mean(np.square(difference)))),
            "relative_l2": float(np.linalg.norm(difference) / max(baseline_norm, 1e-12)),
            "cosine_similarity": float(np.dot(baseline_flat, ablated_flat) / denominator),
            "max_abs_change": float(np.max(np.abs(difference))),
        }

    def _run_first_query_ablations(
        self,
        model_observation: Mapping[str, np.ndarray],
        baseline_deltas: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        diagnostics = self.attention_diagnostics
        if diagnostics is None or not self.config.run_attention_ablations:
            return {}

        proprio_min = np.asarray(self.dataset_stats["proprio_min"], dtype=np.float32)
        proprio_max = np.asarray(self.dataset_stats["proprio_max"], dtype=np.float32)
        no_proprio = {key: value.copy() for key, value in model_observation.items()}
        # This maps exactly to normalized zero, matching the new training dropout.
        no_proprio["proprio"] = (proprio_min + proprio_max) / 2.0

        neutral_vision = {key: value.copy() for key, value in model_observation.items()}
        for key in _CAMERA_KEYS:
            neutral_vision[key] = np.full_like(neutral_vision[key], 127, dtype=np.uint8)

        diagnostics.enabled = False
        try:
            no_proprio_deltas = self._sample_deltas(no_proprio)
            neutral_vision_deltas = self._sample_deltas(neutral_vision)
        finally:
            diagnostics.enabled = True
        return {
            "normalized_zero_proprio": self._ablation_metrics(baseline_deltas, no_proprio_deltas),
            "neutral_gray_vision": self._ablation_metrics(baseline_deltas, neutral_vision_deltas),
        }

    def infer(self, observation: Mapping[str, Any]) -> list[np.ndarray]:
        """Predict, unnormalize, q0-anchor, and return executable actions."""

        model_observation = prepare_model_observation(observation)
        q0 = model_observation["proprio"].copy()
        if self.attention_diagnostics is not None:
            self.attention_diagnostics.start_query(model_observation)

        started_at = time.perf_counter()
        predicted_deltas = self._sample_deltas(model_observation)
        ablations = {}
        if self._query_count == 0:
            ablations = self._run_first_query_ablations(model_observation, predicted_deltas)
        planned_actions = anchor_action_chunk(
            q0,
            predicted_deltas,
            prediction_steps=self.config.prediction_steps,
            chunk_size=self.config.chunk_size,
        )
        self.last_planned_actions = planned_actions
        executable_actions = np.ascontiguousarray(
            planned_actions[: self.config.execute_steps], dtype=np.float32
        )
        if self.attention_diagnostics is not None:
            self.attention_diagnostics.end_query(q0, predicted_deltas, ablations)

        self._query_count += 1
        elapsed = time.perf_counter() - started_at
        print(
            f"[LeHomeCosmosPolicy] query={self._query_count} "
            f"inference_s={elapsed:.3f} model_steps={self.config.chunk_size} "
            f"prediction_steps={len(planned_actions)} "
            f"execute_steps={len(executable_actions)}",
            flush=True,
        )
        return [action for action in executable_actions]
