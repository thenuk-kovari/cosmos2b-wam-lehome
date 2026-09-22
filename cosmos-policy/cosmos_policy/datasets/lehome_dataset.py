"""Native LeHome loader for Cosmos Policy training.

The source data follows the LeRobot v3 layout: proprioception is stored in
Parquet shards while each camera is one concatenated MP4 per garment variant.
This loader reads only the selected episodes and constructs executable
q0-anchored joint-delta chunks. Its dense-video mode supplies every future
top-camera frame aligned with the 60-step action horizon.
"""

from __future__ import annotations

import json
import os
import pickle
from collections import OrderedDict
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from filelock import FileLock
from torch.utils.data import Dataset

from cosmos_policy.datasets.dataset_common import calculate_epoch_structure, determine_sample_type
from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.lehome_dual_endpoint_layout import (
    ACTION_LATENT_IDX as DUAL_ENDPOINT_ACTION_LATENT_IDX,
    ENDPOINT_HEAD_LATENT_IDX,
    ENDPOINT_LEFT_LATENT_IDX,
    ENDPOINT_RIGHT_LATENT_IDX,
    FUTURE_IMAGE_END_LATENT_IDX,
    FUTURE_IMAGE_START_LATENT_IDX,
    FUTURE_PROPRIO_LATENT_IDX as DUAL_ENDPOINT_FUTURE_PROPRIO_LATENT_IDX,
    MIDPOINT_HEAD_LATENT_IDX,
    MIDPOINT_LEFT_LATENT_IDX,
    MIDPOINT_RIGHT_LATENT_IDX,
    MIDPOINT_STEP,
    RAW_SEQUENCE_FRAMES as DUAL_ENDPOINT_RAW_SEQUENCE_FRAMES,
    build_dual_endpoint_raw_sequence,
    validate_dual_endpoint_layout_parameters,
)
from cosmos_policy.lehome_layout import (
    ACTION_CHUNK_SIZE,
    ACTION_LATENT_IDX,
    FUTURE_VIDEO_END_LATENT_IDX,
    FUTURE_VIDEO_START_LATENT_IDX,
    STATE_T,
    VIDEO_FPS,
    build_dense_video_raw_sequence,
    validate_dense_layout_parameters,
)

TASK_PROMPTS = {
    "record_pant_long_release_10": "fold the pants",
    "record_pant_short_release_10": "fold the shorts",
    "record_top_long_release_10": "fold the shirt",
    "record_top_short_release_10": "fold the t-shirt",
}

CAMERA_PATHS = {
    "left": "observation.images.left_rgb",
    "right": "observation.images.right_rgb",
    "top": "observation.images.top_rgb",
}

TRAIN_VARIANTS = ("001", "002", "003", "004", "005")
VALIDATION_VARIANTS = ("001", "003", "005")
TRAIN_EPISODES = (0, 1, 2, 3, 4)
VALIDATION_EPISODES = (24,)


class _PyAVVideoReader:
    """Small seekable reader for LeHome's AV1-in-MP4 camera streams."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        # libdav1d otherwise sizes a decoder thread pool from the host CPU
        # count for every open stream. With distributed DataLoader workers,
        # that multiplies into hundreds of threads and large decoder heaps.
        self.stream.codec_context.thread_count = 1
        rate = self.stream.average_rate or self.stream.guessed_rate
        if rate is None or self.stream.time_base is None:
            raise RuntimeError(f"Missing frame-rate metadata in {path}")
        self.fps = float(rate)
        self.time_base = float(self.stream.time_base)
        self.start_pts = self.stream.start_time or 0
        self.length = int(self.stream.frames)
        if self.length <= 0 and self.stream.duration is not None:
            self.length = int(round(self.stream.duration * self.time_base * self.fps))
        if self.length <= 0:
            raise RuntimeError(f"Missing frame-count metadata in {path}")

    def __len__(self) -> int:
        return self.length

    def close(self) -> None:
        container = getattr(self, "container", None)
        if container is not None:
            container.close()
            self.container = None

    def __del__(self) -> None:
        self.close()

    def get_batch(self, indices: list[int]) -> np.ndarray:
        requested = [int(index) for index in indices]
        if not requested:
            raise ValueError("At least one video frame index is required")
        if min(requested) < 0 or max(requested) >= self.length:
            raise IndexError(f"Video indices {requested} are out of range for {self.path} ({self.length} frames)")

        first_index = min(requested)
        last_index = max(requested)
        seek_pts = self.start_pts + round(first_index / self.fps / self.time_base)
        self.container.seek(seek_pts, stream=self.stream, backward=True, any_frame=False)

        wanted = set(requested)
        decoded = {}
        for frame in self.container.decode(self.stream):
            if frame.pts is None:
                continue
            frame_index = round((frame.pts - self.start_pts) * self.time_base * self.fps)
            if frame_index in wanted:
                decoded[frame_index] = frame.to_ndarray(format="rgb24")
            if frame_index >= last_index and wanted.issubset(decoded):
                break

        missing = wanted - decoded.keys()
        if missing:
            raise RuntimeError(f"Could not decode video indices {sorted(missing)} from {self.path}")
        return np.stack([decoded[index] for index in requested])


def _scale_to_unit_range(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Min/max scale per joint to [-1, 1], mapping constant joints to zero."""
    span = high - low
    safe_span = np.where(span > 1e-8, span, 1.0)
    scaled = 2.0 * ((values - low) / safe_span) - 1.0
    scaled[..., span <= 1e-8] = 0.0
    return scaled.astype(np.float32, copy=False)


def _anchored_delta_chunk(states: np.ndarray, anchor: int, chunk_size: int) -> np.ndarray:
    """Return q[t+j]-q[t] for j=1..K, clamping j at the episode end."""
    future_indices = np.minimum(anchor + np.arange(1, chunk_size + 1), len(states) - 1)
    return (states[future_indices] - states[anchor]).astype(np.float32, copy=False)


def _apply_conditioning_dropout(values: np.ndarray, probability: float) -> tuple[np.ndarray, bool]:
    """Randomly replace a conditioning vector with zeros."""
    dropped = probability > 0.0 and torch.rand(()).item() < probability
    return (np.zeros_like(values), True) if dropped else (values, False)


class LeHomeDataset(Dataset):
    """Selected LeHome episodes in endpoint, dual-endpoint, or dense layout."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        chunk_size: int = 60,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        stats_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        num_duplicates_per_image: int = 4,
        demonstration_sampling_prob: float = 0.75,
        proprio_conditioning_dropout_prob: float = 0.0,
        predict_dense_video: bool = False,
        predict_midpoint_images: bool = False,
        max_open_video_readers: int = 3,
        debug: bool = False,
        # Compatibility options inherited when Hydra recursively merges this
        # dataset over the LIBERO experiment. LeHome always uses proprio and
        # its three fixed cameras, and does not load rollout/value targets.
        use_wrist_images: bool = False,
        use_third_person_images: bool = True,
        use_proprio: bool = True,
        rollout_data_dir: str = "",
        success_rollout_sampling_prob: float = 0.5,
        treat_success_rollouts_as_demos: bool = False,
        return_value_function_returns: bool = False,
        gamma: float = 0.99,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError(f"split must be 'train' or 'validation', got {split!r}")
        if not 0.0 < demonstration_sampling_prob < 1.0:
            raise ValueError("demonstration_sampling_prob must be strictly between 0 and 1")
        if not 0.0 <= proprio_conditioning_dropout_prob <= 1.0:
            raise ValueError("proprio_conditioning_dropout_prob must be between 0 and 1")
        if predict_dense_video and predict_midpoint_images:
            raise ValueError(
                "predict_dense_video and predict_midpoint_images are mutually exclusive"
            )
        if predict_dense_video:
            validate_dense_layout_parameters(chunk_size, num_duplicates_per_image)
        if predict_midpoint_images:
            validate_dual_endpoint_layout_parameters(
                chunk_size, num_duplicates_per_image
            )
        if max_open_video_readers < 3:
            raise ValueError("max_open_video_readers must be at least 3 for the three LeHome cameras")

        self.data_dir = Path(data_dir)
        self.split = split
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.num_duplicates_per_image = num_duplicates_per_image
        self.demonstration_sampling_prob = demonstration_sampling_prob
        self.proprio_conditioning_dropout_prob = proprio_conditioning_dropout_prob
        self.predict_dense_video = predict_dense_video
        self.predict_midpoint_images = predict_midpoint_images
        self.max_open_video_readers = max_open_video_readers
        self.debug = debug
        self._video_readers: OrderedDict[str, _PyAVVideoReader] = OrderedDict()

        # These arguments deliberately have no effect for LeHome. Keeping them
        # in the signature makes the adapter safe to instantiate from the
        # recursively composed Cosmos Policy experiment config.
        del (
            use_wrist_images,
            use_third_person_images,
            use_proprio,
            rollout_data_dir,
            success_rollout_sampling_prob,
            treat_success_rollouts_as_demos,
            return_value_function_returns,
            gamma,
        )

        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"LeHome data directory does not exist: {self.data_dir}")

        variants = TRAIN_VARIANTS if split == "train" else VALIDATION_VARIANTS
        episode_indices = TRAIN_EPISODES if split == "train" else VALIDATION_EPISODES
        self.episodes = self._load_episodes(variants, episode_indices)
        if not self.episodes:
            raise RuntimeError(f"No {split} episodes found under {self.data_dir}")

        self._step_to_episode_map: list[tuple[int, int]] = []
        for episode_idx, episode in enumerate(self.episodes):
            self._step_to_episode_map.extend((episode_idx, i) for i in range(episode["num_steps"]))
        self.num_steps = len(self._step_to_episode_map)

        # The same successful demonstrations supply both objectives. Oversample the
        # demo view to 75%, with the rollout view becoming world-model samples.
        epoch = calculate_epoch_structure(
            num_steps=self.num_steps,
            rollout_success_total_steps=self.num_steps,
            rollout_failure_total_steps=0,
            demonstration_sampling_prob=self.demonstration_sampling_prob,
            success_rollout_sampling_prob=1.0,
        )
        self.adjusted_demo_count = epoch["adjusted_demo_count"]
        self.adjusted_success_rollout_count = epoch["adjusted_success_rollout_count"]
        self.epoch_length = epoch["epoch_length"]

        self.stats_path = Path(stats_path) if stats_path else self.data_dir / "lehome_cosmos_stats.json"
        self.dataset_stats = self._load_or_create_stats()

        self.t5_text_embeddings = None
        if t5_text_embeddings_path:
            embeddings_path = Path(t5_text_embeddings_path)
            if not embeddings_path.is_file():
                raise FileNotFoundError(f"T5 embeddings file does not exist: {embeddings_path}")
            with embeddings_path.open("rb") as file:
                self.t5_text_embeddings = pickle.load(file)
            missing = set(TASK_PROMPTS.values()) - set(self.t5_text_embeddings)
            if missing:
                raise KeyError(f"T5 embeddings are missing prompts: {sorted(missing)}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video_readers"] = OrderedDict()
        return state

    def __del__(self) -> None:
        for reader in getattr(self, "_video_readers", {}).values():
            reader.close()

    def _load_episodes(self, variants: tuple[str, ...], episode_indices: tuple[int, ...]) -> list[dict]:
        episodes = []
        required_columns = ["observation.state", "episode_index", "frame_index", "index"]

        for task, command in TASK_PROMPTS.items():
            for variant in variants:
                variant_dir = self.data_dir / task / variant
                parquet_files = sorted((variant_dir / "data").glob("chunk-*/*.parquet"))
                if not parquet_files:
                    raise FileNotFoundError(f"No data Parquet files found under {variant_dir}")

                states_parts = []
                episode_parts = []
                frame_parts = []
                index_parts = []
                for parquet_path in parquet_files:
                    table = pq.read_table(parquet_path, columns=required_columns)
                    states_parts.append(np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32))
                    episode_parts.append(table.column("episode_index").to_numpy(zero_copy_only=False))
                    frame_parts.append(table.column("frame_index").to_numpy(zero_copy_only=False))
                    index_parts.append(table.column("index").to_numpy(zero_copy_only=False))

                all_states = np.concatenate(states_parts)
                all_episode_indices = np.concatenate(episode_parts).astype(np.int64, copy=False)
                all_frame_indices = np.concatenate(frame_parts).astype(np.int64, copy=False)
                all_video_indices = np.concatenate(index_parts).astype(np.int64, copy=False)

                video_paths = {}
                for camera_name, source_key in CAMERA_PATHS.items():
                    candidates = sorted((variant_dir / "videos" / source_key).glob("chunk-*/*.mp4"))
                    if len(candidates) != 1:
                        raise RuntimeError(
                            f"Expected one concatenated video for {task}/{variant}/{source_key}, found {len(candidates)}"
                        )
                    video_paths[camera_name] = str(candidates[0])

                for source_episode_idx in episode_indices:
                    selected = np.flatnonzero(all_episode_indices == source_episode_idx)
                    if selected.size == 0:
                        raise RuntimeError(f"Episode {source_episode_idx} missing from {task}/{variant}")
                    order = np.argsort(all_frame_indices[selected])
                    rows = selected[order]
                    expected_frames = np.arange(len(rows), dtype=np.int64)
                    if not np.array_equal(all_frame_indices[rows], expected_frames):
                        raise RuntimeError(
                            f"Non-contiguous frame_index values in {task}/{variant}/{source_episode_idx}"
                        )

                    episode_states = all_states[rows]
                    if episode_states.ndim != 2 or episode_states.shape[1] != 12:
                        raise RuntimeError(
                            f"Expected 12D observation.state in {task}/{variant}, got {episode_states.shape}"
                        )
                    episodes.append(
                        {
                            "task": task,
                            "variant": variant,
                            "source_episode_idx": source_episode_idx,
                            "command": command,
                            "states": episode_states,
                            "video_indices": all_video_indices[rows],
                            "video_paths": video_paths,
                            "num_steps": len(rows),
                        }
                    )

        return episodes

    def _load_or_create_stats(self) -> dict[str, np.ndarray]:
        lock_path = f"{self.stats_path}.lock"
        with FileLock(lock_path):
            if not self.stats_path.exists():
                if self.split != "train":
                    raise FileNotFoundError(
                        f"Training-only normalization statistics must be created before validation: {self.stats_path}"
                    )
                self.stats_path.parent.mkdir(parents=True, exist_ok=True)
                raw_stats = self._compute_training_stats()
                payload = {
                    "chunk_size": self.chunk_size,
                    "proprio_min": raw_stats["proprio_min"].tolist(),
                    "proprio_max": raw_stats["proprio_max"].tolist(),
                    "actions_min": raw_stats["actions_min"].tolist(),
                    "actions_max": raw_stats["actions_max"].tolist(),
                }
                temp_path = Path(f"{self.stats_path}.tmp.{os.getpid()}")
                temp_path.write_text(json.dumps(payload, indent=2) + "\n")
                os.replace(temp_path, self.stats_path)

            payload = json.loads(self.stats_path.read_text())

        if payload.get("chunk_size") != self.chunk_size:
            raise ValueError(
                f"Stats at {self.stats_path} use chunk_size={payload.get('chunk_size')}, expected {self.chunk_size}"
            )
        return {
            key: np.asarray(payload[key], dtype=np.float32)
            for key in ("proprio_min", "proprio_max", "actions_min", "actions_max")
        }

    def _compute_training_stats(self) -> dict[str, np.ndarray]:
        all_states = np.concatenate([episode["states"] for episode in self.episodes])
        proprio_min = all_states.min(axis=0)
        proprio_max = all_states.max(axis=0)
        actions_min = np.full(all_states.shape[1], np.inf, dtype=np.float32)
        actions_max = np.full(all_states.shape[1], -np.inf, dtype=np.float32)

        for episode in self.episodes:
            states = episode["states"]
            anchors = np.arange(len(states))
            for offset in range(1, self.chunk_size + 1):
                future = states[np.minimum(anchors + offset, len(states) - 1)]
                deltas = future - states
                actions_min = np.minimum(actions_min, deltas.min(axis=0))
                actions_max = np.maximum(actions_max, deltas.max(axis=0))

        return {
            "proprio_min": proprio_min,
            "proprio_max": proprio_max,
            "actions_min": actions_min,
            "actions_max": actions_max,
        }

    def _get_video_reader(self, path: str) -> _PyAVVideoReader:
        reader = self._video_readers.pop(path, None)
        if reader is None:
            while len(self._video_readers) >= self.max_open_video_readers:
                _, stale_reader = self._video_readers.popitem(last=False)
                stale_reader.close()
            reader = _PyAVVideoReader(path)
        self._video_readers[path] = reader
        return reader

    def _read_current_and_future_frames(self, episode: dict, current: int, future: int) -> dict[str, np.ndarray]:
        source_indices = [int(episode["video_indices"][current]), int(episode["video_indices"][future])]
        result = {}
        for camera_name, path in episode["video_paths"].items():
            reader = self._get_video_reader(path)
            if source_indices[-1] >= len(reader):
                raise IndexError(f"Video index {source_indices[-1]} is out of range for {path} ({len(reader)} frames)")
            decoded = reader.get_batch(source_indices)
            result[f"{camera_name}_current"] = decoded[0]
            result[f"{camera_name}_future"] = decoded[1]
        return result

    def _read_current_midpoint_and_future_frames(
        self,
        episode: dict,
        current: int,
        midpoint: int,
        future: int,
    ) -> dict[str, np.ndarray]:
        """Decode aligned current, t+15, and t+30 frames from all cameras."""

        relative_indices = (current, midpoint, future)
        source_indices = [
            int(episode["video_indices"][index]) for index in relative_indices
        ]
        result: dict[str, np.ndarray] = {}
        for camera_name, path in episode["video_paths"].items():
            reader = self._get_video_reader(path)
            if max(source_indices) >= len(reader):
                raise IndexError(
                    f"Video index {max(source_indices)} is out of range for "
                    f"{path} ({len(reader)} frames)"
                )
            decoded = reader.get_batch(source_indices)
            result[f"{camera_name}_current"] = decoded[0]
            result[f"{camera_name}_midpoint"] = decoded[1]
            result[f"{camera_name}_future"] = decoded[2]
        return result

    def _read_current_frames_and_future_top_video(
        self,
        episode: dict,
        current: int,
    ) -> dict[str, np.ndarray]:
        """Decode current three-camera context plus top frames t+1..t+60."""

        future_relative_indices = np.minimum(
            current + np.arange(1, ACTION_CHUNK_SIZE + 1),
            episode["num_steps"] - 1,
        )
        current_source_idx = int(episode["video_indices"][current])
        result: dict[str, np.ndarray] = {}
        for camera_name in ("left", "right"):
            path = episode["video_paths"][camera_name]
            reader = self._get_video_reader(path)
            if current_source_idx >= len(reader):
                raise IndexError(
                    f"Video index {current_source_idx} is out of range for {path} ({len(reader)} frames)"
                )
            result[f"{camera_name}_current"] = reader.get_batch([current_source_idx])[0]

        top_path = episode["video_paths"]["top"]
        top_reader = self._get_video_reader(top_path)
        top_source_indices = [current_source_idx] + [
            int(episode["video_indices"][index]) for index in future_relative_indices
        ]
        if max(top_source_indices) >= len(top_reader):
            raise IndexError(
                f"Video index {max(top_source_indices)} is out of range for "
                f"{top_path} ({len(top_reader)} frames)"
            )
        top_frames = top_reader.get_batch(top_source_indices)
        result["top_current"] = top_frames[0]
        result["top_future_video"] = top_frames[1:]
        if result["top_future_video"].shape[0] != ACTION_CHUNK_SIZE:
            raise AssertionError(
                "Dense LeHome decoding must return exactly "
                f"{ACTION_CHUNK_SIZE} future frames"
            )
        return result

    def __len__(self) -> int:
        return 1 if self.debug else self.epoch_length

    def __getitem__(self, idx: int) -> dict:
        if self.debug:
            idx = 0
        sample_type = determine_sample_type(idx, self.adjusted_demo_count, self.adjusted_success_rollout_count)
        if sample_type == "demo":
            base_step_idx = idx % self.num_steps
            is_world_model_sample = False
            global_rollout_idx = -1
        else:
            base_step_idx = (idx - self.adjusted_demo_count) % self.num_steps
            is_world_model_sample = True
            global_rollout_idx = base_step_idx

        episode_idx, relative_step_idx = self._step_to_episode_map[base_step_idx]
        episode = self.episodes[episode_idx]
        states = episode["states"]
        future_frame_idx = min(relative_step_idx + self.chunk_size, len(states) - 1)
        midpoint_frame_idx = min(
            relative_step_idx + MIDPOINT_STEP, len(states) - 1
        )
        next_relative_step_idx = future_frame_idx

        action_chunk = _anchored_delta_chunk(states, relative_step_idx, self.chunk_size)
        next_action_chunk = _anchored_delta_chunk(states, next_relative_step_idx, self.chunk_size)
        proprio = states[relative_step_idx]
        future_proprio = states[future_frame_idx]

        if self.normalize_actions:
            action_chunk = _scale_to_unit_range(
                action_chunk, self.dataset_stats["actions_min"], self.dataset_stats["actions_max"]
            )
            next_action_chunk = _scale_to_unit_range(
                next_action_chunk, self.dataset_stats["actions_min"], self.dataset_stats["actions_max"]
            )
        if self.normalize_proprio:
            proprio = _scale_to_unit_range(
                proprio, self.dataset_stats["proprio_min"], self.dataset_stats["proprio_max"]
            )
            future_proprio = _scale_to_unit_range(
                future_proprio, self.dataset_stats["proprio_min"], self.dataset_stats["proprio_max"]
            )

        # Drop only the current proprio conditioning vector. Action and future
        # proprio targets remain unchanged. Validation configures this as zero.
        proprio, proprio_conditioning_dropped = _apply_conditioning_dropout(
            proprio, self.proprio_conditioning_dropout_prob
        )

        if self.predict_dense_video:
            decoded = self._read_current_frames_and_future_top_video(
                episode, relative_step_idx
            )
            raw_frames = build_dense_video_raw_sequence(
                decoded["left_current"],
                decoded["right_current"],
                decoded["top_current"],
                decoded["top_future_video"],
            )
            # preprocess_image samples one spatial/color transform and applies
            # it to the whole sequence, preserving temporal motion coherence.
            all_images = preprocess_image(
                raw_frames,
                final_image_size=self.final_image_size,
                normalize_images=self.normalize_images,
                use_image_aug=self.use_image_aug,
                stronger_image_aug=self.use_stronger_image_aug,
            )
            expected_raw_frames = 1 + self.num_duplicates_per_image * (STATE_T - 1)
            if all_images.shape[1] != expected_raw_frames:
                raise AssertionError(
                    f"Dense LeHome sample has {all_images.shape[1]} raw frames, "
                    f"expected {expected_raw_frames}"
                )
        elif self.predict_midpoint_images:
            decoded = self._read_current_midpoint_and_future_frames(
                episode,
                relative_step_idx,
                midpoint_frame_idx,
                future_frame_idx,
            )
            raw_frames = build_dual_endpoint_raw_sequence(
                decoded["left_current"],
                decoded["right_current"],
                decoded["top_current"],
                decoded["left_midpoint"],
                decoded["right_midpoint"],
                decoded["top_midpoint"],
                decoded["left_future"],
                decoded["right_future"],
                decoded["top_future"],
            )
            all_images = preprocess_image(
                raw_frames,
                final_image_size=self.final_image_size,
                normalize_images=self.normalize_images,
                use_image_aug=self.use_image_aug,
                stronger_image_aug=self.use_stronger_image_aug,
            )
            if all_images.shape[1] != DUAL_ENDPOINT_RAW_SEQUENCE_FRAMES:
                raise AssertionError(
                    f"Dual-endpoint LeHome sample has {all_images.shape[1]} "
                    f"raw frames, expected {DUAL_ENDPOINT_RAW_SEQUENCE_FRAMES}"
                )
        else:
            decoded = self._read_current_and_future_frames(
                episode, relative_step_idx, future_frame_idx
            )
            blank = np.zeros_like(decoded["top_current"])
            unique_frames = np.stack(
                [
                    blank,
                    blank,
                    decoded["left_current"],
                    decoded["right_current"],
                    decoded["top_current"],
                    blank,
                    blank,
                    decoded["left_future"],
                    decoded["right_future"],
                    decoded["top_future"],
                ]
            ).astype(np.uint8, copy=False)
            unique_frames = preprocess_image(
                unique_frames,
                final_image_size=self.final_image_size,
                normalize_images=self.normalize_images,
                use_image_aug=self.use_image_aug,
                stronger_image_aug=self.use_stronger_image_aug,
            )
            repeats = torch.tensor([1] + [self.num_duplicates_per_image] * 9, dtype=torch.long)
            all_images = torch.repeat_interleave(unique_frames, repeats, dim=1)

        if self.t5_text_embeddings is None:
            raise RuntimeError("LeHome T5 embeddings are required before training or sampling the dataset")
        text_embedding = torch.squeeze(self.t5_text_embeddings[episode["command"]])

        if self.predict_dense_video:
            latent_indices = {
                "action_latent_idx": ACTION_LATENT_IDX,
                "value_latent_idx": -1,
                "current_proprio_latent_idx": 1,
                "current_wrist_image_latent_idx": 2,
                "current_wrist_image2_latent_idx": 3,
                "current_image_latent_idx": 4,
                "future_proprio_latent_idx": -1,
                "future_wrist_image_latent_idx": -1,
                "future_wrist_image2_latent_idx": -1,
                # The legacy metric key points at the final predicted video
                # latent. Diffusion loss still covers all slots 5 through 19.
                "future_image_latent_idx": FUTURE_VIDEO_END_LATENT_IDX,
                "future_video_start_latent_idx": FUTURE_VIDEO_START_LATENT_IDX,
                "future_video_end_latent_idx": FUTURE_VIDEO_END_LATENT_IDX,
            }
        elif self.predict_midpoint_images:
            latent_indices = {
                "action_latent_idx": DUAL_ENDPOINT_ACTION_LATENT_IDX,
                "value_latent_idx": -1,
                "current_proprio_latent_idx": 1,
                "current_wrist_image_latent_idx": 2,
                "current_wrist_image2_latent_idx": 3,
                "current_image_latent_idx": 4,
                "future_proprio_latent_idx": DUAL_ENDPOINT_FUTURE_PROPRIO_LATENT_IDX,
                "future_wrist_image_latent_idx": ENDPOINT_LEFT_LATENT_IDX,
                "future_wrist_image2_latent_idx": ENDPOINT_RIGHT_LATENT_IDX,
                "future_image_latent_idx": ENDPOINT_HEAD_LATENT_IDX,
                "midpoint_wrist_image_latent_idx": MIDPOINT_LEFT_LATENT_IDX,
                "midpoint_wrist_image2_latent_idx": MIDPOINT_RIGHT_LATENT_IDX,
                "midpoint_image_latent_idx": MIDPOINT_HEAD_LATENT_IDX,
                "future_video_start_latent_idx": FUTURE_IMAGE_START_LATENT_IDX,
                "future_video_end_latent_idx": FUTURE_IMAGE_END_LATENT_IDX,
            }
        else:
            latent_indices = {
                "action_latent_idx": 5,
                "value_latent_idx": -1,
                "current_proprio_latent_idx": 1,
                "current_wrist_image_latent_idx": 2,
                "current_wrist_image2_latent_idx": 3,
                "current_image_latent_idx": 4,
                "future_proprio_latent_idx": 6,
                "future_wrist_image_latent_idx": 7,
                "future_wrist_image2_latent_idx": 8,
                "future_image_latent_idx": 9,
                "future_video_start_latent_idx": -1,
                "future_video_end_latent_idx": -1,
            }

        return {
            "video": all_images,
            "command": episode["command"],
            "actions": action_chunk,
            "t5_text_embeddings": text_embedding,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": VIDEO_FPS if (self.predict_dense_video or self.predict_midpoint_images) else 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": proprio,
            "proprio_conditioning_dropped": proprio_conditioning_dropped,
            "future_proprio": future_proprio,
            "__key__": idx,
            "value_function_return": 0.0,
            "next_action_chunk": next_action_chunk,
            "next_value_function_return": 0.0,
            "rollout_data_mask": 0 if sample_type == "demo" else 1,
            "rollout_data_success_mask": 1 if sample_type != "demo" else 0,
            "world_model_sample_mask": 1 if is_world_model_sample else 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": global_rollout_idx,
            **latent_indices,
        }
