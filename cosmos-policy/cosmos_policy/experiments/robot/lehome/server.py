"""HTTP bridge between LeHome Isaac Sim and Cosmos Policy inference."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import numpy as np

from .inference import CHUNK_SIZE, LeHomeCosmosPolicy, LeHomeInferenceConfig

_MAX_REQUEST_BYTES = 64 * 1024 * 1024


def deserialize_observation(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    """Decode the wire format produced by scripts.eval_policy.DockerPolicy."""

    observation: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and {"base64", "shape", "dtype"} <= value.keys():
            try:
                dtype = np.dtype(value["dtype"])
                shape = tuple(int(dimension) for dimension in value["shape"])
                encoded = value["base64"].encode("ascii")
                buffer = base64.b64decode(encoded, validate=True)
            except (TypeError, ValueError, UnicodeEncodeError, binascii.Error) as error:
                raise ValueError(f"Invalid encoded array for {key!r}: {error}") from error

            if dtype.hasobject:
                raise ValueError(f"Object dtype is not allowed for {key!r}")
            if not shape or any(dimension < 0 for dimension in shape):
                raise ValueError(f"Invalid shape for {key!r}: {shape}")
            expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if len(buffer) != expected_bytes:
                raise ValueError(
                    f"Byte length mismatch for {key!r}: expected {expected_bytes}, got {len(buffer)}"
                )
            observation[key] = np.frombuffer(buffer, dtype=dtype).reshape(shape)
        elif isinstance(value, list):
            observation[key] = np.asarray(value, dtype=np.float32)
    return observation


class _PolicyHTTPServer(HTTPServer):
    def __init__(self, address: tuple[str, int], policy: LeHomeCosmosPolicy):
        super().__init__(address, _PolicyRequestHandler)
        self.policy = policy


class _PolicyRequestHandler(BaseHTTPRequestHandler):
    server: _PolicyHTTPServer

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_request(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length < 0 or content_length > _MAX_REQUEST_BYTES:
            raise ValueError(
                f"Request body must be between 0 and {_MAX_REQUEST_BYTES} bytes"
            )
        body = self.rfile.read(content_length) if content_length else b"{}"
        request = json.loads(body)
        if not isinstance(request, dict):
            raise ValueError("Request body must be a JSON object")
        return request

    def do_GET(self) -> None:
        if self.path == "/health":
            config = self.server.policy.config
            self._send_json(
                200,
                {
                    "status": "ok",
                    "model_steps": config.chunk_size,
                    "prediction_steps": config.prediction_steps,
                    "execute_steps": config.execute_steps,
                    "generated_endpoint_cameras": 3,
                },
            )
        else:
            self._send_json(404, {"error": f"Unknown endpoint: {self.path}"})

    def do_POST(self) -> None:
        try:
            request = self._read_request()
            if self.path == "/reset":
                self.server.policy.reset()
                self._send_json(200, {"status": "ok"})
            elif self.path == "/infer":
                observation = deserialize_observation(request)
                actions = self.server.policy.infer(observation)
                self._send_json(
                    200,
                    {
                        # DockerPolicy executes this prefix and requeries when
                        # its cache is empty.
                        "actions": [action.tolist() for action in actions],
                        # Keep the complete planning prefix observable without
                        # asking the simulator to execute all of it.
                        "predicted_actions": self.server.policy.last_planned_actions.tolist(),
                        "model_steps": self.server.policy.config.chunk_size,
                        "prediction_steps": self.server.policy.config.prediction_steps,
                        "execute_steps": self.server.policy.config.execute_steps,
                    },
                )
            else:
                self._send_json(404, {"error": f"Unknown endpoint: {self.path}"})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": str(error)})
        except Exception as error:
            traceback.print_exc()
            self._send_json(500, {"error": f"Inference failed: {error}"})

    def log_message(self, format_string: str, *args: Any) -> None:
        print(
            f"[LeHomeCosmosServer] {self.address_string()} "
            f"{format_string % args}",
            flush=True,
        )


def run_server(
    policy: LeHomeCosmosPolicy,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    server = _PolicyHTTPServer((host, port), policy)
    print(f"[LeHomeCosmosServer] listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[LeHomeCosmosServer] shutting down", flush=True)
    finally:
        server.server_close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a LeHome Cosmos Policy checkpoint to the LeHome simulator"
    )
    parser.add_argument("--ckpt-path", required=True, help="Checkpoint directory or .pt file")
    parser.add_argument(
        "--dataset-stats-path",
        required=True,
        help="lehome_cosmos_stats.json generated by the training dataset",
    )
    parser.add_argument(
        "--t5-text-embeddings-path",
        required=True,
        help="Safe NPZ or trusted pickle containing the training task prompt embeddings",
    )
    parser.add_argument(
        "--task-label",
        required=True,
        help='Training prompt, e.g. "fold the shirt"',
    )
    parser.add_argument(
        "--config",
        default="cosmos_predict2_2b_480p_lehome_100_demos_endpoint_dropout50_prod_v1",
        help="Cosmos experiment config name",
    )
    parser.add_argument(
        "--config-file",
        default="cosmos_policy/config/config.py",
        help="Cosmos config router",
    )
    parser.add_argument(
        "--num-denoising-steps",
        type=int,
        default=5,
        help="Diffusion denoising steps per policy query",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Native action horizon used to train this checkpoint",
    )
    parser.add_argument(
        "--prediction-steps",
        type=int,
        default=None,
        help=(
            "Number of leading targets to expose as the predicted plan. The "
            "checkpoint is always sampled at its native chunk-size horizon."
        ),
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=None,
        help=(
            "Number of predictions to return before the simulator requeries. "
            "Default executes the complete native chunk."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--randomize-seed",
        action="store_true",
        help="Use a fresh sampling seed for each action query",
    )
    parser.add_argument(
        "--generated-video-output-dir",
        default=None,
        help=(
            "Optional directory for the generated horizon-end left wrist, "
            "right wrist, and head images from every action query"
        ),
    )
    parser.add_argument(
        "--future-image-output-dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trained-with-image-aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the deterministic test crop corresponding to training augmentation",
    )
    parser.add_argument(
        "--predict-midpoint-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the 13-slot LeHome layout with camera targets at t+15 and t+30",
    )
    parser.add_argument(
        "--attention-output-dir",
        default=None,
        help="Optional directory for action-query attention diagnostics",
    )
    parser.add_argument(
        "--attention-layers",
        default="0,7,14,21,27",
        help="Comma-separated transformer block indices to inspect",
    )
    parser.add_argument("--attention-query-samples", type=int, default=32)
    parser.add_argument(
        "--attention-ablations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare first-query deltas with normalized-zero proprio and neutral-gray vision",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = LeHomeInferenceConfig(
        ckpt_path=args.ckpt_path,
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        task_label=args.task_label,
        config=args.config,
        config_file=args.config_file,
        chunk_size=args.chunk_size,
        prediction_steps=args.prediction_steps or args.chunk_size,
        execute_steps=args.execute_steps or args.chunk_size,
        num_denoising_steps_action=args.num_denoising_steps,
        seed=args.seed,
        randomize_seed=args.randomize_seed,
        generated_video_output_dir=args.generated_video_output_dir,
        future_image_output_dir=args.future_image_output_dir,
        trained_with_image_aug=args.trained_with_image_aug,
        predict_midpoint_images=args.predict_midpoint_images,
        attention_output_dir=args.attention_output_dir,
        attention_layers=tuple(int(value) for value in args.attention_layers.split(",") if value),
        attention_query_samples=args.attention_query_samples,
        run_attention_ablations=args.attention_ablations,
    )
    policy = LeHomeCosmosPolicy(config)
    run_server(policy, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
