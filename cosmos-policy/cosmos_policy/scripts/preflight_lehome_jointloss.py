"""Bounded real-data training smoke test with loss/gradient assertions.

Use the normal train arguments, plus trainer.max_iter=3 and a separate job name.
The wrapper is only active in this preflight process, never in production.
"""

import inspect
import json
import os
import runpy
import sys

import torch
import torch.distributed as dist

from cosmos_policy.models.policy_text2world_model import CosmosPolicyDiffusionModel


def main():
    max_iter = int(os.environ.get("JOINTLOSS_PREFLIGHT_MAX_ITER", "3"))
    chunk_size = int(os.environ.get("JOINTLOSS_PREFLIGHT_CHUNK_SIZE", "60"))
    state_t = int(os.environ.get("JOINTLOSS_PREFLIGHT_STATE_T", "10"))
    image_start = int(os.environ.get("JOINTLOSS_PREFLIGHT_IMAGE_START", "-1"))
    image_end = int(os.environ.get("JOINTLOSS_PREFLIGHT_IMAGE_END", "-1"))
    image_multiplier = float(os.environ.get("JOINTLOSS_PREFLIGHT_IMAGE_MULTIPLIER", "1"))
    loss_scale = float(os.environ.get("JOINTLOSS_PREFLIGHT_LOSS_SCALE", "10"))
    if f"trainer.max_iter={max_iter}" not in sys.argv:
        raise ValueError(f"Preflight must be bounded by trainer.max_iter={max_iter}")
    original = CosmosPolicyDiffusionModel.compute_loss_with_epsilon_and_sigma
    signature = inspect.signature(original)
    audited = False

    def checked_loss(self, *args, **kwargs):
        nonlocal audited
        result = original(self, *args, **kwargs)
        if audited:
            return result
        audited = True
        bound = signature.bind(self, *args, **kwargs).arguments
        output, loss, mse, edm = result
        assert loss.shape[2] == state_t
        assert self.config.action_loss_multiplier == 1
        assert self.config.future_image_loss_multiplier == image_multiplier
        assert self.config.loss_scale == loss_scale
        assert not self.config.mask_loss_for_action_future_state_prediction
        expected_loss = edm.clone()
        if image_start >= 0:
            expected_loss[:, :, image_start : image_end + 1] *= image_multiplier
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        world = bound["world_model_sample_mask"].bool()
        expected = torch.zeros((loss.shape[0], state_t), dtype=torch.bool, device=loss.device)
        expected[:, 5:] = True
        expected[world, 5] = False
        actual_condition = output["condition"].condition_video_input_mask_B_C_T_H_W
        actual_condition = actual_condition[:, 0, :, 0, 0].bool()
        assert torch.equal(~actual_condition, expected)
        per_slot_mse = mse.detach().mean(dim=(1, 3, 4))
        assert torch.isfinite(loss).all()
        assert torch.all(per_slot_mse[expected] > 0)
        assert torch.all(per_slot_mse[~expected] == 0)
        if image_start >= 0:
            assert torch.all(bound["future_video_start_indices"] == image_start)
            assert torch.all(bound["future_video_end_indices"] == image_end)
            demo_image_mse = mse[~world, :, image_start : image_end + 1].mean()
            torch.testing.assert_close(
                output["demo_sample_future_image_mse_loss"],
                demo_image_mse,
            )
        assert torch.all(bound["value_indices"] == -1)
        assert tuple(bound["action_chunk"].shape[1:]) == (chunk_size, 12)
        dropped = (bound["proprio"] == 0).all(dim=1).sum()
        counts = torch.stack([(~world).sum(), world.sum(), dropped])
        dist.all_reduce(counts)
        if dist.get_rank() == 0:
            print("JOINT_LOSS_PREFLIGHT " + json.dumps({
                "status": "forward_passed", "global_policy_samples": int(counts[0]),
                "global_state_samples": int(counts[1]), "global_proprio_dropped": int(counts[2]),
                "policy_supervised_slots": list(range(5, state_t)),
                "state_supervised_slots": list(range(6, state_t)),
                "image_supervised_slots": list(range(image_start, image_end + 1)) if image_start >= 0 else None,
                "action_multiplier": 1,
                "image_multiplier": image_multiplier,
                "loss_scale": loss_scale,
            }), flush=True)

        def check_gradient(gradient):
            per_slot = gradient.detach().abs().sum(dim=(1, 3, 4))
            assert torch.isfinite(gradient).all()
            assert torch.all(per_slot[expected] > 0)
            assert torch.all(per_slot[~expected] == 0)
            print(f"JOINT_LOSS_PREFLIGHT backward_passed rank={dist.get_rank()}", flush=True)

        output["model_pred"].x0.register_hook(check_gradient)
        return result

    CosmosPolicyDiffusionModel.compute_loss_with_epsilon_and_sigma = checked_loss
    runpy.run_module("cosmos_policy.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
