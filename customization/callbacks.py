from __future__ import annotations

import logging

from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.policy import Policy
import numpy as np

class CheckpointCallback(DefaultCallbacks):
    """Base class for checkpoint saving callbacks.

    Note: This callback must be used so that `Result.from_path()` can be used to load the checkpoint.
    """

    def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs) -> None:
        if algorithm._storage:
            algorithm._storage.current_checkpoint_index += 1
            result["checkpoint_dir_name"] = algorithm._storage.checkpoint_dir_name
            algorithm._storage.current_checkpoint_index -= 1


def LoadCheckpointCallback(checkpoint_dir: str, strict=False, base_cls=CheckpointCallback):
    """Create a callback class that loads a checkpoint before training."""

    class _LoadCheckpointCallback(base_cls):
        def __init__(self):
            self.logger = logging.getLogger(__name__)

        def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
            policies = TorchPolicyV2.from_checkpoint(checkpoint_dir)
            weights = (
                {pid: policy.get_weights() for pid, policy in policies.items()}
                if isinstance(policies, dict)
                else {"default_policy": policies.get_weights()}
            )

            if strict:
                algorithm.set_weights(weights)
            else:
                worker = algorithm.workers.local_worker()
                for pid, weight in weights.items():
                    policy: TorchPolicyV2 = worker.policy_map[pid]
                    weight = convert_to_torch_tensor(weight, device=policy.device)
                    policy.model.load_state_dict(weight, strict=False)

            self.logger.info(f"Loaded checkpoint from {checkpoint_dir}")

    return _LoadCheckpointCallback

def LoadModelCallback(checkpoint_path: str, strict:bool = True):
    class CustomCallback(DefaultCallbacks):
        def on_algorithm_init(self, *, algorithm, **kwargs):
            try:
                if checkpoint_path is None:
                    return
                policies = Policy.from_checkpoint(checkpoint_path)
                weights =(
                    {pid: policy.get_weights() for pid, policy in policies.items()}
                    if isinstance(policies, dict)
                    else {"default_policy": policies.get_weights()}
                )

                if strict:
                    algorithm.set_weights(weights)
                else:
                    worker = algorithm.workers.local_worker()
                    for pid, weight in weights.items():
                        policy = worker.policy_map[pid]
                        weight = convert_to_torch_tensor(weight, device=policy.device)
                        policy.model.load_state_dict(weight, strict=False)
            except Exception as e:
                print(f"Error loading model:{e}")

        def on_postprocess_trajectory(self, *, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
            batch = postprocessed_batch
            rewards = batch["rewards"]
            if len(rewards) == 0:
                return
            final_reward = rewards[-1]
            steps = len(rewards)
            #衰减系数分配reward
            decay = 0.8
            total_weight = 0
            weights = []

            for i in range(steps):
                weight = decay ** i 
                weights.insert(0,weight)
                
            
            #归一化权重并分配reward
            weights = [w  for w in weights]
            batch["rewards"] = np.array([final_reward * w for w in weights], dtype=rewards.dtype)
            
           
    return CustomCallback
