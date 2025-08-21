import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class UAVCustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.original_space = getattr(obs_space, "original_space", obs_space)
        if (
            hasattr(self.original_space, "spaces")
            and "obs" in self.original_space.spaces
        ):
            self.original_space = self.original_space["obs"]

        # Extract the shapes from the observation space
        obstacle_shape = self.original_space[
            "obstacle"
        ].shape  # (width, height, num_features)
        goal_shape = self.original_space["goal"].shape  # (num_features,)
        # Fully connected layers for obstacle
        self.fc_obstacle = nn.Sequential(
            nn.Linear(obstacle_shape[0], 128),
            nn.ReLU(),
        )

        # Fully connected layers for goal
        self.fc_goal = nn.Sequential(
            nn.Linear(goal_shape[0], 64),
            nn.ReLU(),
        )

        combined_input_size = 128 + 64  # Now 64 from obstacle and 64 from goal

        self.final_out = nn.Sequential(
            nn.Linear(combined_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
        )

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = restore_original_dimensions(
            input_dict["obs"], self.original_space, "torch"
        )

        # Extract observations
        obstacle_features = obs["obstacle"]  # Shape: [B, num_features]
        goal_features = obs["goal"]  # Shape: [B, num_features]

        # Process additional features
        obstacle_features_out = self.fc_obstacle(obstacle_features.float())

        goal_features_out = self.fc_goal(goal_features.float())

        # Combine outputs
        combined_out = torch.cat([obstacle_features_out, goal_features_out], dim=1)
        embedding = self.final_out(combined_out)
        return embedding, state


ModelCatalog.register_custom_model("lpp3d_model", UAVCustomModel)
ModelCatalog.register_custom_model("uav_custom_model", UAVCustomModel)
