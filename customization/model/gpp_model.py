import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from uav_3d.examples.rllib.action_mask_model import ActionMaskModel


class CNNAttention(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=2
    ):
        super(CNNAttention, self).__init__()
        self.w_qs = nn.Conv3d(
            in_channels, out_channels // 8, kernel_size, stride, padding
        )
        self.w_ks = nn.Conv3d(
            in_channels, out_channels // 8, kernel_size, stride, padding
        )
        self.w_vs = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.pool = nn.AvgPool3d(pool_size)
        nn.init.orthogonal_(self.w_qs.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.w_ks.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.w_vs.weight, gain=np.sqrt(2))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        batch_size, _, D, H, W = inputs.shape
        q = self.pool(self.pool(self.w_qs(inputs)))
        k = self.pool(self.pool(self.w_ks(inputs)))
        v = self.pool(self.pool(self.w_vs(inputs)))
        _, _, D_p, H_p, W_p = q.shape
        output_size = D_p * H_p * W_p
        q = q.view(batch_size, -1, output_size).permute(0, 2, 1)
        k = k.view(batch_size, -1, output_size)
        v = v.view(batch_size, -1, output_size)
        attn = torch.bmm(q, k)
        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(batch_size, -1, D_p, H_p, W_p)
        out = nn.functional.interpolate(
            out, size=(D, H, W), mode="trilinear", align_corners=False
        )
        out = self.gamma * out + self.w_vs(inputs)
        return out


class Custom3DCNN(nn.Module):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(Custom3DCNN, self).__init__()
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            CNNAttention(n_input_channels, 32),
            nn.ReLU(),
            CNNAttention(32, 16),
            nn.ReLU(),
            CNNAttention(16, 16),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample_input).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class UavModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Note: action_mask_model need manually extract the original_space
        self.original_space = getattr(obs_space, "original_space", obs_space)
        if (
            hasattr(self.original_space, "spaces")
            and "obs" in self.original_space.spaces
        ):
            self.original_space = self.original_space["obs"]

        color_grid_space = self.original_space["color_grid"]
        features_space = self.original_space["features"]

        self.grid_extractor = Custom3DCNN(color_grid_space, features_dim=256)
        self.semantic_extractor = nn.Sequential(
            nn.Linear(features_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        combined_features_dim = self.grid_extractor.features_dim + 64

        self.combined_layer = nn.Sequential(
            nn.Linear(combined_features_dim, 1024), nn.ReLU()
        )

        self.q_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs),
        )

    def forward(self, input_dict, state, seq_lens):
        observations = restore_original_dimensions(
            input_dict["obs"], self.original_space, "torch"
        )
        grid_features = self.grid_extractor(observations["color_grid"].float())
        semantic_features = self.semantic_extractor(observations["features"].float())
        combined = torch.cat([grid_features, semantic_features], dim=1)
        combined = self.combined_layer(combined)
        logits = self.q_head(combined)
        return logits, state


ModelCatalog.register_custom_model("custom_model", UavModel)
ModelCatalog.register_custom_model("gpp_model", UavModel)
ModelCatalog.register_custom_model("action_mask_model", ActionMaskModel)
