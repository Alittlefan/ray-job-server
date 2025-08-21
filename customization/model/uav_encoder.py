from typing import Sequence, Dict, Union, List

import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn
import torch.nn.functional as F

from model.modules.conv_encoder import ConvEncoder
from model.modules.dueling_head import DuelingHead
from model.modules.inverse_dynamics import InverseDynamic

import gymnasium as gym


class UavEncoder(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs
    ):
        nn.Module.__init__(self)
        super(UavEncoder, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        action_num = action_space.n
        custom_model_config = model_config["custom_model_config"]
        hidden_dim: int = custom_model_config.get("hidden_dim", 256)
        raster_shape: Sequence[int] = custom_model_config.get(
            "raster_shape", (16, 16, 16)
        )
        cnn_channels: Sequence[int] = custom_model_config.get(
            "cnn_channels", (32, 64, 128)
        )
        kernel_sizes: Sequence[int] = custom_model_config.get("kernel_sizes", (3, 3, 3))
        strides: Sequence[int] = custom_model_config.get("strides", (1, 1, 1))
        vec_dim: int = custom_model_config.get("vec_dim", 50)
        self.action_num = action_num
        self.raster_shape = raster_shape
        self.vec_dim = vec_dim
        # hidden_dim = num_outputs

        self.encoder = ConvEncoder(
            raster_shape=raster_shape,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            vec_dim=vec_dim,
            vec_out=hidden_dim,
        )
        # self.q_head = DuelingHead(hidden_dim, action_num)
        self.inverse_dynamics_model = InverseDynamic(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            action_num=action_num,
        )

    def forward(self, input_dict: dict[str, dict[str, torch.Tensor]], state, seq_lens):
        observation, raster = (
            input_dict["obs"]["observation"],
            input_dict["obs"]["raster"],
        )
        embed = self.encoder(observation, raster)
        return embed, state

    def custom_loss(
        self, policy_loss: TensorType, loss_inputs: Dict[str, TensorType]
    ) -> Union[List[TensorType], TensorType]:
        batch_size = loss_inputs["obs"].shape[0]
        obs_t = loss_inputs["obs"][:, : self.vec_dim]
        rast_t = loss_inputs["obs"][:, self.vec_dim :].reshape(
            batch_size, *self.raster_shape
        )
        obs_tp1 = loss_inputs["new_obs"][:, : self.vec_dim]
        rast_tp1 = loss_inputs["new_obs"][:, self.vec_dim :].reshape(
            batch_size, *self.raster_shape
        )
        action_predict = self.inverse_dynamics(obs_t, rast_t, obs_tp1, rast_tp1)
        action_t = loss_inputs["actions"]
        if isinstance(loss_inputs["actions"], list):
            action_t = torch.tensor(action_t, dtype=torch.int64).to(obs_t.device())
        action_t = action_t.to(torch.int64)
        actions_one_hot = F.one_hot(action_t, num_classes=self.action_num).to(
            torch.float32
        )
        self_supervised_loss = F.cross_entropy(action_predict, actions_one_hot)
        policy_loss[0] += 0.25 * self_supervised_loss
        return policy_loss

    def inverse_dynamics(self, s_t, r_t, s_tp1, r_tp1):
        embed_t = self.encoder(s_t, r_t)
        embed_tp1 = self.encoder(s_tp1, r_tp1)
        predicted_action = self.inverse_dynamics_model(embed_t, embed_tp1)
        return predicted_action
