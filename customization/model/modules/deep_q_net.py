from typing import Sequence

from torch import nn

from model.modules.conv_encoder import ConvEncoder
from model.modules.dueling_head import DuelingHead
from model.modules.inverse_dynamics import InverseDynamic


class DeepQNet(nn.Module):
    def __init__(
        self,
        raster_shape: Sequence[int] = (16, 16, 16),
        cnn_channels: Sequence[int] = (32, 64, 64),
        kernel_sizes: Sequence[int] = (3, 3, 3),
        strides: Sequence[int] = (1, 1, 1),
        obs_dim=14,
        hidden_dim=256,
        action_num=15,
    ):
        super(DeepQNet, self).__init__()
        self.encoder = ConvEncoder(
            raster_shape=raster_shape,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            vec_dim=obs_dim,
            vec_out=hidden_dim,
        )
        self.q_head = DuelingHead(hidden_dim, action_num)
        self.inverse_dynamics_model = InverseDynamic(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            action_num=action_num,
        )

    def forward(self, observation, raster):
        embed = self.encoder(observation, raster)
        q_values = self.q_head(embed)
        return q_values

    def inverse_dynamics(self, s_t, r_t, s_tp1, r_tp1):
        embed_t = self.encoder(s_t, r_t)
        embed_tp1 = self.encoder(s_tp1, r_tp1)
        predicted_action = self.inverse_dynamics_model(embed_t, embed_tp1)
        return predicted_action
