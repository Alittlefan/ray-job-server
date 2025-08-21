import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class DefensePPOModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        obs_space = getattr(obs_space, "original_space", obs_space)
        
        self.map_channels = obs_space["obs"]["map"].shape[0]
        self.enemy_vector_dim = obs_space["obs"]["semantic"].shape[0]

        # 减小卷积层通道数
        self.encoder = nn.Sequential(
            nn.Conv2d(self.map_channels, 8, kernel_size=3, stride=1, padding=1),  # 16->8
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 32->16
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(16 * 90 * 68, 256),  # 512->256
            nn.LayerNorm(256),
            nn.LeakyReLU(),
        )

        # 减小全连接层维度
        self.enemy_fc = nn.Sequential(
            nn.Linear(self.enemy_vector_dim, 128),  # 256->128
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        hidden_dim = 256 + 128  # 总特征维度

        # 共享中间层，进一步减小参数量
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim, 192),
            nn.LayerNorm(192),
            nn.LeakyReLU(),
        )

        # Policy和value heads变得更小
        self.policy_head = nn.Sequential(
            nn.Linear(192, num_outputs),
        )

        self.value_head = nn.Sequential(
            nn.Linear(192, 1),
        )

    def encode(self, map_input, enemy_vector):
        map_features = self.encoder(map_input)
        enemy_features = self.enemy_fc(enemy_vector)
        combined = torch.cat((map_features, enemy_features), dim=-1)
        return self.shared_layer(combined)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        map_input = obs["obs"]["map"]
        action_mask = obs["action_mask"]
        enemy_vector = obs["obs"]["semantic"]

        features = self.encode(map_input, enemy_vector)
        
        # 计算policy logits和value
        logits = self.policy_head(features)
        value = self.value_head(features)

        # 如果是初始化检查阶段（全是0），直接返回原始logits
        if all(mask.sum().item() == 0 for mask in action_mask):
            self._value_out = value.squeeze(-1)
            return logits, state

        # 分别处理三个部分
        logits[:, :40] = logits[:, :40].masked_fill(action_mask[0] == 0, -1e7)
        logits[:, 40:46] = logits[:, 40:46].masked_fill(action_mask[1] == 0, -1e7)
        logits[:, 46:52] = logits[:, 46:52].masked_fill(action_mask[2] == 0, -1e7)

        self._value_out = value.squeeze(-1)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out

    @override(TorchModelV2)
    def get_initial_state(self):
        return []


# 注册自定义模型
ModelCatalog.register_custom_model("defense_c_model", DefensePPOModel)
ModelCatalog.register_custom_model("custom_ppo_model", DefensePPOModel)
