import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class DefenseEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class DefenseActor(nn.Module):
    def __init__(self, input_dim, num_outputs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, num_outputs),
        )

    def forward(self, x):
        return self.network(x)


class DefenseCritic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)


class DefensePPOModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        obs_space = getattr(obs_space, "original_space", obs_space)
        self.obs_dim = obs_space["obs"].shape[0]  # 观测空间维度

        # 初始化网络组件
        self.encoder = DefenseEncoder(self.obs_dim)
        self.actor = DefenseActor(128, num_outputs)  # encoder输出维度为128
        self.critic = DefenseCritic(128)

        # 存储最近的value输出
        self._value_out = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs_dict = input_dict["obs"]
        obs = obs_dict["obs"]
        action_mask = obs_dict["action_mask"]

        # 编码观测
        encoded_obs = self.encoder(obs)

        # 计算策略logits
        logits = self.actor(encoded_obs)

        # 应用动作掩码
        logits = logits.masked_fill(action_mask == 0, -1e7)

        # 计算value
        self._value_out = self.critic(encoded_obs).squeeze(1)

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out

    @override(TorchModelV2)
    def get_initial_state(self):
        return []

ModelCatalog.register_custom_model("defense_sim_model", DefensePPOModel)
ModelCatalog.register_custom_model("defense_ppo_model", DefensePPOModel)
