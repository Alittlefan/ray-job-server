import os
import numpy as np
import glob
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import ray
from ray import tune, train
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.algorithms.pg import PGConfig
from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel

import torchvision.models as models


# 支持的算法映射
ALGO_CLASSES = {
    "DQN": "dqn.DQN",
    "DDQN": "dqn.DQN",
    "PPO": "ppo.PPO",
    "SAC": "sac.SAC",
    "A3C": "a3c.A3C",
    "A2C": "a2c.A2C",
    "TRPO": "ppo.PPO",
    "DDPG": "ddpg.DDPG",
    "PG": "pg.PG",
    "TD3": "ddpg.DDPG",  # 新增算法：Twin Delayed DDPG
}

# 激活函数映射（支持中英文）
ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU(),
    "ReLU": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "Sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "gelu": nn.GELU(),
}


class HopfieldNetwork(nn.Module):
    """Hopfield网络实现"""

    def __init__(self, input_dim, encode_num=2, decode_num=2, char_dimension=512):
        super().__init__()
        self.input_dim = input_dim
        self.encode_num = encode_num
        self.decode_num = decode_num
        self.char_dimension = char_dimension

        # 编码层
        self.encode_layers = nn.ModuleList(
            [
                nn.Linear(input_dim if i == 0 else char_dimension, char_dimension)
                for i in range(encode_num)
            ]
        )

        # 存储模式的权重矩阵
        self.weight = nn.Parameter(torch.randn(char_dimension, char_dimension))

        # 解码层
        self.decode_layers = nn.ModuleList(
            [
                nn.Linear(
                    char_dimension, char_dimension if i < decode_num - 1 else input_dim
                )
                for i in range(decode_num)
            ]
        )

        self.output_dim = input_dim

    def forward(self, x):
        # 编码
        for layer in self.encode_layers:
            x = torch.tanh(layer(x))

        # Hopfield动态更新
        energy = x @ self.weight
        x = torch.tanh(energy)

        # 解码
        for i, layer in enumerate(self.decode_layers):
            x = layer(x)
            if i < len(self.decode_layers) - 1:
                x = torch.tanh(x)

        return x


class AutoEncoder(nn.Module):
    """自动编码器实现"""

    def __init__(self, input_dim, encode_data, decode_data, loss_fun="MSE"):
        super().__init__()
        self.loss_fun = loss_fun

        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for layer_config in encode_data:
            hidden_dim = int(layer_config["neuronSize"])
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))

            # 添加激活函数
            activation = layer_config.get("activationFunction", "ReLU")
            if activation in ACTIVATION_FUNCTIONS:
                encoder_layers.append(ACTIVATION_FUNCTIONS[activation])

            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # 解码器
        decoder_layers = []
        for i, layer_config in enumerate(decode_data):
            if i == 0:
                in_dim = prev_dim
            else:
                in_dim = int(decode_data[i - 1]["neuronSize"])

            out_dim = (
                int(layer_config["neuronSize"])
                if i < len(decode_data) - 1
                else input_dim
            )
            decoder_layers.append(nn.Linear(in_dim, out_dim))

            # 添加激活函数（最后一层除外）
            if i < len(decode_data) - 1:
                activation = layer_config.get("activationFunction", "ReLU")
                if activation in ACTIVATION_FUNCTIONS:
                    decoder_layers.append(ACTIVATION_FUNCTIONS[activation])

        self.decoder = nn.Sequential(*decoder_layers)
        self.output_dim = prev_dim  # 编码器输出维度

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class FNNNetwork(nn.Module):
    """前馈神经网络 (Feedforward Neural Network)"""

    def __init__(self, input_dim, layer_norm=1, fc_layer_num=1, hidden_dim=64):
        super().__init__()
        layers = []

        for i in range(fc_layer_num):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, x):
        return self.network(x)


class DefenseActorCritic(nn.Module):
    """Defense Actor-Critic网络"""

    def __init__(
        self,
        input_dim,
        defense_actor_linear=1,
        defense_actor_layer_norm=2,
        defense_actor_relu=3,
        defense_critic_linear=4,
        defense_critic_layer_norm=5,
        defense_critic_relu=6,
        hidden_dim=64,
    ):
        super().__init__()

        # Actor部分
        actor_layers = []
        for i in range(defense_actor_linear):
            actor_layers.append(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            )

        for i in range(defense_actor_layer_norm):
            actor_layers.append(nn.LayerNorm(hidden_dim))

        for i in range(defense_actor_relu):
            actor_layers.append(nn.ReLU())

        self.actor = nn.Sequential(*actor_layers)

        # Critic部分
        critic_layers = []
        for i in range(defense_critic_linear):
            critic_layers.append(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            )

        for i in range(defense_critic_layer_norm):
            critic_layers.append(nn.LayerNorm(hidden_dim))

        for i in range(defense_critic_relu):
            critic_layers.append(nn.ReLU())

        self.critic = nn.Sequential(*critic_layers)
        self.output_dim = hidden_dim

    def forward(self, x):
        actor_out = self.actor(x)
        critic_out = self.critic(x)
        return actor_out + critic_out  # 简单融合


class GNNNetwork(nn.Module):
    """图神经网络"""

    def __init__(self, input_dim, hidden_dimension_size=64):
        super().__init__()
        try:
            from torch_geometric.nn import GCNConv, global_mean_pool

            self.conv1 = GCNConv(input_dim, hidden_dimension_size)
            self.conv2 = GCNConv(hidden_dimension_size, hidden_dimension_size)
            self.output_dim = hidden_dimension_size
        except ImportError:
            # 如果没有torch_geometric，使用简单的全连接层替代
            self.conv1 = nn.Linear(input_dim, hidden_dimension_size)
            self.conv2 = nn.Linear(hidden_dimension_size, hidden_dimension_size)
            self.output_dim = hidden_dimension_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x


class MLPNetwork(nn.Module):
    """多层感知机"""

    def __init__(self, input_dim, hidden_layers, activation="relu"):
        super().__init__()
        self.activation = ACTIVATION_FUNCTIONS.get(activation, nn.ReLU())

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_layers[-1] if hidden_layers else input_dim

    def forward(self, x):
        return self.network(x)


class RNNWrapper(nn.Module):
    """RNN包装器"""

    def __init__(self, rnn_type, input_dim, hidden_size, circul_num=1):
        super().__init__()
        if rnn_type == "RNN":
            self.rnn = nn.RNN(input_dim, hidden_size, circul_num, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_size, circul_num, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_dim, hidden_size, circul_num, batch_first=True)

        self.output_dim = hidden_size

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        output, _ = self.rnn(x)
        return output[:, -1, :]


class CNNNetwork(nn.Module):
    """卷积神经网络"""

    def __init__(self, input_dim, conv_layers):
        super().__init__()
        layers = []
        in_channels = 1  # 假设输入是1维

        for layer_config in conv_layers:
            out_channels = int(layer_config["neuronCount"])
            kernel_size = int(layer_config["neuronSize"])

            layers.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        # 计算输出维度（简化计算）
        self.output_dim = in_channels * input_dim

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # 添加通道维度
        x = self.conv_layers(x)
        x = self.flatten(x)
        return x


class TransformerNetwork(nn.Module):
    """Transformer网络"""

    def __init__(
        self, input_dim, att_head_num=8, decode_layer=6, encode_layer=6, hidden_size=512
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(input_dim, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=att_head_num, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encode_layer)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=att_head_num, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decode_layer)

        self.output_dim = hidden_size

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        x = self.input_projection(x)

        # Encoder
        encoded = self.encoder(x)

        # Decoder (使用编码输出作为memory)
        decoded = self.decoder(x, encoded)

        return decoded[:, -1, :]  # 返回最后一个时间步


class VectorResNetNetwork(nn.Module):
    """
    将一维特征向量转换为图像后接入 ResNet，解决非图像输入问题
    """

    def __init__(self, input_dim, version="ResNet18", img_size=32, out_dim=64):
        super().__init__()
        # 将向量投影为 CxHxW 的图像
        # 使用 3 通道 (RGB)；H=W=img_size
        self.img_size = img_size
        self.fc_proj = nn.Linear(input_dim, 3 * img_size * img_size)
        # 加载 ResNet
        if version == "ResNet18":
            backbone = models.resnet18(pretrained=False)
        else:
            backbone = models.resnet34(pretrained=False)
        # 替换最后一层全连接
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, out_dim)
        self.backbone = backbone
        self.output_dim = out_dim

    def forward(self, x):
        # x: [B, input_dim]
        B = x.size(0)
        # 投影并 reshape 为图像
        img = self.fc_proj(x).view(B, 3, self.img_size, self.img_size)
        # 通过 ResNet
        return self.backbone(img)


# 网络构建器映射
def build_network(network_type, config, input_dim, obs_shape=None):
    """根据配置构建网络"""

    if network_type in ["MLP"]:
        return MLPNetwork(
            input_dim,
            config.get("hidden_layers", [64, 64]),
            config.get("activation", "relu"),
        )

    elif network_type in ["RNN"]:
        return RNNWrapper(
            "RNN", input_dim, config.get("hidden_size", 64), config.get("circul_num", 1)
        )

    elif network_type in ["LSTM"]:
        return RNNWrapper(
            "LSTM",
            input_dim,
            config.get("hidden_size", 64),
            config.get("circul_num", 1),
        )

    elif network_type in ["GRU"]:
        return RNNWrapper(
            "GRU", input_dim, config.get("hidden_size", 64), config.get("circul_num", 1)
        )

    elif network_type in ["Hopfield网络", "Hopfield"]:
        return HopfieldNetwork(
            input_dim,
            config.get("encode_num", 2),
            config.get("decode_num", 2),
            config.get("char_dimension", 512),
        )

    elif network_type in ["自动编码器", "AutoEncoder"]:
        return AutoEncoder(
            input_dim,
            config.get(
                "encode_data", [{"neuronSize": "64", "activationFunction": "ReLU"}]
            ),
            config.get(
                "decode_data", [{"neuronSize": "64", "activationFunction": "ReLU"}]
            ),
            config.get("loss_fun", "MSE"),
        )

    elif network_type in ["CNN"]:
        return CNNNetwork(
            input_dim,
            config.get("conv_layers", [{"neuronCount": "32", "neuronSize": "3"}]),
        )

    elif network_type in ["ResNet"]:
        # 如果是纯向量输入
        if obs_shape is not None and len(obs_shape) == 1:
            version = config.get("version", "ResNet18")
            return VectorResNetNetwork(input_dim, version)
        else:
            # 否则当做真实图像来用
            version = config.get("version", "ResNet18")
            if version == "ResNet18":
                model = models.resnet18(pretrained=False)
            else:
                model = models.resnet34(pretrained=False)
            # 最后一层映射到一个隐藏向量
            model.fc = nn.Linear(model.fc.in_features, config.get("out_dim", 64))
            model.output_dim = config.get("out_dim", 64)
            return model

    elif network_type in ["Transformer"]:
        return TransformerNetwork(
            input_dim,
            config.get("att_head_num", 8),
            config.get("decode_layer", 6),
            config.get("encode_layer", 6),
            config.get("hidden_size", 512),
        )

    elif network_type in ["GNN"]:
        # 如果是向量观测，就退回到 MLP，以避免 GCNConv 缺少 edge_index
        if obs_shape is not None and len(obs_shape) == 1:
            return FNNNetwork(
                input_dim,
                layer_norm=1,
                fc_layer_num=2,
                hidden_dim=config.get("hidden_dimension_size", 64),
            )
        else:
            return GNNNetwork(input_dim, config.get("hidden_dimension_size", 64))

    elif network_type in ["FNN"]:
        return FNNNetwork(
            input_dim, config.get("layer_norm", 1), config.get("fc_layer_num", 1)
        )

    elif network_type in ["DefenseActor_Critic"]:
        return DefenseActorCritic(
            input_dim,
            config.get("defense_actor_linear", 1),
            config.get("defense_actor_layer_norm", 2),
            config.get("defense_actor_relu", 3),
            config.get("defense_critic_linear", 4),
            config.get("defense_critic_layer_norm", 5),
            config.get("defense_critic_relu", 6),
        )

    else:
        raise ValueError(f"不支持的网络类型: {network_type}")


class CustomModel(TorchModelV2, nn.Module):
    """自定义模型类"""

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        nn.Module.__init__(self)
        super(CustomModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        obs_shape = obs_space.shape
        custom_conf = model_config["custom_model_config"]
        agent_model_conf = custom_conf["agent_model"]

        input_dim = obs_space.shape[0]

        # 构建 encoder
        enc_cfg = agent_model_conf.get("encoder", {})
        self.encoder = build_network(
            enc_cfg["network_type"], enc_cfg, input_dim, obs_shape
        )
        encoder_output_dim = getattr(self.encoder, "output_dim", 64)

        # 构建 core（如果存在）
        self.core = None
        if "core" in agent_model_conf:
            core_cfg = agent_model_conf["core"]
            self.core = build_network(
                core_cfg["network_type"], core_cfg, encoder_output_dim
            )
            core_output_dim = getattr(self.core, "output_dim", encoder_output_dim)
        else:
            core_output_dim = encoder_output_dim

        if num_outputs is None:
            if hasattr(action_space, "n"):
                num_outputs = action_space.n
            elif hasattr(action_space, "shape"):
                num_outputs = int(np.product(action_space.shape))
            else:
                raise ValueError("Cannot infer num_outputs from action_space")
        self.action_head = nn.Linear(core_output_dim, num_outputs)

        # 构建 critic
        crt_cfg = agent_model_conf["critic"]
        self.critic = build_network(crt_cfg["network_type"], crt_cfg, core_output_dim)
        critic_output_dim = getattr(self.critic, "output_dim", 64)
        self.value_head = nn.Linear(critic_output_dim, 1)

        # 保存最后的特征用于 value function
        self._last_features = None

    def forward(self, input_dict, state=None, seq_lens=None):
        obs = input_dict["obs"].float()

        # Encoder
        features = self.encoder(obs)

        # Core (如果存在)
        if self.core is not None:
            features = self.core(features)

        # 保存特征用于 value function
        self._last_features = features

        # Action logits
        action_logits = self.action_head(features)

        return action_logits, state

    def value_function(self):
        if self._last_features is None:
            return torch.zeros(1)

        critic_features = self.critic(self._last_features)
        value = self.value_head(critic_features)
        return torch.reshape(value, [-1])


class CustomSACModel(SACTorchModel, nn.Module):
    """自定义模型类"""

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        nn.Module.__init__(self)
        if model_config is None:
            model_config = {}
        super(CustomSACModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        obs_shape = obs_space.shape
        custom_conf = model_config.get("custom_model_config", {})
        agent_model_conf = custom_conf.get("agent_model", {})

        input_dim = obs_space.shape[0]

        # 构建 encoder
        enc_cfg = agent_model_conf.get("encoder", {})
        self.encoder = build_network(
            enc_cfg["network_type"], enc_cfg, input_dim, obs_shape
        )
        encoder_output_dim = getattr(self.encoder, "output_dim", 64)

        # 构建 core（如果存在）
        self.core = None
        if "core" in agent_model_conf:
            core_cfg = agent_model_conf["core"]
            self.core = build_network(
                core_cfg["network_type"], core_cfg, encoder_output_dim
            )
            core_output_dim = getattr(self.core, "output_dim", encoder_output_dim)
        else:
            core_output_dim = encoder_output_dim

        if num_outputs is None:
            if hasattr(action_space, "n"):
                num_outputs = action_space.n
            elif hasattr(action_space, "shape"):
                num_outputs = int(np.product(action_space.shape))
            else:
                raise ValueError("Cannot infer num_outputs from action_space")
        self.action_head = nn.Linear(core_output_dim, num_outputs)

        # 构建 critic
        crt_cfg = agent_model_conf["critic"]
        self.critic = build_network(crt_cfg["network_type"], crt_cfg, core_output_dim)
        critic_output_dim = getattr(self.critic, "output_dim", 64)
        self.value_head = nn.Linear(critic_output_dim, 1)

        # 保存最后的特征用于 value function
        self._last_features = None

    def forward(self, input_dict, state=None, seq_lens=None):
        obs = input_dict["obs"].float()

        # Encoder
        features = self.encoder(obs)

        # Core (如果存在)
        if self.core is not None:
            features = self.core(features)

        # 保存特征用于 value function
        self._last_features = features

        # Action logits
        action_logits = self.action_head(features)

        return action_logits, state

    def value_function(self):
        if self._last_features is None:
            return torch.zeros(1)

        critic_features = self.critic(self._last_features)
        value = self.value_head(critic_features)
        return torch.reshape(value, [-1])


def get_algorithm_config(algo_name):
    """获取算法配置"""
    config_map = {
        "PPO": PPOConfig(),
        "DQN": DQNConfig(),
        "DDQN": DQNConfig(),
        "SAC": SACConfig(),
        "A3C": A3CConfig(),
        "A2C": A2CConfig(),
        "TRPO": PPOConfig(),  # 使用PPO配置模拟TRPO
        "DDPG": DDPGConfig(),
        "PG": PGConfig(),
        "TD3": DDPGConfig(),  # 新增的TD3算法
    }
    return config_map.get(algo_name)


def build_tuner(task_config, algo_config):
    """构建调优器"""
    pre = task_config["pretrain"]
    algo_name = pre["algorithm"]

    # 设置自定义模型
    if algo_name == "SAC":
        algo_config.training(
            model={
                "custom_model": CustomSACModel,
                "custom_model_config": {"agent_model": pre["agent_model"]},
            },
            _enable_learner_api=False,
        )
        # 3) 拿到最终 dict 之后，手动复制一份给 Q_model
        # config_dict = algo_config.to_dict()
        algo_config["Q_model"] = algo_config["model"]

        # algo_config.training(
        #     policy_model_config={
        #         "custom_model": CustomSACModel,
        #         "custom_model_config": {"agent_model": copy.deepcopy(pre["agent_model"])},
        #     },
        #     q_model_config={
        #         "custom_model": CustomSACModel,
        #         "custom_model_config": {"agent_model":  copy.deepcopy(pre["agent_model"])},
        #     },
        # _enable_learner_api=False,

        # )
    else:
        algo_config.training(
            model={
                "custom_model": CustomModel,
                "custom_model_config": {"agent_model": pre["agent_model"]},
            },
            _enable_learner_api=False,
        )

    algo_config.rl_module(_enable_rl_module_api=False)

    # 根据算法类型进行特殊配置
    if algo_name == "DDQN":
        algo_config.training(double_q=True),
        algo_name = "DQN"
    elif algo_name == "TRPO":
        # TRPO-like配置
        algo_config.training(
            clip_param=0.1,
            train_batch_size=2000,
            num_sgd_iter=10,
            lr=3e-4,
            entropy_coeff=0.01,
            kl_coeff=0.2,
            kl_target=0.01,
        )
        algo_name = "PPO"
    elif algo_name == "TD3":
        # TD3特殊配置
        algo_config.training(
            twin_q=True,
            policy_delay=2,
            smooth_target_policy=True,
            target_noise=0.2,
            target_noise_clip=0.5,
        )

    algo_class = get_trainable_cls(algo_name)
    base_dir = os.getcwd()
    save_subdir = os.path.join("ray_results_custom_model", f"{algo_name}_run")
    save_dir = os.path.join(base_dir, save_subdir)
    os.makedirs(save_dir, exist_ok=True)

    tuner = tune.Tuner(
        algo_class,
        param_space=algo_config.to_dict(),
        run_config=train.RunConfig(
            name=save_dir,
            checkpoint_config=train.CheckpointConfig(**pre.get("checkpoint", {})),
            stop=pre.get("stop", {}),
        ),
    )
    return tuner


def load_all_task_configs():
    """遍历脚本目录下 all_config 文件夹里的所有 .json，返回 list of (task_id, task_config dict)"""
    # 1. 当前文件（脚本）所在的绝对目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. all_config 目录
    config_dir = os.path.join(base_dir, "all_config")
    # 3. 匹配所有 json 文件
    pattern = os.path.join(config_dir, "*.json")

    tasks = []
    for path in glob.glob(pattern):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        task_id = data.get("task_id", os.path.splitext(os.path.basename(path))[0])
        task_cfg = data["task_config"]
        tasks.append((task_id, task_cfg))
    return tasks


# # 测试示例
# if __name__ == "__main__":
#     ray.init(num_cpus=4, num_gpus=0)

#     # 注册自定义模型
#     ModelCatalog.register_custom_model("custom_model", CustomModel)

#     # 测试配置
#     task_config = {
#         "pretrain": {
#             "agent_model": {
#                 "encoder": {
#                     "network_type": "自动编码器",
#                     "loss_fun": "MSE",
#                     "decode_data": [
#                         {
#                             "id": "eeaa2864-807e-4b65-9a5c-07364715f5e3",
#                             "neuronCount": "",
#                             "activationFunction": "Sigmoid",
#                             "neuronSize": "2",
#                         }
#                     ],
#                     "encode_data": [{"neuronSize": "1", "activationFunction": "ReLU"}],
#                 },
#                 "core": {"network_type": "GRU", "hidden_size": 3, "circul_num": 3},
#                 "critic": {
#                     "network_type": "MLP",
#                     "hidden_layers": [128, 128],
#                     "activation": "relu",
#                 },
#             },
#             "algorithm": "DDQN",
#             "stop": {"training_iteration": 3},
#         }
#     }

#     # 构建算法配置
#     algo_cfg = get_algorithm_config("DDQN")
#     algo_cfg = (
#         algo_cfg.environment("CartPole-v1").framework("torch").resources(num_gpus=0)
#     )

#     try:
#         tuner = build_tuner(task_config, algo_cfg)
#         results = tuner.fit()
#         print("训练完成!")
#     except Exception as e:
#         print(f"训练失败: {e}")

#     ray.shutdown()


# if __name__ == "__main__":
#     import os
#     import ray
#     from ray import tune, train
#     from ray.rllib.algorithms.ppo import PPOConfig
#     from ray.rllib.algorithms.dqn import DQNConfig
#     from ray.rllib.algorithms.sac import SACConfig
#     from ray.rllib.algorithms.a3c import A3CConfig
#     from ray.rllib.algorithms.a2c import A2CConfig
#     from ray.rllib.algorithms.ddpg import DDPGConfig
#     from ray.rllib.algorithms.pg import PGConfig

#     ray.init(local_mode=True, num_cpus=16, num_gpus=0)
#     # 注册自定义模型
#     ModelCatalog.register_custom_model("custom_model", CustomModel)
#     algos = ["PPO", "DQN", "DDQN", "SAC", "A3C", "A2C", "TRPO", "DDPG", "PG", "TD3"]
#     nets = [
#         "MLP",
#         "RNN",
#         "LSTM",
#         "GRU",
#         "Hopfield",
#         "AutoEncoder",
#         "CNN",
#         "ResNet",
#         "Transformer",
#         "GNN",
#     ]

#     results = {}

#     for algo in algos:
#         # 拿到对应的算法配置
#         algo_cfg = get_algorithm_config(algo)
#         # 简单设定环境和框架
#         algo_cfg = algo_cfg.environment("CartPole-v1").framework("torch")
#         # 如果是 DDQN，就先打开 double_q
#         if algo == "DDQN":
#             algo_cfg = algo_cfg.training(double_q=True)

#         for net in nets:
#             combo = f"{algo}-{net}"
#             print(combo)
#             try:
#                 # 构造一个伪 task_config 只包含 encoder；critic 用默认 MLP
#                 fake_task = {
#                     "pretrain": {
#                         "algorithm": algo,
#                         "agent_model": {
#                             "encoder": {
#                                 "network_type": net,
#                                 # 对于有 hidden_layers 的网络，给个示例参数
#                                 **({"hidden_layers": [64, 64]} if net == "MLP" else {}),
#                                 **(
#                                     {"hidden_size": 32, "circul_num": 1}
#                                     if net in ["RNN", "LSTM", "GRU"]
#                                     else {}
#                                 ),
#                                 **(
#                                     {
#                                         "encode_data": [
#                                             {
#                                                 "neuronSize": "32",
#                                                 "activationFunction": "ReLU",
#                                             }
#                                         ],
#                                         "decode_data": [
#                                             {
#                                                 "neuronSize": "32",
#                                                 "activationFunction": "ReLU",
#                                             }
#                                         ],
#                                     }
#                                     if net == "AutoEncoder"
#                                     else {}
#                                 ),
#                             },
#                             # 简单给 critic 也用 MLP
#                             "critic": {
#                                 "network_type": "MLP",
#                                 "hidden_layers": [32, 32],
#                                 "activation": "relu",
#                             },
#                         },
#                         # 为了让 Tuner 一启动就停掉
#                         "stop": {"training_iteration": 0},
#                     }
#                 }

#                 # 调用 build_tuner（不实际跑训练，只做初始化和参数检查）
#                 tuner = build_tuner(fake_task, algo_cfg)
#                 # 尝试一次 fit()，它会立刻因为 training_iteration=0 而退出
#                 tuner.fit()
#                 results[combo] = "OK"
#             except Exception as e:
#                 results[combo] = f"FAILED: {type(e).__name__}: {e}"

#     # 打印所有组合的测试结果
#     for combo, status in results.items():
#         print(f"{combo}: {status}")

#     ray.shutdown()

#     """
#     PPO-MLP: OK
#     PPO-RNN: OK
#     PPO-LSTM: OK
#     PPO-GRU: OK
#     PPO-Hopfield: OK
#     PPO-AutoEncoder: OK
#     PPO-CNN: OK
#     PPO-ResNet: OK
#     PPO-Transformer: OK
#     PPO-GNN: OK
#     DQN-MLP: OK
#     DQN-RNN: OK
#     DQN-LSTM: OK
#     DQN-GRU: OK
#     DQN-Hopfield: OK
#     DQN-AutoEncoder: OK
#     DQN-CNN: OK
#     DQN-ResNet: OK
#     DQN-Transformer: OK
#     DQN-GNN: OK
#     DDQN-MLP: OK
#     DDQN-RNN: OK
#     DDQN-LSTM: OK
#     DDQN-GRU: OK
#     DDQN-Hopfield: OK
#     DDQN-AutoEncoder: OK
#     DDQN-CNN: OK
#     DDQN-ResNet: OK
#     DDQN-Transformer: OK
#     DDQN-GNN: OK
#     SAC-MLP: OK
#     SAC-RNN: OK
#     SAC-LSTM: OK
#     SAC-GRU: OK
#     SAC-Hopfield: OK
#     SAC-AutoEncoder: OK
#     SAC-CNN: OK
#     SAC-ResNet: OK
#     SAC-Transformer: OK
#     SAC-GNN: OK
#     A3C-MLP: OK
#     A3C-RNN: OK
#     A3C-LSTM: OK
#     A3C-GRU: OK
#     A3C-Hopfield: OK
#     A3C-AutoEncoder: OK
#     A3C-CNN: OK
#     A3C-ResNet: OK
#     A3C-Transformer: OK
#     A3C-GNN: OK
#     A2C-MLP: OK
#     A2C-RNN: OK
#     A2C-LSTM: OK
#     A2C-GRU: OK
#     A2C-Hopfield: OK
#     A2C-AutoEncoder: OK
#     A2C-CNN: OK
#     A2C-ResNet: OK
#     A2C-Transformer: OK
#     A2C-GNN: OK
#     TRPO-MLP: FAILED: TypeError: training() got an unexpected keyword argument 'rollout_fragment_length'
#     TRPO-RNN: FAILED: TypeError: training() got an unexpected keyword argument 'rollout_fragment_length'
#     TRPO-LSTM: FAILED: TypeError: training() got an unexpected keyword argument 'rollout_fragment_length'
#     TRPO-GRU: FAILED: TypeError: training() got an unexpected keyword argument 'rollout_fragment_length'
#     TRPO-Hopfield: FAILED: TypeError: training() got an unexpected keyword argument 'rollout_fragment_length'
#     TRPO-AutoEncoder: FAILED: TypeError: training() got an unexpected keyword argument 'rollout_fragment_length'
#     TRPO-CNN: FAILED: TypeError: training() got an unexpected keyword argument 'rollout_fragment_length'
#     TRPO-ResNet: FAILED: TypeError: training() got an unexpected keyword argument 'rollout_fragment_length'
#     TRPO-Transformer: FAILED: TypeError: training() got an unexpected keyword argument 'rollout_fragment_length'
#     TRPO-GNN: FAILED: TypeError: training() got an unexpected keyword argument 'rollout_fragment_length'
#     DDPG-MLP: OK
#     DDPG-RNN: OK
#     DDPG-LSTM: OK
#     DDPG-GRU: OK
#     DDPG-Hopfield: OK
#     DDPG-AutoEncoder: OK
#     DDPG-CNN: OK
#     DDPG-ResNet: OK
#     DDPG-Transformer: OK
#     DDPG-GNN: OK
#     PG-MLP: OK
#     PG-RNN: OK
#     PG-LSTM: OK
#     PG-GRU: OK
#     PG-Hopfield: OK
#     PG-AutoEncoder: OK
#     PG-CNN: OK
#     PG-ResNet: OK
#     PG-Transformer: OK
#     PG-GNN: OK
#     TD3-MLP: OK
#     TD3-RNN: OK
#     TD3-LSTM: OK
#     TD3-GRU: OK
#     TD3-Hopfield: OK
#     TD3-AutoEncoder: OK
#     TD3-CNN: OK
#     TD3-ResNet: OK
#     TD3-Transformer: OK
#     TD3-GNN: OK
#     """

if __name__ == "__main__":
    ray.init(local_mode=True, num_cpus=16, num_gpus=0)

    # 注册 custom model
    from ray.rllib.models import ModelCatalog

    # ModelCatalog.register_custom_model("custom_model", CustomModel)
    # ModelCatalog.register_custom_model("custom_sac_model", CustomSACModel)

    # 加载所有任务配置
    all_tasks = load_all_task_configs()

    final_results = {}
    pass_id = np.arange(3)
    number = -1
    for task_id, task in all_tasks:
        number += 1
        if number in pass_id:
            continue

        combo = f"task_id:{task_id}"
        print(combo)
        # try:
        pre = task["pretrain"]
        pre["stop"] = {"training_iteration": 0}
        algo_name = pre["algorithm"]

        # 拿到对应算法的基础 config
        base_cfg = get_algorithm_config(algo_name)
        algo_cfg = base_cfg.environment("CartPole-v1").framework("torch")

        # 构建 tuner
        tuner = build_tuner(task, algo_cfg)

        print(f"开始运行任务 {task_id} ({algo_name}) …")
        results = tuner.fit()
        final_results[combo] = "OK"
        # except Exception as e:
        #     final_results[combo] = f"FAILED: {type(e).__name__}: {e}"

    # 打印所有组合的测试结果
    for combo, status in final_results.items():
        print(f"{combo}: {status}")
    ray.shutdown()
