import os
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from gymnasium import spaces
torch, nn = try_import_torch()

# 全局地图缓存
GLOBAL_MAP_CACHE = {}
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def preload_maps(maps_dir=f"{current_dir}/envs/defense_sim/obs_map"):
    """预加载所有地图到全局缓存"""
    global GLOBAL_MAP_CACHE
    
    if not GLOBAL_MAP_CACHE:  # 只在第一次调用时加载
        print(f"预加载地图从 {maps_dir}")
        if not os.path.exists(maps_dir):
            raise ValueError(f"地图目录不存在: {maps_dir}")
            
        for file_name in os.listdir(maps_dir):
            if file_name.endswith(".npy"):
                map_path = os.path.join(maps_dir, file_name)
                try:
                    # 从文件名获取map_id，假设格式为map_X.npy
                    map_id = int(file_name.split("_")[1].split(".")[0])
                    map_data = np.load(map_path)
                    # 转换为float32类型以确保兼容性
                    if map_data.dtype != np.float32:
                        map_data = map_data.astype(np.float32)
                    GLOBAL_MAP_CACHE[map_id] = map_data
                except Exception as e:
                    print(f"加载地图出错 {file_name}: {e}")
                    
        print(f"已加载 {len(GLOBAL_MAP_CACHE)} 张地图到内存")
    return GLOBAL_MAP_CACHE

class BKYPPOModel(TorchModelV2, nn.Module):
    """自定义PPO模型，处理地图ID输入"""
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, 
                 map_dir="obs_map", map_features=64, hidden_dim=256):
        # 检查action_space是否为Dict类型
        if isinstance(action_space, spaces.Dict):
            # 如果只需要预测原始动作部分，则使用该部分的大小作为num_outputs
            original_action_space = action_space.spaces.get("deployment_pos")
            if original_action_space and isinstance(original_action_space, spaces.Discrete):
                num_outputs = original_action_space.n
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # 获取配置参数
        custom_config = model_config.get("custom_model_config", {})
        self.maps_dir = f"{current_dir}\envs\defense_sim\obs_map"
        self.map_features = custom_config.get("map_features", map_features)
        self.hidden_dim = custom_config.get("hidden_dim", hidden_dim)
        
        # 确保地图已预加载
        self.map_cache = preload_maps(self.maps_dir)
        
        # 获取样本地图的形状以配置网络
        sample_map_id = next(iter(self.map_cache.keys()))
        sample_map = self.map_cache[sample_map_id]
        self.map_height, self.map_width = sample_map.shape
        
        # 地图编码器网络
        # 假设地图是2D灰度图像
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 计算卷积后的特征尺寸
        with torch.no_grad():
            dummy_map = torch.zeros(1, 1, self.map_height, self.map_width)
            conv_out_size = self.map_encoder(dummy_map).shape[1]
        
        # 添加一个FC层来获得固定大小的地图特征
        self.map_fc = nn.Linear(conv_out_size, self.map_features)
        
        # 处理剩余观察空间（除了map_id）
        # 假设观察空间的第一个元素是map_id，剩下的是常规观察
        self.orig_obs = obs_space.original_space.spaces['obs']
        if isinstance(self.orig_obs, spaces.Box):
            #第一个是map_id，剩余是实际观察
            obs_dim = np.prod(self.orig_obs.shape) - 1
        else:
            raise ValueError(f"不支持的观察类型:{type(self.orig_obs)}")
        
        
        # 观察的处理网络 （除了map_id部分）
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU()
        )
        
        # 合并后的主干网络
        self.combined_dims = self.map_features + 128
        self.backbone = nn.Sequential(
            nn.Linear(self.combined_dims, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        
        # 策略头
        self.policy_head = nn.Linear(self.hidden_dim, num_outputs)
        
        # 价值头
        self.value_head = nn.Linear(self.hidden_dim, 1)
        
        # 价值函数的存储
        self._cur_value = None
    
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """前向计算，处理包含map_id的输入"""
        obs_dict = input_dict["obs"]
        action_mask = obs_dict.get("action_mask")
        obs = obs_dict["obs"]
        
        device = obs.device
        # 提取map_id
        map_id = int(obs[0][0]) if int(obs[0][0]) != 0 else 1

        # 提取其他观察特征
        other_obs = obs[:, 1:]

        map_data = self.map_cache[map_id]

        
        
        # 转换为torch tensor并处理
        maps_tensor = torch.tensor(np.array(map_data), dtype=torch.float32, device=device)
        maps_tensor = maps_tensor.unsqueeze(0)  # 添加通道维度 [B, 1, H, W]
        maps_tensor = maps_tensor.unsqueeze(0)
        # 编码地图
        map_features = self.map_encoder(maps_tensor)
        map_embeddings = self.map_fc(map_features)
        # 扩展至（batch_size, 64)
        batch_size = len(obs)
        map_embeddings = map_embeddings.expand(batch_size, -1)
        # 编码其他观察
        other_embeddings = self.obs_encoder(other_obs)
        
        # 组合所有特征
        combined = torch.cat([map_embeddings, other_embeddings], dim=1)
        
        # 通过主干网络
        features = self.backbone(combined)
        
        # 计算策略输出
        logits = self.policy_head(features)

        # 应用动作掩码
        if action_mask is not None:
            inf_mask = torch.clamp(torch.log(action_mask.float()), min=-1e38)
            logits = logits + inf_mask
        
        # 计算价值函数
        self._cur_value = self.value_head(features).squeeze(-1)
        
        return logits, state
    
    @override(TorchModelV2)
    def value_function(self):
        """返回最后计算的状态值"""
        assert self._cur_value is not None, "必须先调用forward()!"
        return self._cur_value



# 注册PPO模型
ModelCatalog.register_custom_model("BKY_PPO_Model", BKYPPOModel)
# 注册BC模型
ModelCatalog.register_custom_model("BKY_BC_Model", BKYPPOModel)