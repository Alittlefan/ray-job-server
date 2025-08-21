import copy
import json as js
import os
import random
import time
from hashlib import sha256

import gymnasium as gym
import numpy as np
import pyproj
from gymnasium import spaces
from tcp_comm import *


def load_epoch_json_files(root_dir):
    """数据加载函数"""
    all_data = []
    scene_folders = sorted(os.listdir(root_dir))
    for scene_idx, scene_folder in enumerate(scene_folders):
        scene_path = os.path.join(root_dir, scene_folder)

        json_files = [f for f in os.listdir(scene_path) if f.endswith('.json')]
        for json_file in json_files:
            file_path = os.path.join(scene_path, json_file)
            with open(file_path, 'r') as f:
                json_data = js.load(f)
            
            all_data.append({
                "scene": scene_idx,
                "data": json_data
            })
    scene_pos = {}
    for data in all_data:
        scene_idx = data['scene']
        data = data['data']
        if scene_idx not in scene_pos:
            scene_pos[scene_idx] = {
                "bazhua_pos":set(),
                "bigua_pos": set()
            }
        bazhua_pos = data.get("bazhua_pos", [])
        bigua_pos = data.get("bigua_pos", [])
        for pos in bazhua_pos:
            scene_pos[scene_idx]["bazhua_pos"].add(tuple(pos))
        for pos in bigua_pos:
            scene_pos[scene_idx]["bigua_pos"].add(tuple(pos))

    return all_data, scene_pos


class DefenseGym_train(gym.Env):
    def __init__(self, config=None):
        super(DefenseGym_train, self).__init__()
        self.config = config
        self.red_bigua_pos = config["red_bigua_pos"]
        self.red_bazhua_pos = config["red_bazhua_pos"]
        self.blue_deploy_pos = config["blue_deploy_pos"]
        self.blue_target_pos = config["blue_target_pos"]
        self.num_blue_group = config["num_blue_group"]
        self.max_time = config["max_time"]

        #设置action的水平范围和俯仰范围
        self.action_low_Prange = (-90, -70)
        self.action_high_Prange = (20, 60)
        self.action_low_Hrange = (-150, -120)
        self.action_high_Hrange = (120, 150)

        #设置动作和观察空间
        self.action_space = spaces.Dict({
            "deployment_pos" : spaces.Discrete(len(config["red_bigua_pos"]) + len(config["red_bazhua_pos"])),
            "angles": spaces.Box(
                low=np.array([self.action_low_Prange[0], self.action_high_Prange[0],
                               self.action_low_Hrange[0], self.action_high_Hrange[0]]),
                high=np.array([self.action_low_Prange[1], self.action_high_Prange[1],
                               self.action_low_Hrange[1], self.action_high_Hrange[1]]),
                dtype=np.float32
            )
        })

        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        1+     #地图id
                        4 * config["num_blue_group"]     #(4  经+纬+人数+车数)
                        +  4   #(3经纬高+1 是否使用)
                        * (
                            # len(config["red_bazhua_pos"]) + len(config["red_bigua_pos"])
                            config["num_total_red_deploy"]
                        ),
                    ),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(
                        len(config["red_bigua_pos"]) + len(config["red_bazhua_pos"]),
                    ),
                    dtype=np.int8,
                ),
            }
        )
        self._action_mask = np.ones(len(self.red_bigua_pos) + len(self.red_bazhua_pos),dtype=np.int8)

        
        self._episode_actions = []
        self._step_index = 0

        self.messages = []

        self.units_state = {}
        self.conv_pos = {}
        self.actual_blue_deploy_pos = []
        self.blue_unit_target = []
        

        wgs84 = pyproj.CRS("EPSG:4326")
        geocentric = pyproj.CRS("EPSG:32651")
    
        self.transformer_wgs84 = pyproj.Transformer.from_crs(geocentric, wgs84, always_xy=True)
        self.transformer_geocentric = pyproj.Transformer.from_crs(wgs84, geocentric, always_xy=True)

        #维护一个变量存json读取后的数据 data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = f"{current_dir}\\dataset"
        self.all_data, scene_pos= load_epoch_json_files(self.root_dir)
        self.blue_deployment_signatures = {}
        # self._preprocess_dataset()
        #地图选择
        self.map_id=config["map_id"]


        #场景方案选取下对红方单位的部署点限制：
        self.restrict_num = self._map_red_positions(scene_pos)


    def reset(self, seed=None, options=None):

        # selcet sceanrio
        self.actual_blue_deploy_pos = []
        self.scenario_id =random.randint(0,4)
        self.total_soldier = random.randint(self.config["total_soldier"][0], self.config["total_soldier"][1])
        self.total_vehicle = random.randint(self.config["total_vehicle"][0], self.config["total_vehicle"][1])
        self.actual_blue_deploy_pos = self.generate_blue_deployment(self.total_soldier, self.total_vehicle, self.scenario_id)
        # blue_deploy_pos=self.blue_deploy_pos[self.scenario_id]
        (self.num_red_bazhua, self.num_red_bigua) = [(9,4),(6,4),(7,4),(8,4),(8,4)][self.scenario_id]

        self.messages = []
        self.units_state = {}
        self._episode_actions = []
        self._step_index = 0
        self.blue_unit_target = []
        self.single_reward = 0
        
        self._episode_step_num = self.num_red_bazhua + self.num_red_bigua
        self.action_mapping(self.red_bazhua_pos, self.red_bigua_pos)
        self.state = self.get_observation(self.actual_blue_deploy_pos)
        self._action_mask = np.ones(len(self.red_bigua_pos) + len(self.red_bazhua_pos),dtype=np.int8)

        #方案限制处理mask
        self._action_mask[:] = 0
        self._action_mask[self.restrict_num[self.scenario_id]] = 1
        self._action_mask[len(self.red_bigua_pos) :] = 0   #先壁挂后八爪
       
        #本轮info信息初始化
        self.info = {
            "enemy":{
                "soldier":self.total_soldier,
                "vehicle":self.total_vehicle,
            },
            "red":{
                "BZ": 0,
                "BG": 0,
            },
            }


        return {"obs": self.state, "action_mask": self._action_mask} , {}

    def _normalize_blue_position(self, pos):
        """标准化蓝方部署点格式（6位小数精度）"""
        return (
            round(pos[0], 6),  # 经度
            round(pos[1], 6),  # 纬度
            round(pos[2], 6),  # 高度
            int(pos[3]),       # 朝向
            int(pos[4]),       # 士兵数
            int(pos[5])        # 车辆数
        )

    def _generate_signature(self, positions):
        """生成部署配置唯一签名"""
        sorted_pos = sorted(positions)
        return sha256(str(sorted_pos).encode()).hexdigest()

    def get_current_blue_signature(self):
        """获取当前蓝方部署特征签名"""
        # 提取有效部署点（实际部署的兵力）
        valid_blue = [
            self._normalize_blue_position(pos)
            for pos in self.actual_blue_deploy_pos
            if pos[4] > 0 or pos[5] > 0
        ]
        return self._generate_signature(valid_blue)

    def get_results_from_dataset(self):
        """根据部署情况得到结果"""
        action_set = set(tuple(action) for action in self._episode_actions)
        for data in self.all_data:
            if data["scene"] == self.scenario_id:
 
                data = data["data"]
                current_bazhua = set(tuple(pos) for pos in data.get("bazhua_pos",[]))
                current_bigua = set(tuple(pos) for pos in data.get("bigua_pos", []))
                current_all_pos = current_bazhua.union(current_bigua)

                if current_all_pos == action_set:
                    return  data["blue_death"]             
        return None
      
      
        # 转换红方部署为动作序列
        return 
    
    def _map_red_positions(self, scene_pos):
        """映射红方部署位置到动作序号"""
        action_sequence = []
      
        # 创建位置映射表（提升查询性能）
        bigua_mapping = {
            self._normalize_red_position(pos): idx 
            for idx, pos in enumerate(self.red_bigua_pos)
        }
        bazhua_mapping = {
            self._normalize_red_position(pos): idx 
            for idx, pos in enumerate(self.red_bazhua_pos)
        }

        for idx in range(len(scene_pos)):
            single_scene_sequence = []
            # 处理壁挂部署（bigua）
            for data_pos in scene_pos[idx].get('bigua_pos', []):
                norm_pos = self._normalize_red_position(data_pos)
                if norm_pos in bigua_mapping:
                    single_scene_sequence.append(bigua_mapping[norm_pos])
        
            # 处理八爪部署（bazhua，序号接续）
            for data_pos in scene_pos[idx].get('bazhua_pos', []):
                norm_pos = self._normalize_red_position(data_pos)
                if norm_pos in bazhua_mapping:
                    single_scene_sequence.append(len(self.red_bigua_pos) + bazhua_mapping[norm_pos])
            action_sequence.append(single_scene_sequence)
        return action_sequence

    def _normalize_red_position(self, pos):
        """标准化红方位置格式（4位坐标，6位小数）"""
        return (
            round(pos[0], 6),
            round(pos[1], 6),
            round(pos[2], 6),
            int(pos[3])
        )

    def step(self, action):
        # TODO:#实际action

        self.update_action_mask(action)
        c_action = self.convert_action(action)
        self._episode_actions.append(c_action[:-1])
        self.update_state(c_action, self.actual_blue_deploy_pos)
        
        # info信息处理
        self.update_info(c_action)

        if self._step_index < self._episode_step_num -1:
            self._step_index += 1
            return {"obs": self.state, "action_mask": self._action_mask}, 0, False, False, self.info
        
        else:
            blue_deaths = self.get_results_from_dataset()
            if blue_deaths is None:
                print("未匹配对局推演数据")
                reward = 0
            else:
                reward = self.get_reward(blue_deaths)
            return {"obs": self.state, "action_mask": self._action_mask}, reward, True, False, self.info


    def update_action_mask(self, action):
        self._action_mask[action] = 0
        # if bigua deploy,set
        # if self._step_index == self.num_red_bigua - 1:       #壁挂部署完毕后，仅开放八爪actionmask
        #     self._action_mask[: len(self.red_bigua_pos)] = 0 
        #     self._action_mask[len(self.red_bigua_pos) :] = 1

        #有方案限制的情况
        if self._step_index == self.num_red_bigua - 1:       
            self._action_mask[:] = 0
            self._action_mask[self.restrict_num[self.scenario_id]] = 1
            self._action_mask[: len(self.red_bigua_pos)] = 0 
            

    def close(self):
        pass

    def to_xyz(self, lon, lat, alt, offsets=(644911.9272301428, 1769688.8120349043, 0.0)):
        x,y,z = self.transformer_geocentric.transform(lon,lat,alt)
        x -= offsets[0]
        y -= offsets[1]
        z -= offsets[2]
        return x,y,z

    def _compare_positions(self, actual, historical, epsilon=1e-6):
        """带误差范围的位置比较"""
        if len(actual) != len(historical):
            return False
    
        # 对每个坐标进行近似比较
        sorted_actual = sorted(actual)
        sorted_historical = sorted(historical)
    
        for a, h in zip(sorted_actual, sorted_historical):
            if not all(abs(a_dim - h_dim) < epsilon for a_dim, h_dim in zip(a, h)):
                return False
        return True

    def check_deploy_pos(self, episode_actions):
        """新版部署检查函数
        Args:
            episode_actions: 当前回合红方部署动作序列
        
        Returns:
            tuple: (blue_death, result) 或 (None, None)
        """
        # 将动作转换为标准位置格式（前4位坐标+类型）
        red_deployments = {
            'bazhua': [tuple(act[:4]) for act in episode_actions if act[4] == 2],  # 动作类型2是八爪
            'bigua': [tuple(act[:4]) for act in episode_actions if act[4] == 1]    # 类型1是壁挂
        }
    
        # 转换蓝方部署为可比较格式（忽略时间相关字段）
        blue_deployments = [tuple(pos[:6]) for pos in self.actual_blue_deploy_pos]

        # 在历史数据中寻找完全匹配项
        for data in self.all_data:
            # 标准化历史数据格式
            historical_red = {
                'bazhua': [tuple(pos[:4]) for pos in data.get('bazhua_pos', [])],
                'bigua': [tuple(pos[:4]) for pos in data.get('bigua_pos', [])]
            }
            historical_blue = [tuple(pos[:6]) for pos in data.get('blue_deploy_pos', [])]

            # 精确匹配检查
            if (self._compare_positions(red_deployments['bazhua'], historical_red['bazhua']) and
                self._compare_positions(red_deployments['bigua'], historical_red['bigua']) and
                self._compare_positions(blue_deployments, historical_blue)):
                return data.get('blue_death', 0), data.get('result', 0)
    
        return None, None  # 无匹配数据

    def update_info(self, c_action):
        """更新敌我属性信息"""
        if c_action[-1] == 1:
            self.info["red"]["BG"] += 1
        elif c_action[-1] == 2:
            self.info["red"]["BZ"] += 1
        

    def update_units_state(self, frame):
        """
        更新状态字典
        Args:
            frame_data: 新的一帧数据
        """
        new_objects = frame.objects

        for object_name, object_data in new_objects.items():
            self.units_state[object_name] = copy.deepcopy(object_data)

    def count_casualties(self):
        """
        统计红蓝双方死亡数量
        Returns:
            tuple: (红方死亡数, 蓝方死亡数)
        """
        red_deaths = 0
        blue_deaths = 0

        for unit in self.units_state.values():
            # 判断生命值为0且阵营为红方
            if unit.lifeValue == 0:  # 默认值1确保数据缺失时不计入死亡
                if unit.objectCamp == 0:
                    red_deaths += 1
                elif unit.objectCamp == 1:
                    blue_deaths += 1
        return blue_deaths, red_deaths

    def is_done(self, blue_deaths, red_deaths, start_time):
        """
        判断是否结束
        Args:
            blue_deaths: 蓝方死亡数
        Returns:
            bool: 是否结束
        """
        # 红方死亡数等于红方总数或蓝方死亡数等于蓝方总数或最后frame的时间戳-开始时间戳>最大时间
        # TODO: modify the conditino to blue_death = total exclude fixed_blue
        return (
            self.is_win(blue_deaths)
            or red_deaths == self.num_red_bigua + self.num_red_bazhua
            or time.time() - start_time > self.max_time
            # TODO: blue_group all arrive the end
            or self.is_blue_group_all_arrived(self.blue_unit_target)
        )

    def is_blue_group_all_arrived(self, blue_unit_target):
        if self.units_state == {}:
            return False
        for group in blue_unit_target:
            for blue_unit in group:
                if blue_unit['name'] in self.units_state.keys():
                    lat = self.units_state[blue_unit['name']].objectPosition.lat
                    lon = self.units_state[blue_unit['name']].objectPosition.lon
                    if(abs(lat - blue_unit['targets'][0]) > 1e-3 or abs(lon - blue_unit['targets'][1]) > 1e-3):
                        if self.units_state[blue_unit['name']].lifeValue == 0:
                            continue
                        return False
        return True

    def is_win(self, blue_deaths):
        num_person = 0
        num_vehicle = 0
        num_blue = 0
        for blue_deploy_pos in self.actual_blue_deploy_pos:
            num_person += blue_deploy_pos[4]
            num_vehicle += blue_deploy_pos[5]
        self.num_blue = num_person + num_vehicle
        return blue_deaths == num_blue
    
    def get_reward(self, blue_deaths):
        return blue_deaths * 5 - 400


    def get_observation(self,actual_blue_deploy_pos):
        observation = []
        

        #地图id获取
        observation.append(self.map_id)
        for pos in self.actual_blue_deploy_pos:
            observation.extend([pos[0], pos[1], pos[4], pos[5]])
        # TODO: judge the pos in actual_blue_deployed_pos
        
        # observation.extend([actual_blue_deploy_pos[0][0], actual_blue_deploy_pos[0][1], 0, 0])
                # observation.extend([pos[0], pos[1], 0, 0])
        for pos in self.red_bigua_pos + self.red_bazhua_pos:
            observation.extend([pos[0], pos[1],pos[2], 0])

        return np.array(observation)


    def convert_action(self, action):
        action_type = 1 if self._step_index < self.num_red_bigua else 2
        c_action = list(self.conv_pos[action])
        c_action.append(action_type)

        return c_action

    def update_state(self, action, actual_blue_deploy_pos):
        # 计算蓝方和红方的数量
        n_blue = len(actual_blue_deploy_pos)
        n_red = len(self.red_bazhua_pos) + len(self.red_bigua_pos)

        # 提取红方部署状态（跳过蓝方的部分）
        red_start = 4 * n_blue + 1  # 蓝方部分的长度
        red_state = self.state[red_start:].reshape(-1, 4)  # 重组成[n_red, 3]的形状

        # 遍历红方部署点，查找匹配的位置并更新
        for i in range(len(red_state)):
            if abs(red_state[i][0] - action[0]) < 1e-8 and abs(red_state[i][1] - action[1]) < 1e-8:
                red_state[i][2] = action[4]  # 更新部署状态
                break

        # 将更新后的红方状态展平并更新回原状态
        self.state[red_start:] = red_state.flatten()

    def action_mapping(self, red_bazhua_pos, red_bigua_pos):
        """
        动作空间的序号映射成实际部署位置
        """
        self.conv_pos = {}
        index = 0
        # 映射bigua部署点
        for pos in red_bigua_pos:
            self.conv_pos[index] = pos
            index += 1
        # 映射bazhua部署点
        for pos in red_bazhua_pos:
            self.conv_pos[index] = pos
            index += 1

    
    def generate_blue_deployment(self, total_men, total_vehicles, scenario_id):
        """
        生成蓝方部署信息
        
        Args:
            total_men (int): 蓝方总人数
            total_vehicles (int): 蓝方总车辆数
            scenario_id (int): 场景编号(1-15)
        
        Returns:
            list: 六个蓝方部署点的信息列表，每个元素为(经度, 纬度, 高度, 朝向, 人数, 车数)
        """
        # 基础部署点信息(经度, 纬度, 高度, 朝向)，来自config["blue_deploy_pos"]
        base_positions = [
            (25.0413175, 121.5577695, 17.5595397949219, 180),
            (25.0376394, 121.5576568, 17.6111297607422, 90),
            (25.0330752, 121.5575369, 17.5547332763672, 0),
            (25.0328376, 121.5683780, 17.5545654296875, 270),
            (25.0390337, 121.5698793, 17.6278381347656, 180),
            (25.0411706, 121.5645556, 17.5545806884766, 90)
        ]
        
        # 场景配置 - 表示每个场景激活的点位索引(0-5)
        scenario_configs = {
            0: [0, 1, 2, 3, 4, 5], # 全方位
            1: [2, 3],          # 下侧(3,4)
            2: [0, 1, 2],       # 左侧(1,2,3)
            3: [3, 4],       # 右侧(4,5,6)
            4: [0, 4, 5],          # 上方(1,6)
            5: [0, 5, 1, 2],    # 上方+左侧
            6: [0, 5, 3, 4],    # 上方+右侧
            7: [0, 5, 2, 3],    # 上方+下侧
            8: [0, 1, 2, 3, 4], # 左侧+下侧+右侧(不含6)
            9: [1, 2, 3, 4, 5], # 左侧+下侧+右侧(不含1)
            10: [0, 1, 2, 4, 5], # 左侧+右侧(不含3,4)
            11: [0, 2, 3, 4, 5], # 上方(仅1)+下侧+右侧
            12: [1, 2, 3, 4],    # 左侧+下侧(不含1)
            13: [0, 1, 2, 3],    # 左侧+下侧(含1)
            14: [2, 3, 4, 5],    # 下侧+右侧
        }
        
        # 检查场景编号是否有效
        if scenario_id not in scenario_configs:
            raise ValueError(f"无效的场景编号: {scenario_id}，有效范围为1-15")
        
        # 获取当前场景激活的点位
        active_positions = scenario_configs[scenario_id]
        num_active = len(active_positions)
        
        # 均分人数和车辆到激活的点位
        men_per_position = total_men // num_active
        men_remainder = total_men % num_active
        
        vehicles_per_position = total_vehicles // num_active
        vehicles_remainder = total_vehicles % num_active
        
        # 创建最终的部署信息
        deployment = []
        for i in range(6):
            if i in active_positions:
                # 分配人数(考虑余数)
                position_men = men_per_position
                if active_positions.index(i) < men_remainder:
                    position_men += 1
                    
                # 分配车辆(考虑余数)
                position_vehicles = vehicles_per_position
                if active_positions.index(i) < vehicles_remainder:
                    position_vehicles += 1
                    
                # 添加完整的部署信息
                pos_info = base_positions[i] + (position_men, position_vehicles)

            else:
                pos_info = base_positions[i] + (0,0)

            deployment.append(pos_info)
        
        return deployment

