import copy
import random
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tcp_comm import *
from tcp_comm.message import SimulationSpeedInfoMsg

from customization.envs.defense_sim.envs.data_types import ScenarioConfig
from customization.envs.defense_sim.envs.scenario_manager_new import ScenarioManager


class DefenseGymV0(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        # 创建场景配置
        scenario_config = ScenarioConfig(
            time_limit=config["max_time"],
            blue_deploy_pos=config["blue_deploy_pos"],
            blue_target_pos=config["blue_target_pos"],
            red_bigua_pos=config["red_bigua_pos"],
            red_bazhua_pos=config["red_bazhua_pos"],
            num_blue_group=config["num_blue_group"],
            num_blue_man=config["num_blue_man"],
            num_blue_vehicle=config["num_blue_vehicle"],
            num_red_bigua=config["num_red_bigua"],
            num_red_bazhua=config["num_red_bazhua"],
        )

        # 地图选择
        self.map_id = config["map_id"]
        # 创建场景管理器
        self.scenario = ScenarioManager(scenario_config)
        # 仿真类型: bky(1) or bq(2)
        self.sim_type = config["sim_type"]

        # 创建通信客户端（bky or bq）
        if self.sim_type == 1:
            self.client = CommClient(config["client"])
            if not self.client.connected():
                self.client.connect()
        else:
            return  

        # 设置动作和观察空间
        self.action_space = spaces.Discrete(
            len(config["red_bigua_pos"]) + len(config["red_bazhua_pos"])
        )

        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        1  # 地图id
                        + 4 * config["num_blue_group"]  # (4  经+纬+人数+车数)
                        + 4  # (3经纬高+1 是否使用)
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

        # 其他初始化
        self._episode_step_num = config["num_red_bazhua"] + config["num_red_bigua"]
        self._episode_actions = []
        self._step_index = 0
        self.num_blue = 0
        self.state = []
        self.blue_target_pos = config["blue_target_pos"]
        # 以下为遍历蓝方部署所需内容与参数
        # json_path="envs\\blue_force2_deployments.json"
        # with open(json_path,"rb") as f:
        #     file_content=f.read()
        # all_scenarios = orjson.loads(file_content)

    # def reset(self, seed=None, options=None):
    #     #0~1随机选择蓝方风格
    #     style=random.randint(0,1)
    #     # 重置场景
    #     self.scenario.reset()
    #     self._episode_actions = []
    #     self._step_index = 0

    #     # 创建蓝方编组
    #     self.scenario.create_blue_formations(style)

    #     # 获取初始观察
    #     self.state = self.get_observation(self.scenario.blue_formations)
    #     self._action_mask = np.ones(
    #         len(self.scenario.config.red_bigua_pos)
    #         + len(self.scenario.config.red_bazhua_pos),
    #         dtype=np.int8,
    #     )
    #     self._action_mask[len(self.scenario.config.red_bigua_pos) :] = 0   #先壁挂后八爪
    #     return {"obs": self.state, "action_mask": self._action_mask}, {}

    # 遍历全部蓝方部署所用reset
    def reset(self, seed=None, options=None):
        # 重置场景
        self.scenario.reset()
        self._episode_actions = []
        self._step_index = 0
        # 场景内蓝方兵力数量
        self.total_soldier = random.randint(
            self.config["total_soldier"][0], self.config["total_soldier"][1]
        )
        self.total_vehicle = random.randint(
            self.config["total_vehicle"][0], self.config["total_vehicle"][1]
        )
        self.total_enemy = self.total_soldier + self.total_vehicle
        # 场景类别
        self.scenario_id = random.randint(0, 4)

        self.blue_deploy_pos, self.all_blue_deploy_pos = self.generate_blue_deployment(
            self.total_soldier, self.total_vehicle, self.scenario_id
        )

        # 获取初始观察
        self.state = self.get_observation()
        self._action_mask = np.ones(
            len(self.scenario.config.red_bigua_pos)
            + len(self.scenario.config.red_bazhua_pos),
            dtype=np.int8,
        )
        self._action_mask[len(self.scenario.config.red_bigua_pos) :] = 0  # 先壁挂后八爪

        return {"obs": self.state, "action_mask": self._action_mask}, {}

    def step(self, action):
        # 更新动作掩码
        self.update_action_mask(action)

        # 转换并记录动作
        # 选取到的带有壁挂八爪类型的位置点信息
        c_action = self.convert_action(action)
        self._episode_actions.append(c_action)

        # 更新状态
        self.update_state(c_action)

        if self._step_index < self._episode_step_num - 1:
            self._step_index += 1
            return (
                {"obs": self.state, "action_mask": self._action_mask},
                0,
                False,
                False,
                {"is_done": False, "is_win": False},
            )
        # 生成消息（bky专属）
        if self.sim_type == 1:
            # self.scenario.generate_unit_messages()
            # 创建蓝方
            self.scenario.create_blue_units(self.blue_deploy_pos, self.blue_target_pos)
            # 创建红方单位并开始仿真
            self.scenario.create_red_turrets(self._episode_actions)
            self.scenario._generate_red_messages()

        # 运行仿真
        reward, is_done, is_win, kill_rate = self.run_simulation()

        return (
            {"obs": self.state, "action_mask": self._action_mask},
            reward,
            True,
            False,
            {"is_done": is_done, "is_win": is_win, "kill_rate": kill_rate},
        )

    def close(self):
        if self.sim_type == 1:
            self.client.disconnect()

    def update_state(self, action):
        # #构造炮塔标识，取经纬高,并标识为1，已部署
        # turret_identifier=(action [0],action[1],action[2],1)
        # #更新炮塔集合
        # self.scenario.deployed_turrets.add(turret_identifier)

        # 计算蓝方和红方的数量
        n_blue = self.scenario.config.num_blue_group
        n_red = len(self.scenario.config.red_bazhua_pos) + len(
            self.scenario.config.red_bigua_pos
        )

        # 提取红方部署状态（跳过蓝方的部分和地图id）
        red_start = 4 * n_blue + 1  # 蓝方部分的长度
        # red_state = self.state[red_start:].reshape(-1, 3)  # 重组成[n_red, 3]的形状
        red_state = self.state[red_start:].reshape(-1, 4)

        # # 遍历红方部署点，查找匹配的位置并更新
        # for i in range(len(red_state)):
        #     if (
        #         abs(red_state[i][0] - action[0]) < 1e-8
        #         and abs(red_state[i][1] - action[1]) < 1e-8
        #     ):
        #         red_state[i][2] = action[4]  # 更新部署状态
        #         break

        # 遍历红方部署点，查找匹配的位置并更新
        for i in range(len(red_state)):
            if (
                abs(red_state[i][0] - action[0]) < 1e-5
                and abs(red_state[i][1] - action[1]) < 1e-5
                # 高度可有可无因为确定经纬度就可以确定一个红方部署
                and abs(red_state[i][2] - action[2]) < 1e-5
            ):
                # 0代表未部署 1代表部署八爪 2代表壁挂
                # action[3]为朝向角
                red_state[i][3] = action[4]  # 更新部署状态
                break
        # action[4]拿到八爪 ，但是似乎没有成功更新到red_state中 对比后发现可能是1e-8太大了

        # 将更新后的红方状态展平并更新回原状态
        self.state[red_start:] = red_state.flatten()

    # def get_observation(self, blue_formations):
    #     """获取观察"""
    #     obs = []

    #     #地图id获取
    #     obs.append(self.map_id)
    #     # 蓝方位置
    #     for formation in blue_formations:
    #         # TODO:经纬转成x,y
    #         obs.extend(tuple(formation.start_point[:2]))
    #         obs.append(len(formation.personnel_list))
    #         obs.append(len(formation.vehicle_list))

    #     # 红方位置
    #     # deployed_set=getattr(self.scenario,"deployed_turrets",set())
    #     #由于每次reset才调用本函数 所以flag都是0
    #     deployed_set=self.scenario.deployed_turrets
    #     for turret in (
    #         self.scenario.config.red_bigua_pos + self.scenario.config.red_bazhua_pos
    #     ):
    #         #经纬高 并没有要朝向
    #         turret_info=tuple(turret[:3])

    #         #新增标记
    #         deployed_set=self.scenario.deployed_turrets
    #         deployed_flag=1 if turret_info in deployed_set else 0

    #         # obs.extend(tuple(turret[:3]))
    #         obs.extend(turret_info + (deployed_flag,))

    #     return np.array(obs, dtype=np.float32)
    def get_observation(self):
        observation = []

        # 地图id获取
        observation.append(self.map_id)
        for pos in self.all_blue_deploy_pos:
            observation.extend([pos[0], pos[1], pos[4], pos[5]])
        # TODO: judge the pos in actual_blue_deployed_pos

        # observation.extend([actual_blue_deploy_pos[0][0], actual_blue_deploy_pos[0][1], 0, 0])
        # observation.extend([pos[0], pos[1], 0, 0])
        for pos in (
            self.scenario.config.red_bigua_pos + self.scenario.config.red_bazhua_pos
        ):
            observation.extend([pos[0], pos[1], pos[2], 0])

        return np.array(observation)

    def update_action_mask(self, action):
        self._action_mask[action] = 0
        # if bigua deploy,set
        if (
            self._step_index == self.scenario.config.num_red_bigua - 1
        ):  # 壁挂部署完毕后，仅开放八爪actionmask
            self._action_mask[: len(self.scenario.config.red_bigua_pos)] = 0
            self._action_mask[len(self.scenario.config.red_bigua_pos) :] = 1

        # print(f"-----{self._step_index}action_mask:{self._action_mask}-----")
        # print(f"action:{action}")

    def convert_action(self, action):
        action_type = 1 if self._step_index < self.scenario.config.num_red_bigua else 2
        c_action = list(self.scenario.conv_pos[action])
        c_action.append(action_type)
        return c_action

    def run_simulation(self):
        """运行仿真"""
        # （bky or bq）开始仿真
        is_win = False
        if self.sim_type == 1:
            self.client.send_message(
                self.client._world_client, SetMsgFrequencyMsg(time.time(), 160000)
            )
            self.client.start_episode(self.scenario.messages)
            self.client.send_message(
                self.client._world_client, SimulationSpeedInfoMsg(time.time(), speed=16)
            )
            start_time = time.time()
            blue_deaths, red_deaths = 0, 0
            while True:
                frame = self.client.get_frame()
                # 帧返回None情况
                if frame is None:
                    continue
                self.update_units_state(frame) if frame else {}
                time.sleep(0.1)
                blue_deaths, red_deaths = self.count_casualties()
                escape_num = self.calculate_escape_num()
                # print(f"死亡数：{blue_deaths}, 撤离数: {escape_num}")
                is_done = self.is_done(blue_deaths, red_deaths, start_time, escape_num)
                if is_done:
                    break
            # 判断是否胜利
            kill_all = self.is_win(blue_deaths)
            time_out = time.time() - start_time > self.scenario.config.time_limit
            if kill_all:
                print(f"蓝方歼灭！")
                is_win = True
            if time_out:
                print(f"时间结束，撤离{escape_num}个单位！")
                if escape_num < int(self.total_enemy * 0.1):
                    is_win = True
            # result = self.client.end_episode()
            # self.logger.info(f"本轮blue_death: {blue_deaths}")
            if is_win == 1:
                print("本轮胜利")
            else:
                print("本轮失败")
        else:
            infantry_deaths, vehicle_deaths, win = self.client.start_episode(
                self.scenario
            )
            is_done = True
            blue_deaths = infantry_deaths + vehicle_deaths

        # 计算奖励
        reward = self.calculate_reward(blue_deaths, escape_num, is_win)

        # 计算击杀占比
        kill_rate = 1 - float(escape_num / self.total_enemy)
        print(f"本轮击杀占比：{kill_rate*100}%")
        self.client.end_episode()
        return reward, True, is_win, kill_rate

    def calculate_reward(self, blue_deaths, escape_num, win):
        # reward = 100 if win else (int(self.total_enemy * 0.1) - escape_num ) * 10
        reward = self.total_enemy * 0.1 - escape_num
        return reward

    def is_done(self, blue_deaths, red_deaths, start_time, escape_num):
        """
        判断是否结束
        Args:
            blue_deaths: 蓝方死亡数
        Returns:
            bool: 是否结束
        """
        # 红方死亡数等于红方总数或蓝方死亡数等于蓝方总数或最后frame的时间戳-开始时间戳>最大时间
        return (
            self.is_win(blue_deaths)
            or red_deaths == len(self.scenario.red_turrets)
            or time.time() - start_time > self.scenario.config.time_limit
            # TODO: blue_group all arrive the end
            or (escape_num >= int(self.total_enemy * 0.1))
        )

    # def is_done_bq(self, blue_deaths, red_deaths, start_time):
    #     """
    #     判断是否结束
    #     Args:
    #         blue_deaths: 蓝方死亡数
    #     Returns:
    #         bool: 是否结束
    #     """
    #     # 红方死亡数等于红方总数或蓝方死亡数等于蓝方总数或最后frame的时间戳-开始时间戳>最大时间
    #     return (
    #         blue_deaths
    #         == self.scenario.config.num_blue_group * self.scenario.config.num_blue_man
    #         + self.scenario.config.num_blue_vehicle
    #         or red_deaths == len(self.scenario.red_turrets)
    #         or time.time() - start_time > self.scenario.config.time_limit
    #         # TODO: blue_group all arrive the end
    #         or self.is_blue_group_all_arrived_bq(self.scenario.blue_unit_target)
    #     )

    def is_win(self, blue_deaths):
        num_blue = int(self.total_enemy * 0.9)
        # print(f"num_blue:{num_blue},blue_deaths:{blue_deaths}")
        return blue_deaths >= num_blue

    def is_blue_group_all_arrived(self, blue_unit_target):
        """
        判断蓝方编组是否全部到达目标点
        Args:
            blue_unit_target: 蓝方编组目标点
        Returns:
            bool: 是否全部到达
        """
        if self.scenario.units_state == {}:
            return False
        for group in blue_unit_target:
            for blue_unit in group:
                if blue_unit["name"] in self.scenario.units_state.keys():
                    lat = self.scenario.units_state[
                        blue_unit["name"]
                    ].objectPosition.lat
                    lon = self.scenario.units_state[
                        blue_unit["name"]
                    ].objectPosition.lon
                    if (
                        abs(lat - blue_unit["targets"][1]) > 1e-3
                        or abs(lon - blue_unit["targets"][0]) > 1e-3
                    ):
                        if self.scenario.units_state[blue_unit["name"]].lifeValue == 0:
                            continue
                        return False
        return True

    def calculate_escape_num(self):
        escape_num = 0
        if self.scenario.units_state == {}:
            return False
        for group in self.scenario.blue_unit_target:
            for blue_unit in group:
                if self.scenario.units_state[blue_unit["name"]].lifeValue == 0:
                    continue
                lat = self.scenario.units_state[blue_unit["name"]].objectPosition.lat
                lon = self.scenario.units_state[blue_unit["name"]].objectPosition.lon
                if (
                    abs(lat - blue_unit["targets"][0]) <= 0.00105
                    and abs(lon - blue_unit["targets"][1]) <= 0.00043
                ):
                    escape_num += 1
        return escape_num

    # def is_blue_group_all_arrived_bq(self, blue_unit_target):
    #     """
    #     判断蓝方编组是否全部到达目标点
    #     Args:
    #         blue_unit_target: 蓝方编组目标点
    #     Returns:
    #         bool: 是否全部到达
    #     """
    #     if not self.scenario.units_state:
    #         return False

    #     for group in blue_unit_target:
    #         for blue_unit in group:
    #             unit_name = blue_unit["name"]
    #             if unit_name not in self.scenario.units_state:
    #                 continue

    #             unit = self.scenario.units_state[unit_name]
    #             if unit["lifevalue"] == 0:
    #                 continue

    #             lat = unit["position"]["lat"]
    #             lon = unit["position"]["lon"]
    #             target_lat, target_lon = blue_unit["targets"]

    #             if (abs(lat - target_lat) > 1e-3 or abs(lon - target_lon) > 1e-3):
    #                 return False

    #     return True

    def update_units_state(self, frame):
        """
        更新状态字典
        Args:
            frame_data: 新的一帧数据
        """
        new_objects = frame.objects if self.sim_type == 1 else frame

        for object_name, object_data in new_objects.items():
            self.scenario.units_state[object_name] = copy.deepcopy(object_data)

    def count_casualties(self):
        """
        统计红蓝双方死亡数量
        Returns:
            tuple: (红方死亡数, 蓝方死亡数)
        """
        red_deaths = 0
        blue_deaths = 0

        for unit in self.scenario.units_state.values():
            # 判断生命值为0且阵营为红方
            if unit.lifeValue == 0:  # 默认值1确保数据缺失时不计入死亡
                if unit.objectCamp == 0:
                    red_deaths += 1
                elif unit.objectCamp == 1:
                    blue_deaths += 1
        return blue_deaths, red_deaths

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
            (25.0411706, 121.5645556, 17.5545806884766, 90),
        ]

        # 场景配置 - 表示每个场景激活的点位索引(0-5)
        scenario_configs = {
            0: [0, 1, 2, 3, 4, 5],  # 全方位
            1: [2, 3],  # 下侧(3,4)
            2: [0, 1, 2],  # 左侧(1,2,3)
            3: [3, 4],  # 右侧(4,5,6)
            4: [0, 4, 5],  # 上方(1,6)
            5: [0, 5, 1, 2],  # 上方+左侧
            6: [0, 5, 3, 4],  # 上方+右侧
            7: [0, 5, 2, 3],  # 上方+下侧
            8: [0, 1, 2, 3, 4],  # 左侧+下侧+右侧(不含6)
            9: [1, 2, 3, 4, 5],  # 左侧+下侧+右侧(不含1)
            10: [0, 1, 2, 4, 5],  # 左侧+右侧(不含3,4)
            11: [0, 2, 3, 4, 5],  # 上方(仅1)+下侧+右侧
            12: [1, 2, 3, 4],  # 左侧+下侧(不含1)
            13: [0, 1, 2, 3],  # 左侧+下侧(含1)
            14: [2, 3, 4, 5],  # 下侧+右侧
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
        all_blue_deployment = []
        all_blue_info = []
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
                all_blue_info = base_positions[i] + (position_men, position_vehicles)
                deployment.append(pos_info)
            else:
                all_blue_info = base_positions[i] + (0, 0)

            all_blue_deployment.append(all_blue_info)
        return deployment, all_blue_deployment
