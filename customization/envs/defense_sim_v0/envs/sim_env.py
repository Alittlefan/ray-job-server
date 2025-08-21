import copy
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from tcp_comm import *

from customization.envs.defense_sim_v0.envs.data_types import ScenarioConfig
from customization.envs.defense_sim_v0.envs.maps.simulate_scene import Scene
from customization.envs.defense_sim_v0.envs.scenario_manager import ScenarioManager


class DefenseGymV0(gym.Env):
    def __init__(self, config=None):
        super().__init__()

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
            self.client = Scene()

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
                        4 * config["num_blue_group"]
                        + 3
                        * (
                            len(config["red_bazhua_pos"]) + len(config["red_bigua_pos"])
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

    def render(self):
        pass

    def reset(self, seed=None, options=None):
        # 重置场景
        self.scenario.reset()
        self._episode_actions = []
        self._step_index = 0

        # 创建蓝方编组
        self.scenario.create_blue_formations()

        # 获取初始观察
        self.state = self.get_observation(self.scenario.blue_formations)
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

        # 创建红方单位并开始仿真
        self.scenario.create_red_turrets(self._episode_actions)
        # 生成消息（bky专属）
        if self.sim_type == 1:
            self.scenario.generate_unit_messages()

        # 运行仿真
        reward, is_done, is_win = self.run_simulation()

        return (
            {"obs": self.state, "action_mask": self._action_mask},
            reward,
            True,
            False,
            {"is_done": is_done, "is_win": is_win},
        )

    def close(self):
        if self.sim_type == 1:
            self.client.disconnect()

    def update_state(self, action):
        # 计算蓝方和红方的数量
        n_blue = self.scenario.config.num_blue_group
        n_red = len(self.scenario.config.red_bazhua_pos) + len(
            self.scenario.config.red_bigua_pos
        )

        # 提取红方部署状态（跳过蓝方的部分）
        red_start = 4 * n_blue  # 蓝方部分的长度
        red_state = self.state[red_start:].reshape(-1, 3)  # 重组成[n_red, 3]的形状

        # 遍历红方部署点，查找匹配的位置并更新
        for i in range(len(red_state)):
            if (
                abs(red_state[i][0] - action[0]) < 1e-8
                and abs(red_state[i][1] - action[1]) < 1e-8
            ):
                red_state[i][2] = action[4]  # 更新部署状态
                break

        # 将更新后的红方状态展平并更新回原状态
        self.state[red_start:] = red_state.flatten()

    def get_observation(self, blue_formations):
        """获取观察"""
        obs = []

        # 蓝方位置
        for formation in blue_formations:
            # TODO:经纬转成x,y
            obs.extend(tuple(formation.start_point[:2]))
            obs.append(len(formation.personnel_list))
            obs.append(len(formation.vehicle_list))

        # 红方位置
        for turret in (
            self.scenario.config.red_bigua_pos + self.scenario.config.red_bazhua_pos
        ):
            obs.extend(tuple(turret[:3]))

        return np.array(obs, dtype=np.float32)

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
        if self.sim_type == 1:
            self.client.start_episode(self.scenario.messages)
            self.client.send_message(
                self.client._world_client, SimulationSpeedInfoMsg(time.time(), speed=16)
            )
            start_time = time.time()
            blue_deaths, red_deaths = 0, 0
            while True:
                frame = self.client.get_frame()
                self.update_units_state(frame) if frame else {}
                time.sleep(0.5)
                blue_deaths, red_deaths = self.count_casualties()
                is_done = self.is_done(blue_deaths, red_deaths, start_time)
                if is_done:
                    break
            # 判断是否胜利
            win = self.is_win(blue_deaths)
            result = self.client.end_episode()
        else:
            infantry_deaths, vehicle_deaths, win = self.client.start_episode(
                self.scenario
            )
            is_done = True
            blue_deaths = infantry_deaths + vehicle_deaths

        # 计算奖励
        reward = self.calculate_reward(blue_deaths, win)
        return reward, is_done, win

    def calculate_reward(self, blue_deaths, win):
        return 100 if win else -100

    def is_done(self, blue_deaths, red_deaths, start_time):
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
            or self.is_blue_group_all_arrived(self.scenario.blue_unit_target)
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
        num_person = 0
        num_vehicle = 0
        num_blue = 0
        for blue_formation in self.scenario.blue_formations:
            num_person += len(blue_formation.personnel_list)
            num_vehicle += len(blue_formation.vehicle_list)
        num_blue = num_person + num_vehicle
        # print(f"num_blue:{num_blue},blue_deaths:{blue_deaths}")
        return blue_deaths == num_blue

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
