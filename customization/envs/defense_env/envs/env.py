import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from customization.envs.defense_env.envs.simulator import WarChessGame

# TODO:修改观测空间和动作空间（主要是432）
SPACES_CONFIG = {
    "one_dim_grid": spaces.Dict(
        {
            "obs": spaces.Dict(
                {
                    "map": spaces.Box(
                        low=0, high=4, shape=(1, 180, 135), dtype=np.int64
                    ),
                    "semantic": spaces.Box(
                        low=np.zeros(160, dtype=np.int64),
                        high=np.full(160, 180, dtype=np.int64),
                        shape=(160,),
                        dtype=np.int64,
                    ),
                }
            ),
            "action_mask": spaces.Tuple(
                (
                    spaces.Box(low=0, high=1, shape=(40,), dtype=np.int8),
                    spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8),
                    spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8),
                )
            ),
        }
    ),
    "circle_decision": spaces.MultiDiscrete([40, 6, 6]),
}


class BattleEnv(gym.Env):
    def __init__(self, config) -> None:
        super(BattleEnv, self).__init__()

        self._config = config

        self._simulator = WarChessGame(config)

        self._observation_space_type = config["state_space"]
        self.observation_space = SPACES_CONFIG[config["state_space"]]
        self._action_space_name_type = config["action_space"]
        self.action_space = SPACES_CONFIG[config["action_space"]]

        self._done = False
        self._result = -1
        self.render_mode = config.get("render_mode", None)
        self.num_far_turret = config["num_far_turret"]
        self.num_near_turret = config["num_near_turret"]
        self._episode_step_num = self.num_far_turret + self.num_near_turret
        self._step_index = 0
        self.actions = []

        # TODO: 初始化action_mask
        self.action_mask = (
            np.ones(40, dtype=np.int8),
            np.ones(6, dtype=np.int8),
            np.ones(6, dtype=np.int8),
        )

    # TODO：通过options传递设置参数，但是训练时场景参数不能这样实现
    def reset(self, seed=None, options=None):
        self._step_index = 0
        self._done = False
        self._result = -1
        blue_style = random.randint(1, 2)
        self._simulator.reset(render_mode=self.render_mode, style=blue_style)
        self._simulator.blue_team_deployment(blue_style)
        self._state = self._get_state()
        self.actions = []
        # TODO:初始化action_mask
        return {"obs": self._state, "action_mask": self.action_mask}, {}

    # TODO:实现接收一个动作
    def step(self, action):
        self._step_index += 1
        action = self._mapping_action(action)
        self.actions.append(action)
        # TODO：实现更新部署状态[基于PPO现有实现迁移]
        current_action_dim1 = action[0]
        self._update_state(current_action_dim1)

        # TODO:更新action_mask,基于动作以及部署塔的类型
        self._update_action_mask(current_action_dim1)

        if self._step_index < self._episode_step_num:
            return (
                {"obs": self._state, "action_mask": self.action_mask},
                0,
                False,
                False,
                {},
            )

        self._simulator.red_team_deployment(self.actions)
        summary = self._simulator.simulate_battle()
        reward = self._get_reward(summary)
        truncated = False
        return (
            {"obs": self._state, "action_mask": self.action_mask},
            reward,
            True,
            False,
            summary,
        )

    def _update_state(self, action):
        """
        根据动作选取更新状态state
        """
        reshape_state = self._state["semantic"][-120:].reshape(40, 3)  # 变形成（x,y,n)
        red_deployment = self._simulator._red_deployment_points
        x, y = red_deployment[action]
        # 遍历reshape_state，找到与 x, y 匹配的位置并更新n值为1，表示已部署
        for i in range(len(reshape_state)):
            # 需要找到与x, y匹配的部署位置
            if reshape_state[i, 0] == x and reshape_state[i, 1] == y:
                reshape_state[i, 2] = 1  # 更新n为1，表示该位置已部署
                break  # 找到对应位置后跳出循环

        # 更新回state的最后120项
        self._state["semantic"][-120:] = reshape_state.flatten()

    def _update_action_mask(self, action_dim1):
        """
        基于动作进行掩码更新
        """
        # 将现有的mask转换为列表，以便修改
        mask_1, mask_2, mask_3 = (
            list(self.action_mask[0]),
            list(self.action_mask[1]),
            list(self.action_mask[2]),
        )

        # 修改第一个维度的掩码
        mask_1[action_dim1] = 0

        # 近程塔和远程塔的掩码更新
        if self._step_index <= self.num_near_turret:
            mask_3[-1] = 0  # 近程塔无法打击360度范围
        else:
            mask_3[-1] = 1  # 远程塔可以打击

        # 将修改后的掩码转换为numpy数组并组成新的tuple
        self.action_mask = (
            np.array(mask_1, dtype=np.int8),
            np.array(mask_2, dtype=np.int8),
            np.array(mask_3, dtype=np.int8),
        )

    def render(self):
        pass

    def simplify_enemy_deploy(self, enemy_deploy):
        """将敌人分布简化
        Returns:
            _type_: 形状为(n, p)的敌人分布状态, n为聚类后的敌人出生点,p为此区域的敌人总和
        """
        labels = self._simulator.enemy_labels
        # 统计每个聚类的敌人总数
        unique_labels = self._simulator.enemy_cluster
        simplified_deploy = []
        for label in unique_labels:
            region_points = enemy_deploy[labels == label]
            type1_sum = np.sum(region_points[:, 2])  # 第一种敌人的总数
            type2_sum = np.sum(region_points[:, 3])  # 第二种敌人的总数
            simplified_deploy.append((type1_sum, type2_sum))

        return np.array(simplified_deploy)

    def _get_state(self):
        map = self._simulator.get_map()
        map[map == 2] = 1
        map[map == 3] = 1
        map[map == 4] = 0
        map[map == 5] = 0
        map[map == 6] = 0
        enemy_deploy = self._simulator.get_enemy_deploy()
        # -------------聚类敌人分布设置------------------
        enemy_deploy = self.simplify_enemy_deploy(enemy_deploy)  ##形状（20,2）
        # #形状从(20,2)扩充为(20,3)
        # enemy_deploy = np.concatenate([enemy_deploy, np.zeros((enemy_deploy.shape[0], 1))], axis=1)
        # --------------------------------------------
        red_deploy = self._simulator.get_red_deploy()

        ##分别展平后再拼接
        enemy_deploy_flattened = enemy_deploy.flatten()  # 展平为一维数组
        red_deploy_flattened = red_deploy.flatten()  # 展平为一维数组

        map = map[np.newaxis, ...]
        ##特征展平为一维
        semantic = np.concatenate([enemy_deploy_flattened, red_deploy_flattened])

        return {"map": map, "semantic": semantic}

    def _mapping_action(self, action):
        return action

    def _get_reward(self, summary):
        reward_option = 3
        # if summary['result']==0:
        #     return 0
        if reward_option == 1:
            # 火力覆盖路径奖励
            route_record_in_map = np.zeros_like(self._simulator.firepower_coverage)
            for x, y in self._simulator.route_records:
                route_record_in_map[y][x] += 1
            max_value = np.max(route_record_in_map)
            route_record_in_map_normalized = route_record_in_map / max_value
            attack_reward = np.multiply(
                self._simulator.total_firepower_coverage, route_record_in_map_normalized
            )
            reward = np.sum(attack_reward) * 0.3
            if summary["result"] == 0:
                reward += summary["blue_evacuated"] * -5

        elif reward_option == 2:
            # 蓝方撤离奖励
            reward = summary["blue_evacuated"] * (-5)

        elif reward_option == 3:
            # 蓝方击杀数
            reward = summary["blue_dead"]
            if summary["result"] == 0:
                reward -= 80
        elif reward_option == 4:
            # 胜负奖励
            if summary["result"] == 1:
                reward = 100
            else:
                reward = -100
        # for red_turrent in self._simulator.red_deployment_info :
        #     abs_angle = abs(red_turrent[1] - red_turrent[2])
        #     angle_reward += abs_angle   #攻击范围的奖励
        # no_attack_reward = self._simulator.total_no_under_attack
        # reward = no_attack_reward   #无伤攻击的奖励
        # if summary['result'] == 0 :
        #     reward -= 100
        # return summary["blue_evacuated"] * (-5)
        return reward

    def _update_terrain(self, terrain, enemy_positions, max_radius=5):
        """
        更新二维地形数组，扩展敌人覆盖范围。

        参数:
        terrain (2D np.array): 二维地形数组，值为0, 1, 2, 3, 4表示不同地形。
        enemy_positions (list of tuples): 敌人位置的列表，每个元素为(x, y, n)，其中x, y是二维数组下标，n是敌人数量。
        max_radius (int): 最大覆盖半径（范围）。

        返回:
        2D np.array: 更新后的二维地形数组。
        """
        # 计算敌人数量的最大值和最小值用于归一化
        max_enemies = max(enemy[2] for enemy in enemy_positions)

        for x, y, n in enemy_positions:
            # 归一化敌人数量并计算覆盖范围半径
            normalized_n = n / max_enemies  # 归一化到[0, 1]之间
            radius = int(normalized_n * max_radius)  # 映射到[0, max_radius]之间

            # 遍历覆盖范围内的所有格子
            for i in range(max(0, x - radius), min(terrain.shape[0], x + radius + 1)):
                for j in range(
                    max(0, y - radius), min(terrain.shape[1], y + radius + 1)
                ):
                    # 更新地形值为4
                    terrain[i, j] = 3

        return terrain
