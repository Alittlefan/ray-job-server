import copy
import json
import math
import os
import random
from collections import defaultdict
from copy import deepcopy

import imageio
import numpy as np
import pandas as pd
import PIL
import pygame as pg
from PIL import Image

from customization.envs.defense_env.envs import draw, utils
from customization.envs.defense_env.envs.constants import STONE_STATE
from customization.envs.defense_env.envs.stone import *


class WarChessGame:
    # 初始化
    def __init__(self, config):
        # read config
        self._config = config

        self.render_mode = None  # 是否渲染

        # read map data and transpose the axis
        # 0:road, 1:building, 2:deploy_area, 3:red_deploy_point
        # 4:blue_start_point, 5:blue_target_point, 6:blue_route_point
        self._map = pd.read_csv(self._config["map_path"], header=None).values.T
        self._red_deployment_points = np.stack(np.where(self._map == 3), axis=1)
        self._blue_start_points = np.stack(np.where(self._map == 4), axis=1)
        self._blue_target_point = np.stack(np.where(self._map == 5), axis=1)
        self.new_map = copy.deepcopy(self._map)
        self.action_distribution = []

        self.screen = None
        self.cs_time = (
            0  # 用於統計渲染時的遊戲輪次，每進行一局+1。目前用於生成動態GIF名稱。
        )

        # read blue routes
        self._blue_routes = json.load(
            open(config["blue_routes_path"])
        )  # "resource/grounding_path.json"
        self.no_under_attack = 0
        # 数据分析记录
        self.red_deploy_count = defaultdict(int)  # 红方部署点单位数量统计
        self.firepower_coverage = np.zeros(
            (self._map.shape[1], self._map.shape[0]), dtype=float
        )  # 创建了一个和地图等大的全0二维数组，用于记录火力强度。
        self.total_firepower_coverage = np.zeros(
            (self._map.shape[1], self._map.shape[0]), dtype=int
        )  # 创建了一个和地图等大的全0二维数组，用于记录火力强度。
        self.route_records = []  # 蓝方路径记录
        self.start_array = []
        self.blue_deploy_rule = []  # 规则部署的接口提供
        # game state
        self._summary = {
            "red_dead": 0,
            "blue_dead": 0,
            "blue_evacuated": 0,
            "steps": 0,
            "result": -1,
        }

        # warchess team
        self._blue_team = []
        self._blue_team_preparation = []
        self._all_blue_team = []  # 不做删减的蓝方队伍，用来记录数据
        self._all_red_team = []  # 不做删减的红方队伍，用来记录数据
        self.blue_deployed_nums = defaultdict(lambda: defaultdict(int))
        self._red_team = []  # 红方单位对象列表
        self.fire_coverage_ratio = 0
        self._fixed_solider_stats = {}
        self.weight_fire = 0
        self.red_attack_times_per_turn = 0
        self.red_attack_successful_per_turn = 0
        self.blue_attack_times_per_turn = 0
        self.blue_attack_successful_per_turn = 0
        # 重制敌人出生聚类和出生点所属标签
        self.enemy_cluster = None
        self.enemy_labels = None

        # 分图层渲染相关的全局变量：
        self.background_surface = None  # 背景图层

    def reset(self, style, render_mode):
        # 重置统计信息：
        self._summary = {  # 用于存储游戏过程中的一些统计信息
            "red_dead": 0,
            "blue_dead": 0,
            "blue_evacuated": 0,
            "steps": 0,
            "result": -1,
        }
        self.action_distribution = []

        # 重置队伍列表
        self._blue_team = []
        self._blue_team_preparation = []
        self._all_blue_team = []  # 不做删减的蓝方队伍，用来记录数据
        self._all_red_team = []  # 不做删减的红方队伍，用来记录数据
        self._red_team = []
        # 重置部署和火力覆盖信息
        self.blue_deployed_nums = defaultdict(lambda: defaultdict(int))
        self.red_deploy_count = defaultdict(int)
        self.firepower_coverage = np.zeros(
            (self._map.shape[1], self._map.shape[0]), dtype=float
        )  # 重置火力强度二维数组为全0
        self.red_attack_times_per_turn = 0
        self.red_attack_successful_per_turn = 0
        self.blue_attack_times_per_turn = 0
        self.blue_attack_successful_per_turn = 0
        self.render_mode = render_mode  # 可视化模式设定
        self.no_under_attack = 0
        self.blue_deploy_rule = []
        self.style = style
        if self.render_mode == "human" and self.screen is None:
            pg.init()
            self.screen = pg.display.set_mode(
                (self._map.shape[0] * 5, self._map.shape[1] * 5)
            )  # (宽，高)，一个值变5像素？
            self.background_surface = pg.Surface(
                (self._map.shape[0] * 5, self._map.shape[1] * 5)
            )
            pg.display.set_caption("WarChess Battle Simulation")  # 给窗口定标题
        elif self.render_mode == "gif":
            os.makedirs("battle_gif/lose", exist_ok=True)
            os.makedirs("battle_gif/win", exist_ok=True)
            self.screenshots = []
            # 创建一个与屏幕尺寸相同的内存 Surface
            self.screen = pg.Surface((self._map.shape[0] * 5, self._map.shape[1] * 5))
            self.background_surface = pg.Surface(
                (self._map.shape[0] * 5, self._map.shape[1] * 5)
            )

        if self.render_mode is not None:
            self.draw_map()

        # 重置路线记录
        self.route_records = []
        # 12.16新增：
        self.route_record_in_map = np.zeros(
            (self._map.shape[1], self._map.shape[0]), dtype=int
        )
        self.total_firepower_coverage = np.zeros(
            (self._map.shape[1], self._map.shape[0]), dtype=int
        )  # 用于存取不消减的火力覆盖范围
        self.fire_coverage_ratio = 0
        self.weight_fire = 0
        # 重制敌人出生聚类和出生点所属标签
        self.get_enemy_cluster()

    # 渲染游戏画面########################################### 原
    def render(self):
        if self.render_mode is None:  # 若不渲染，则跳过该方法
            return
        # 清空屏幕
        self.screen.fill((255, 255, 255))

        self.screen.blit(self.background_surface, (0, 0))

        # 绘制火力覆盖
        self.draw_firepower_coverage()
        # 绘制红蓝方单位
        self.draw_units()

        if self.render_mode == "human":
            # 更新显示
            pg.display.flip()
            pg.time.delay(100)
        else:
            screenshot = pg.image.tostring(self.screen, "RGB")  # 将 Surface 转为字符串
            image = Image.frombytes(
                "RGB", self.screen.get_size(), screenshot
            )  # 转为 PIL 图像
            self.screenshots.append(image)  # 保存到帧列表

    # 绘制地图背景
    def draw_map(self):
        """绘制地图背景"""
        for x in range(self.new_map.shape[0]):
            for y in range(self.new_map.shape[1]):
                color = self.get_tile_color(self.new_map[x, y])
                rect = pg.Rect(x * 5, y * 5, 5, 5)
                pg.draw.rect(self.background_surface, color, rect)
                if self.render_mode == "human":
                    pg.draw.rect(self.screen, color, rect)
                elif self.render_mode == "gif":
                    self.screen.blit(self.background_surface, (0, 0))
        if self.render_mode == "gif":
            pg.image.save(self.screen, "resource/map2.png")

    # 根据地图值返回相应颜色
    def get_tile_color(self, tile_value):
        """根据地图值返回相应颜色"""
        if tile_value == 0:
            return (211, 211, 211)  # 浅灰色 (Light Gray)
        elif tile_value == 1:
            return (139, 69, 19)  # 棕色 (Saddle Brown)
        elif tile_value == 2:
            return (0, 0, 255)  # 蓝色 (Blue)
        elif tile_value == 3:
            return (255, 255, 255)  # 白色 (White)
        elif tile_value == 4:
            return (0, 255, 0)  # 绿色 (Green)
        elif tile_value == 5:
            return (128, 0, 128)  # 紫色 (Purple)
        elif tile_value == 6:
            return (173, 216, 230)  # 亮蓝色 (Light Blue)
        else:
            return (0, 0, 0)  # 默认颜色为黑色

    # 绘制火力覆盖区域
    def draw_firepower_coverage(
        self, color_f=[0, 0, 255], color_n=[0, 255, 0], color_b=[255, 0, 0]
    ):
        """
        绘制火力覆盖区域，使用一次性大Surface渲染
        :param color_f:([r,g,b])远程塔攻击范围颜色，有默认值[0, 0, 255]
        :param color_n:([r,g,b])近程塔攻击范围颜色，有默认值[0, 255, 0]
        :param color_b:([r,g,b])两种塔叠加区域的攻击范围颜色，有默认值[255, 0, 0]
        :return:无
        """

        max_intensity = 6  # 定义火力强度的最大值，用于归一化透明度
        cell_size = 5  # 每个单元格的大小

        # 创建一个足够大的Surface来存储整个地图
        big_surface = pg.Surface(
            (self._map.shape[0] * cell_size, self._map.shape[1] * cell_size),
            pg.SRCALPHA,
        )

        # 遍历地图中的每个点
        for y in range(self._map.shape[0]):
            for x in range(self._map.shape[1]):
                intensity = self.firepower_coverage[x, y]  # 获取当前点的火力强度

                if intensity > 0:
                    # 由于浮点数在计算机中的表示方式以及运算精度的问题，输入2.01时，这个值在内部首先被近似为一个略有不同但非常接近2.01的二进制浮点数，
                    # 从这个近似值中减去整数部分的2时，结果会是0.009999999999999787而非0.01，省去小数点两位后的部分就是0.00，从而导致渲染颜色错误。
                    # 所以采用字符串分割的方式，避免精度问题。
                    a, b = str(intensity).split(".")  # 获取远近火力强度
                    a, b = int(a), int(b)

                    # 根据火力强度计算归一化透明度
                    alpha = min(
                        255, int(((a + b) / max_intensity) * 255)
                    )  # 综合强度归一化透明度

                    # 根据火力类型确定颜色（准备颜色信息）
                    if a > 0 and b == 0:
                        color = color_f + [alpha]
                    elif b > 0 and a == 0:
                        color = color_n + [alpha]
                    else:  # a > 0 and b > 0
                        color = color_b + [alpha]

                    # 计算在大Surface中的位置（准备位置信息）
                    rect = pg.Rect(
                        y * cell_size, x * cell_size, cell_size, cell_size
                    )  # 左上坐标，长和宽

                    # 填充颜色到对应的区域（将颜色信息和位置信息放进big_surface 里）
                    big_surface.fill(color, rect)

        # 将整个大Surface绘制到屏幕上
        self.screen.blit(big_surface, (0, 0))

        return

    # 进行分层渲染的尝试：
    # # 绘制红蓝方单位
    # def draw_units(self):
    #     """绘制红蓝方单位"""
    #     # 假设 self._blue_team 和 self._red_team 存储着蓝方和红方的单位
    #     for unit in self._blue_team:
    #         self.draw_unit(unit, (255, 0, 0))  # 蓝色表示蓝方单位
    #     # 红方单位根据类型区分颜色
    #     for unit in self._red_team:
    #         if 'far' in unit.get_name():  # 如果名字中包含'far'
    #             self.draw_unit(unit, (255, 69, 0))  # 红色表示远程塔
    #         elif 'near' in unit.get_name():  # 如果名字中包含'near'
    #             self.draw_unit(unit, (255, 165, 0))  # 橙色表示近战塔

    # # 绘制单个单位
    # def draw_unit(self, unit, color):
    #     """绘制单个单位"""
    #     x = unit.get_posx()  # 假设单位有一个获取位置的方法
    #     y = unit.get_posy()
    #     rect = pg.Rect(x * 5, y * 5, 5, 5)
    #     pg.draw.rect(self.screen, color, rect)

    def draw_units(self):
        """绘制红蓝方单位"""
        # 计算蓝方单位的坐标分布
        blue_unit_positions = {}
        for unit in self._blue_team:  #
            pos = (unit.get_posx(), unit.get_posy())
            if pos not in blue_unit_positions:
                blue_unit_positions[pos] = []
            blue_unit_positions[pos].append(unit)

        # 绘制蓝方单位，根据单位数量调整透明度
        for pos, units in blue_unit_positions.items():
            unit_count = len(units)
            if unit_count <= 1:
                alpha = int(255 * 0.5)  # 40% 透明度
            elif 2 <= unit_count <= 3:
                alpha = int(255 * 0.8)  # 70% 透明度
            else:
                alpha = 255  # 100% 透明度

            for unit in units:

                # # 测试输出所有的单位名称（测试用，可删去）
                # with open(r"D:\PycharmProject\PythonProjectRL\1210-tobeiyou\battle_gif\x.txt", 'a',
                #           encoding='utf-8') as f:
                #     f.write(unit.get_name() + "， ")

                if "vehicle_" in unit.get_name():  # 如果名字中包含'vehicle_'
                    color = (0, 0, 0, alpha)  # 黑色 (R, G, B, A)
                    break
                else:  # 如果该格子没有车辆，则显示普通单位颜色
                    color = (255, 0, 0, alpha)  # 蓝色 (R, G, B, A)
            rect = pg.Rect(pos[0] * 5, pos[1] * 5, 5, 5)
            surface = pg.Surface((5, 5), pg.SRCALPHA)
            surface.fill(color)
            self.screen.blit(surface, rect)

        # 绘制红方单位
        for unit in self._red_team:
            if "far" in unit.get_name():  # 如果名字中包含'far'
                color = (255, 69, 0)  # 红色表示远程塔
            elif "near" in unit.get_name():  # 如果名字中包含'near'
                color = (255, 165, 0)  # 橙色表示近战塔
            else:
                continue  # 如果不符合规则，不绘制

            x = unit.get_posx()
            y = unit.get_posy()
            rect = pg.Rect(x * 5, y * 5, 5, 5)
            pg.draw.rect(self.screen, color, rect)

    #
    def get_red_deploy(self):
        return np.pad(
            self._red_deployment_points,
            ((0, 0), (0, 1)),
            mode="constant",
            constant_values=0,
        )  # 在列的末尾填充0

    #
    def get_enemy_deploy(self):
        # 将敌人部署点转换为(x, y, 'soldier', 'vehicle')的形式
        # -----------------------关键路口权重形式-----------------------------
        # epoch_data = []
        # for (x, y) in self._blue_start_points:
        #     single_enemy = self.path_getter.env_path_collect(int(x), int(y), self.blue_deployed_nums[(x, y)], is_Exp=True, selected_route=None)
        #     epoch_data.append(single_enemy)
        # enemy_deploy = RuleW.blue_path_w(epoch_data, mode="train")
        # return enemy_deploy
        # ----------------------(x, y, n)形式-------------------------------------------
        return np.array(
            [
                (
                    x,
                    y,
                    self.blue_deployed_nums[(x, y)]["soldier"],
                    self.blue_deployed_nums[(x, y)]["vehicle"],
                )
                for (x, y) in self._blue_start_points
            ]
        )

    # 获取地图
    def get_map(self):
        return self._map

    def get_enemy_cluster(self, eps=2, min_samples=1):
        """
        通过聚类算法实现对地图中敌人出生点的区域聚类
        """
        enemy_deploy = self.get_enemy_deploy()
        # 提取坐标部分
        coordinates = enemy_deploy[:, :2]  # 只取(x, y)
        # 使用DBSCAN进行聚类
        clustering = utils.cluster_coordinates(coordinates, eps, min_samples)
        self.enemy_labels = clustering.labels_
        # 统计每个聚类的敌人总数
        self.enemy_cluster = np.unique(self.enemy_labels)

    def get_fire_data(self):
        total_size = self.firepower_coverage.size
        fire_coverage_area = np.count_nonzero(
            self.firepower_coverage > 0
        )  # 火力范围大小
        fire_coverage_ratio = fire_coverage_area / total_size  # 火力覆盖率
        unique_values, counts = np.unique(
            self.firepower_coverage, return_counts=True
        )  # 火力强度统计
        coverage_intensity = {
            int(key): int(value) for key, value in zip(unique_values, counts)
        }
        return {
            "coverage_area": int(fire_coverage_area),  # 火力范围大小
            "coverage_ratio": float(fire_coverage_ratio),  # 火力覆盖率
            "coverage_intensity": coverage_intensity,  # 火力强度
        }

    def calculate_fire_coverage_in_area(self):  #
        """
        计算指定区域内非障碍物的火力覆盖率。

        :param firepower_coverage: 火力覆盖的二维数组，形状为 (y, x)，表示每个点的火力强度
        :param map_array: 地图的二维数组，形状为 (x, y)，其中1表示障碍物，其他值表示可通过区域
        :param x_range: (x_min, x_max) 区域的x轴范围
        :param y_range: (y_min, y_max) 区域的y轴范围
        :return: 指定区域内非障碍物的火力覆盖率
        """
        # 将 map_array 转置，使其形状与 firepower_coverage 一致 (y, x)
        transposed_map = np.transpose(self._map)

        # 提取指定区域的子矩阵

        # 获取非障碍物的掩码（True 表示不是障碍物的格子）
        non_obstacle_mask = transposed_map != 1

        # 计算非障碍物区域总面积
        non_obstacle_area = np.count_nonzero(non_obstacle_mask)

        if non_obstacle_area == 0:
            return 0  # 如果没有非障碍物区域，返回0

        # 获取火力覆盖的掩码（True 表示被火力覆盖的区域）
        fire_coverage_mask = self.total_firepower_coverage > 0
        blue_coverage_mask = self.route_record_in_map > 0
        # 获取非障碍物区域中被火力覆盖的区域
        fire_covered_non_obstacle = np.logical_and(
            non_obstacle_mask, fire_coverage_mask
        )
        fire_covered_routes = np.logical_and(
            fire_covered_non_obstacle, blue_coverage_mask
        )
        # 计算非障碍物区域中被火力覆盖的面积
        fire_covered_area = np.count_nonzero(fire_covered_routes)
        blue_routes_coverage = np.count_nonzero(blue_coverage_mask)
        # 计算火力覆盖率
        self.fire_coverage_ratio = fire_covered_area / (blue_routes_coverage + 1)

    def weight_fire_coverage(self):
        # 获取蓝方路径经过的格子次数数组
        total_fire_coverage = np.array(
            self.total_firepower_coverage
        )  # 红方炮塔攻击强度二维数组

        # 如果route_records是空的，返回0
        if self.route_record_in_map.size == 0 or total_fire_coverage.size == 0:
            return 0

        # 计算归一化后的路径格子数值（避免除以0）
        max_coverage = np.max(self.route_record_in_map.size)
        if max_coverage == 0:
            normalized_coverage = np.zeros_like(
                self.route_record_in_map.size
            )  # 如果最大覆盖为0，归一化后全为0
        else:
            normalized_coverage = (
                self.route_record_in_map.size / max_coverage
            )  # 归一化蓝方路径格子数

        # 对应位置的归一化路径覆盖值与红方炮塔攻击强度相乘
        weighted_coverage = normalized_coverage * total_fire_coverage

        # 将所有相乘的结果求和
        self.weight_fire = np.sum(weighted_coverage)

    # 蓝方部署风格
    def blue_team_deployment(self, blue_style):
        # 创建一个列表units_list，其中包含了一定数量的士兵名称，数量由配置文件中的 num_soldier 决定。
        units_list = [f"soldier_{idx}" for idx in range(self._config["num_soldier"])]
        units_list += [f"vehicle_{idx}" for idx in range(self._config["num_vehicle"])]

        # 这个函数用于将士兵添加到地图上，并根据给定的路由和条件来放置他们。（用于在下面判断完部署风格之后调用）
        #   遍历units列表中的每个单位，如果单位名称以'soldier_'开头，则会根据条件选择一个路由，然后更新蓝方出生点的人数统计，并为每个士兵创建一个包含其位置和能力的统计信息字典。
        #   该字典随后被用来创建一个SoldierStone对象（可能是代表士兵在游戏地图上的一个表示），这些对象被添加到_blue_team_preparation列表中。
        def extend_paths(blue_routes, n):
            # 先获取每个start一条路径
            paths = [
                {"start": route["start"], "path": [random.choice(route["path"])]}
                for route in blue_routes["routes"]
            ]

            # 前 n 个 start 添加一条额外的路径
            for i in range(min(n, len(paths))):
                available_paths = blue_routes["routes"][i]["path"]

                # 从可用路径中选择一条不同于已存在路径的路径
                current_paths = paths[i]["path"]
                remaining_paths = [
                    path for path in available_paths if path not in current_paths
                ]
                if remaining_paths:
                    # 添加新路径
                    current_paths.append(random.choice(remaining_paths))

            return paths

        def add_soldiers(
            units, routes, condition=None, is_remember=False, is_attack_move=False
        ):
            for unit_name in units:
                # 根据类型区分 soldier 和 vehicle
                if unit_name.startswith("soldier"):
                    unit_type = "soldier"
                    UnitClass = SoldierStone
                elif unit_name.startswith("vehicle"):
                    unit_type = "vehicle"
                    UnitClass = VehicleStone
                else:
                    raise ValueError(f"Unsupported unit type for {unit_name}")

                # 选择路线
                selected_route = (
                    random.choice(routes)
                    if condition is None
                    else random.choice([r for r in routes if condition(r)])
                )
                (x, y) = selected_route["start"]  # 蓝方出生点选取
                self.blue_deployed_nums[(x, y)][unit_type] += 1
                self.blue_deploy_rule.append((x, y))

                # 深拷贝选取的路径
                route = copy.deepcopy(random.choice(selected_route["path"]))

                # 动态读取对应的配置
                unit_stats = {
                    "name": unit_name,
                    "posx": x,
                    "posy": y,
                    "attack_power": self._config[f"{unit_type}_attack"],
                    "mobility": self._config[f"{unit_type}_mobility"],
                    "health": self._config[f"{unit_type}_health"],
                    "attack_range": self._config[f"{unit_type}_attack_range"],
                    "accuracy": self._config[f"{unit_type}_accuracy"],
                    "ammunition": self._config[f"{unit_type}_ammunition"],
                    "route": route,
                    "attack_prob": 0.8 if is_attack_move else 1,
                }

                if is_remember:
                    stats = self._fixed_solider_stats.get(unit_name, None)
                    if not stats:
                        stats = unit_stats
                        self._fixed_solider_stats[unit_name] = deepcopy(unit_stats)
                else:
                    stats = unit_stats

                # 记录路径
                self.route_records.extend(
                    draw.path_to_coords(
                        (stats["posx"], stats["posy"]), copy.deepcopy(stats["route"])
                    )
                )

                # 根据类型实例化 SoldierStone 或 VehicleStone
                self._blue_team_preparation.append(UnitClass(deepcopy(stats)))
                self._all_blue_team.append(UnitClass(deepcopy(stats)))

        # def add_soldiers(units, routes, condition=None, is_remeber=False, is_attack_move=False):
        #     # blue_path = path_save()
        #     # selected_blue_path = []
        #     for unit_name in units:
        #         if unit_name.startswith('soldier'):
        #             selected_route = (
        #                 random.choice(routes)
        #                 if condition is None
        #                 else random.choice([r for r in routes if condition(r)])
        #             )
        #             (x, y) = selected_route['start']  # 蓝方出生点选取
        #             self.blue_deployed_nums[(x, y)] += 1
        #             self.blue_deploy_rule.append((x, y))
        #             # 蓝方出生点人数统计
        #             route = copy.deepcopy(random.choice(selected_route['path']))
        #             # 获取到路径和初始点，进行转换坐标
        #             unit_stats = {
        #                 'name': unit_name,
        #                 'posx': x,
        #                 'posy': y,
        #                 'attack_power': self._config['soldier_attack'],
        #                 'mobility': self._config['soldier_mobility'],
        #                 'health': self._config['soldier_health'],
        #                 'attack_range': self._config['soldier_attack_range'],
        #                 'accuracy': self._config['soldier_accuracy'],
        #                 'ammunition': self._config['soldier_ammunition'],
        #                 'route': route,
        #                 'attack_prob': 0.8 if is_attack_move else 1,
        #             }
        #             # ---------------------------

        #             # 蓝方士兵路径选取信息获取
        #             # selected_blue_path.append(blue_path.env_path_collect(x, y, count=1, is_Exp=False, selected_route=route))

        #             # ----------------------------

        #             if is_remeber:
        #                 stats = self._fixed_solider_stats.get(unit_name, None)
        #                 if not stats:
        #                     stats = unit_stats
        #                     self._fixed_solider_stats[unit_name] = deepcopy(unit_stats)
        #             else:
        #                 stats = unit_stats
        #             self.route_records.extend(
        #                 draw.path_to_coords((stats['posx'], stats['posy']), copy.deepcopy(stats['route']))
        #             )
        #             self._blue_team_preparation.append(SoldierStone(deepcopy(stats)))
        #             self._all_blue_team.append(SoldierStone(deepcopy(stats)))
        # ---------------保存蓝方路径json---------------
        # blue_path.save_to_json(selected_blue_path, 'path_file/selected_blue_path.json')

        paths = extend_paths(self._blue_routes, 78)  # 固定路径paths等于这个
        # 额外补充第二个参数为8,16,24,32,39,47,55,63,70的各风格实验，分别对应额外路径10%-90%
        # paths = self._blue_routes['routes'] # 全放开

        # 判断部署风格：
        if blue_style == 0:  # 使用所有的路由部署士兵，不用任何筛选条件。

            add_soldiers(units_list, paths, is_attack_move=True)

        elif (
            blue_style == 1
        ):  # 将路由按位置（左、右、上、下）分组，并随机选择一个分组来部署士兵。（意味着士兵将被集中部署在地图的某个特定区域）

            conditions = {
                "left": lambda item: item["start"][0] < 10,
                "right": lambda item: item["start"][0] > 170,
                "top": lambda item: item["start"][1] < 10,
                "bottom": lambda item: item["start"][1] > 130,
            }
            routes_group = [
                list(filter(cond, paths)) for cond in conditions.values()
            ]  # 按条件分组
            selected_routes = random.choice(routes_group)
            add_soldiers(units_list, selected_routes, is_remember=False)

        elif (
            blue_style == 2
        ):  # 对每个路由，随机选择其路径中的一个点作为士兵的起点，并仅使用这个点作为路径来部署士兵。（非常随机）

            add_soldiers(units_list, paths)
        # elif blue_style == 4:  # 简单场景环境设置
        #     simple_add_soldiers(units_list, self._blue_routes['routes'], is_remember=True)

        for x, y in self.route_records:
            self.route_record_in_map[y][
                x
            ] += 1  # 蓝方单位部署后，更新每个位置下蓝方的经过人数

    def _mapping_action(self, action):
        turret_num = self._config["num_far_turret"] + self._config["num_near_turret"]
        action = np.array(action)
        action = action.reshape(turret_num, 3)
        self.action_distribution = [
            (
                [
                    int(coord) for coord in self._red_deployment_points[act[0]].tolist()
                ],  # 将 (x, y) 坐标转换为标准 Python int
                int(((act[1]) * 60) % 360),  # 确保 angle_start 是标准 int
                int(
                    ((act[2] + 1) * 60 + ((act[1]) * 60)) % 360
                ),  # 确保 angle_end 是标准 int
            )
            for act in action
        ]

        def create_turret(i, act):
            pos = self._red_deployment_points[act[0]]
            angle_start = ((act[1]) * 60) % 360
            angle_end = ((act[2] + 1) * 60 + angle_start) % 360
            angle = [angle_start, angle_end]
            turret_type = "near" if i < self._config["num_near_turret"] else "far"
            return (f"{turret_type}_turret_{i}", pos, angle)

        return [create_turret(i, act) for i, act in enumerate(action)]

    # 里是初始化时部署红队的单位的函数
    def red_team_deployment(self, action):
        """
        units_list -> [unit_name:str, posx:int, posy:int, directions:list]
        unit_name -> '{stone_type}_{idx}'
        directions -> [direction:int]
        """

        # 创建并添加单位到红队列表
        def add_unit(unit_class, *args, **kwargs):
            stats = {  # 单位的统计信息字典stats
                "name": kwargs["name"],
                "posx": kwargs["posx"],
                "posy": kwargs["posy"],
                "attack_power": kwargs["attack"],
                "mobility": 0,
                "health": kwargs["health"],
                "attack_range": kwargs["attack_range"],
                "accuracy": kwargs["accuracy"],
                "ammunition": kwargs["ammunition"],
                "angle_start": kwargs["angle_start"],
                "angle_end": kwargs["angle_end"],
                "max_attack_per_turn": 7,
            }
            new_unit = unit_class(stats)
            # 使用上面这个单位的统计信息字典作为参数来实例化一个新的红方单位，并将这个单位添加到红队列表self._red_team中
            self._red_team.append(new_unit)
            self._all_red_team.append(new_unit)
            return new_unit

        units_list = self._mapping_action(action)
        self.firepower_coverage = np.zeros(
            (self._map.shape[1], self._map.shape[0]), dtype=float
        )  # 重置火力强度二维数组为全0
        self.total_firepower_coverage = np.zeros(
            (self._map.shape[1], self._map.shape[0]), dtype=int
        )  # 用于存取不消减的火力覆盖范围

        for unit_name, (x, y), angle in units_list:
            self.red_deploy_count[(x, y)] += 1

            # 通过检查unit_name的前缀（'near_turret'或'far_turret'）决定要部署的单位类型（近距离炮塔和远距离炮塔）
            if unit_name.startswith("near_turret"):
                new_unit = add_unit(
                    NearTurretStone,
                    name=unit_name,
                    posx=x,
                    posy=y,
                    reward=-1,
                    health=self._config["near_turret_health"],
                    attack=self._config["near_turret_attack"],
                    angle_start=angle[0],
                    angle_end=angle[1],
                    attack_range=self._config["near_turret_attack_range"],
                    accuracy=self._config["near_turret_accuracy"],
                    ammunition=self._config["near_turret_ammunition"],
                )
                #  更新火力覆盖范围
                self.update_firepower_coverage(
                    x,
                    y,
                    self._config["near_turret_attack_range"],
                    angle[0],
                    angle[1],
                    new_unit._attack_per_turn,
                )
            elif unit_name.startswith("far_turret"):
                new_unit = add_unit(
                    FarTurretStone,
                    name=unit_name,
                    posx=x,
                    posy=y,
                    reward=-1,
                    health=self._config["far_turret_health"],
                    attack=self._config["far_turret_attack"],
                    angle_start=angle[0],
                    angle_end=angle[1],
                    attack_range=self._config["far_turret_attack_range"],
                    accuracy=self._config["far_turret_accuracy"],
                    ammunition=self._config["far_turret_ammunition"],
                )
                #  更新火力覆盖范围
                self.update_firepower_coverage(
                    x,
                    y,
                    self._config["far_turret_attack_range"],
                    angle[0],
                    angle[1],
                    new_unit._attack_per_turn,
                )

        self.calculate_fire_coverage_in_area()
        self.weight_fire_coverage()

    # 更新火力强度属性
    # 更新火力强度属性
    def update_firepower_coverage(
        self, x, y, attack_range, angle_start, angle_end, attack_per_turn
    ):
        """
        更新火力覆盖范围，用于新增单位的火力影响。
        :param x: 中心点的 x 坐标
        :param y: 中心点的 y 坐标
        :param attack_range: (min_range, max_range) 最小和最大射程
        :param angle_start: 起始攻击角度
        :param angle_end: 终止攻击角度
        :param attack_per_turn: 单位的每轮攻击次数
        """
        min_range, max_range = attack_range
        full_circle = angle_start == angle_end  # 判断是否为 360° 覆盖

        # 遍历火力可能影响的范围
        for i in range(
            max(0, x - max_range), min(self._map.shape[0], x + max_range + 1)
        ):
            for j in range(
                max(0, y - max_range), min(self._map.shape[1], y + max_range + 1)
            ):
                distance = np.sqrt((x - i) ** 2 + (y - j) ** 2)  # 计算与中心的距离

                # 检查距离是否在攻击范围内
                if min_range <= distance <= max_range:
                    dx, dy = i - x, j - y
                    angle = (
                        math.degrees(math.atan2(dx, -dy)) + 360
                    ) % 360  # 将角度转换为 [0, 360) 范围

                    # 检查角度是否在范围内或是否为 360°覆盖
                    if full_circle or (
                        angle_start <= angle <= angle_end
                        or (
                            angle_start > angle_end
                            and (angle >= angle_start or angle <= angle_end)
                        )
                    ):

                        # 更新总覆盖强度
                        self.total_firepower_coverage[j, i] += 1 * attack_per_turn

                        # 根据远程和近程塔的射程范围分别更新
                        if min_range == 10:  # 远程塔
                            self.firepower_coverage[j, i] += 1
                        elif min_range == 0:  # 近程塔
                            self.firepower_coverage[j, i] += 0.01

                        # 限制精度为两位小数
                        self.firepower_coverage[j, i] = round(
                            self.firepower_coverage[j, i], 2
                        )

                        # # 调试输出
                        # print(f"Updating firepower at ({i}, {j}) -> Total Coverage: {self.total_firepower_coverage[j, i]}, Firepower: {self.firepower_coverage[j, i]}")

        # # 以下代码用于验证数字矩阵的正确性
        # import csv
        # data = self.firepower_coverage.tolist()
        # with open(rf"D:\ceshi\{time.time()}.csv", mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(data)

    def dead_update_firepower_coverage(
        self, x, y, attack_range, angle_start, angle_end, attack_per_turn
    ):
        """
        移除火力覆盖范围，用于移除单位死亡后的火力影响。
        :param x: 中心点的 x 坐标
        :param y: 中心点的 y 坐标
        :param attack_range: (min_range, max_range) 最小和最大射程
        :param angle_start: 起始攻击角度
        :param angle_end: 终止攻击角度
        :param attack_per_turn: 单位的每轮攻击次数
        """
        min_range, max_range = attack_range
        full_circle = angle_start == angle_end  # 检查是否是360°覆盖

        # 遍历火力可能影响的范围
        for i in range(
            max(0, x - max_range), min(self._map.shape[0], x + max_range + 1)
        ):
            for j in range(
                max(0, y - max_range), min(self._map.shape[1], y + max_range + 1)
            ):
                distance = np.sqrt((x - i) ** 2 + (y - j) ** 2)  # 计算与中心的距离

                # 检查距离是否在攻击范围内
                if min_range <= distance <= max_range:
                    dx, dy = i - x, j - y
                    angle = (
                        math.degrees(math.atan2(dx, -dy)) + 360
                    ) % 360  # 将角度转换为 [0, 360) 范围

                    # 检查角度是否在范围内或为360°覆盖
                    if (
                        full_circle
                        or angle_start <= angle <= angle_end
                        or (
                            angle_start > angle_end
                            and (angle >= angle_start or angle <= angle_end)
                        )
                    ):

                        # 更新总覆盖强度
                        self.total_firepower_coverage[j, i] = max(
                            0, self.total_firepower_coverage[j, i] - 1 * attack_per_turn
                        )

                        # 根据远程和近程塔的射程范围分别更新
                        if min_range != 0:  # 远程塔
                            self.firepower_coverage[j, i] = max(
                                0, self.firepower_coverage[j, i] - 1
                            )
                        else:  # 近程塔
                            self.firepower_coverage[j, i] = max(
                                0, self.firepower_coverage[j, i] - 0.01
                            )

                        # 限制精度为两位小数
                        self.firepower_coverage[j, i] = round(
                            self.firepower_coverage[j, i], 2
                        )

                        # # 调试输出
                        # if i == 100 and j == 67:
                        #     print(f"Removing firepower at ({i}, {j}) -> Total Coverage: {self.total_firepower_coverage[j, i]}, Firepower: {self.firepower_coverage[j, i]}")

    # 更新单位的状态，并根据状态的变化来更新团队的统计信息和团队列表。
    def _update_unit_states(self, unit):
        state = unit.get_states()
        if (
            state == STONE_STATE.DEAD
        ):  # 如果单位的状态是 STONE_STATE.DEAD（死亡），则进一步判断该单位属于哪个团队
            if (
                unit in self._red_team
            ):  # 如果单位在红队中，则将其从红队列表中移除，并增加红队死亡单位的统计数
                self._red_team.remove(unit)
                self._summary["red_dead"] += 1

                # 需要将红方死亡单位的攻击范围重新渲染。
                # 在红方单位死亡时，需要将其攻击范围从地图上移除。这可以通过将火力覆盖统计数组中对应位置的值减1来实现。
                self.dead_update_firepower_coverage(
                    unit.get_posx(),
                    unit.get_posy(),
                    unit._attack_range,
                    unit._angle_start,
                    unit._angle_end,
                    unit._attack_per_turn,
                )  # 圆心x,圆心y，半径，起始角度，终止角度
                # self.draw_firepower_coverage()
            elif (
                unit in self._blue_team
            ):  # 如果单位在蓝队中，则将其从蓝队列表中移除，并增加蓝队死亡单位的统计数
                self._blue_team.remove(unit)
                self._summary["blue_dead"] += 1
        if (
            state == STONE_STATE.EVACUATED
        ):  # 如果单位的状态是撤离，则增加蓝队撤离单位的统计数，因此将其从蓝队列表中移除。
            self._summary["blue_evacuated"] += 1
            self._blue_team.remove(unit)

    # 战斗模拟函数：其中包含蓝队和红队之间的交互和攻击
    def simulate_battle(self, if_saveGIF=True):
        # result summary variables
        # 在新一轮开始时初始化gif截图列表
        self.screenshots = []

        while True:
            # 每进行一轮战斗，步数（self._summary['steps']）增加1，并调用render()方法来渲染当前战斗状态
            self._summary["steps"] += 1
            self.render()

            # 直接调取Surface对象保存为图片，用作GIF保存：
            if self.render_mode == "gif":
                screenshot = self.screen.copy()
                path = "battle_gif/screenshot.png"
                if os.path.exists(path):  # 如果文件已存在，则删除它以避免错误
                    os.remove(path)
                pg.image.save(screenshot, path)  # 暂时保存截图到指定路径
                self.screenshots.append(
                    Image.open(path).copy()
                )  # 从指定路径加载图片的copy并添加到列表中

            # blue team deployment
            #   在每一轮开始时，蓝队会从准备队列（self._blue_team_preparation）中取出指定数量的单位（self._config['unit_num_per_step']）加入战斗队列（self._blue_team）
            #   如果准备队列为空，则停止添加。（蓝方刷怪的过程）
            for _ in range(self._config["unit_num_per_step"]):
                if len(self._blue_team_preparation) == 0:
                    break
                self._blue_team.append(self._blue_team_preparation.pop())

            # 攻击阶段：初始化两个字典damage_record和health_record来记录伤害和各个单位的健康值。
            # 红队单位依次攻击蓝队单位，记录造成的伤害。
            # 蓝队单位依次攻击红队单位，同样记录伤害。
            # 合并两次攻击的伤害记录到damage_record中。
            damage_record = defaultdict(int)
            health_record = {
                unit.get_name(): unit.get_health()
                for team in (self._blue_team, self._red_team)
                for unit in team
            }
            for unit in self._red_team:
                if "far" in unit.get_name():
                    (
                        unit_damage,
                        red_attack_times_per_turn,
                        red_attack_successful_per_turn,
                        no_under_attack,
                    ) = unit.attack(self._blue_team, health_record)
                    self.no_under_attack += no_under_attack
                else:
                    (
                        unit_damage,
                        red_attack_times_per_turn,
                        red_attack_successful_per_turn,
                    ) = unit.attack(self._blue_team, health_record)
                damage_record.update(
                    (k, v + damage_record[k]) for k, v in unit_damage.items()
                )
                self.red_attack_times_per_turn += red_attack_times_per_turn
                self.red_attack_successful_per_turn += red_attack_successful_per_turn
            for unit in self._blue_team:
                (
                    unit_damage,
                    blue_attack_times_per_turn,
                    blue_attack_successful_per_turn,
                ) = unit.attack(self._red_team, health_record)
                damage_record.update(
                    (k, v + damage_record[k]) for k, v in unit_damage.items()
                )
                self.blue_attack_times_per_turn += blue_attack_times_per_turn
                self.blue_attack_successful_per_turn += blue_attack_successful_per_turn
            # set damage 应用伤害
            # 遍历红队和蓝队的所有单位，根据damage_record中的记录应用伤害。
            # 更新单位状态（self._update_unit_states(unit)）。
            for unit in self._red_team + self._blue_team:
                unit_name = unit.get_name()
                if unit_name in damage_record:
                    unit.set_damage(damage_record[unit_name])
                    self._update_unit_states(unit)

            # blue team move 蓝方移动
            # 遍历红队和蓝队的所有单位，根据damage_record中的记录应用伤害。
            # 更新单位状态（self._update_unit_states(unit)）。
            for unit in self._blue_team:
                if unit.get_states() == STONE_STATE.MOBILE:
                    unit.move()  # 移动
                    self._update_unit_states(unit)

            # check game result
            # 检查战斗结果：
            # 如果战斗步数达到最大步数（self._config['max_steps']），则结束战斗，并设置结果为1（可能表示红队胜利或其他自定义结果）。
            # 如果蓝队死亡单位数达到总单位数（self._config['num_soldier']），同样结束战斗并设置结果为1。
            # 如果蓝队撤离单位数超过最小撤离单位数（self._config['minimum_unit_evacuated']），则设置结果为0（可能表示蓝队成功撤离或某种形式的胜利）。
            # check game result
            if self._summary["steps"] >= self._config["max_steps"]:
                self._summary["result"] = 1
                break
            if (
                self._summary["blue_dead"]
                == self._config["num_soldier"] + self._config["num_vehicle"]
            ):
                self._summary["result"] = 1
                break
            if len(self._blue_team) == 0:
                break
        if self._summary["blue_evacuated"] > self._config["minimum_unit_evacuated"]:
            self._summary["result"] = 0
        else:
            self._summary["result"] = 1

        if self.render_mode == "gif":
            # 完成战斗数目加一(用于GIF功能的名称动态生成)
            self.cs_time += 1

            # 假设 self.screenshots 可能包含非 PIL 图像对象
            valid_screenshots = [
                img for img in self.screenshots if isinstance(img, PIL.Image.Image)
            ]  # 将之排除，组成新列表
            if self._summary["result"] == 0:
                win_lose = "lose"
            else:
                win_lose = "win"
            # 创建GIF。
            gif_path = f"battle_gif/{win_lose}/{self.style}_{self.cs_time}.gif"  # 路径
            if os.path.exists(gif_path):  # 如果文件已存在，则删除它以避免错误
                os.remove(gif_path)
            imageio.mimsave(
                gif_path,
                [img.convert("RGB") for img in valid_screenshots],
                duration=0.2,
            )  # duration可以调整帧之间的延迟
            if os.path.exists(path):  # 根据路径删除临时图片保存。
                os.remove(path)

        return self._summary

    def save_round_data_to_json(self):
        # 收集蓝方信息
        blue_team_data = []
        for unit in self._all_blue_team:
            blue_team_data.append({})

    def save_special_enemy(self, result):
        """
        保存特定轮次的敌人，当 result 为 0（失败）时，
        将 self._all_blue_team 保存到指定文件。

        参数:
            self: 类的实例对象，包含属性 style 和 _all_blue_team。
            result: int 类型，0 表示失败，1 表示胜利。
        """
        if result == 0:  # 失败时保存敌人数据
            file_name = f"{self.style}-enemy.json"
            try:
                with open(file_name, "w", encoding="utf-8") as file:
                    json.dump(self.special_enemy, file, ensure_ascii=False, indent=4)
                print(f"敌人数据已成功保存到 {file_name}")
            except Exception as e:
                print(f"保存敌人数据时出错: {e}")
