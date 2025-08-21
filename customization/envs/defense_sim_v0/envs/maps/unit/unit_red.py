# unit_red.py

import json
import logging
import math
import random
import uuid
from typing import Dict, List, Tuple

from customization.envs.defense_sim_v0.envs.maps.unit.unit_blue import (
    ArmoredVehicle,
    InfantryEnemy,
)
from customization.envs.defense_sim_v0.envs.maps.unit.weapon import Weapon

# 创建模块级日志器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 设置日志级别

# 避免日志传播到根日志器
logger.propagate = False

# 定义常量用于 max_angles 的索引
MAX_PITCH_UP = 0
MAX_PITCH_DOWN = 1
MAX_YAW_LEFT = 2
MAX_YAW_RIGHT = 3

# 定义类映射
TURRET_CLASSES = {}


def truncate(f: float, n: int) -> float:
    """
    截断浮点数到指定的小数位数，不进行四舍五入。

    :param f: 需要截断的浮点数。
    :param n: 小数位数。
    :return: 截断后的浮点数。
    """
    try:
        multiplier = 10**n
        if f >= 0:
            truncated = math.floor(f * multiplier) / multiplier
        else:
            truncated = math.ceil(f * multiplier) / multiplier
        return truncated
    except Exception as e:
        logger.error(f"截断数值时出错: {e}，数值: {f}, 小数位数: {n}")
        raise  # 重新抛出异常以便上层捕获


class Turret:
    def __init__(
        self,
        position,
        health,
        default_orientation=(0, 0, 0),
        max_angles=None,
        weapons=None,
        los_data=None,
    ):
        """
        初始化炮塔对象
        """
        self.id = uuid.uuid4()
        self.position = position
        self.health = health

        # 默认朝向欧拉角 [roll, pitch, yaw]
        self.default_orientation = default_orientation
        self.roll = default_orientation[0] if len(default_orientation) > 0 else 0
        self.pitch = default_orientation[1] if len(default_orientation) > 1 else 0
        self.yaw = default_orientation[2] if len(default_orientation) > 2 else 0

        # 计算朝向方向的角度 (angle)
        self.angle = self.calculate_angle(self.yaw)

        # 如果未提供 max_angles，则使用默认值
        if max_angles is None:
            self.max_angles = [
                45,
                45,
                90,
                90,
            ]  # [max_pitch_up, max_pitch_down, max_yaw_left, max_yaw_right]
        else:
            if len(max_angles) != 4:
                raise ValueError(
                    "max_angles 必须包含四个元素：[max_pitch_up, max_pitch_down, max_yaw_left, max_yaw_right]"
                )
            self.max_angles = max_angles

        # 初始化武器列表
        if weapons is None:
            self.weapons = []
        else:
            self.weapons = weapons

        self.target = None
        self.time_since_last_attack = 0
        self.fire_rate = 1  # 攻击频率（单位：攻击/秒）
        self.damage = 10  # 每次攻击造成的伤害

        self.los_file_path = "./los_table.json"
        self.los_data = los_data  # 存储 LOS 数据

    def calculate_angle(self, yaw):
        """
        根据偏航角 (yaw) 计算朝向方向的角度
        """
        return math.degrees(
            math.atan2(math.sin(math.radians(yaw)), math.cos(math.radians(yaw)))
        )

    def reset_orientation(self):
        """
        重置炮塔的朝向到默认欧拉角
        """
        self.pitch, self.yaw = self.default_orientation[1], self.default_orientation[2]
        self.angle = self.calculate_angle(self.yaw)
        logger.info(
            f"炮塔 {self.id} 的朝向已重置到默认角度：俯仰 {self.pitch}°, 偏航 {self.yaw}°。"
        )

    def take_damage(self, damage):
        """
        承受伤害，减少生命值
        """
        self.health -= damage
        if self.health <= 0:
            self.health = 0
            logger.info(f"炮塔 {self.id} 被摧毁！")

    def is_visible(self, enemy_position: List[float]) -> bool:
        """
        根据预加载的通视表判断给定的蓝方单位是否可见。

        :param enemy_position: 蓝方单位的位置 [x, y, z]。
        :return: 如果有通视关系，则返回 True，否则返回 False。
        """
        try:
            # 交换 self.position 的第二位和第三位
            swapped_position = (self.position[0], self.position[1], self.position[2])

            # 对交换后的第一位和第三位进行截断到小数点后10位
            truncated_swapped_position = (
                round(swapped_position[0], 10),
                swapped_position[1],  # 第二位不截断
                round(swapped_position[2], 10),
            )

            # 生成键
            key = (truncated_swapped_position, tuple(enemy_position))
            visible = self.los_data.get(key, False)
            return visible
        except Exception as e:
            logger.error(f"判断可视性时出错: {e}")
            return False

    def cal_sight(
        self,
        enemy_pos: Tuple[float, float, float],
        deflection_angle: float,
        range_limits: List[float],
        visible: bool = True,
    ) -> bool:
        """
        计算防御塔是否可以瞄准并看到目标。

        参数：
            enemy_pos (Tuple[float, float, float]): 目标坐标 (x, y, z)。
            deflection_angle (float): 塔的正方向旋转角度，单位为度。
            range_limits (List[float]): 射界范围 [left_range, right_range, up_range, down_range]，单位为度。
                - left_range: 左侧最大旋转角度（正值或负值）
                - right_range: 右侧最大旋转角度（正值或负值）
                - up_range: 向上仰角的最大角度（正值或负值）
                - down_range: 向下俯角的最大角度（正值或负值）
            visible (bool): 目标是否可见。

        返回：
            bool: 能否瞄准目标。
        """

        def calculate_angle(dx: float, dz: float) -> float:
            """计算水平方向的夹角，返回[-180, 180]的角度"""
            return math.degrees(math.atan2(dz, dx))

        def calculate_elevation(dy: float, distance: float) -> float:
            """计算垂直方向的夹角，返回[-90, 90]的角度"""
            return math.degrees(math.atan2(dy, distance))

        def normalize_angle(angle: float) -> float:
            """将角度规范化到[-180, 180]范围内"""
            return (angle + 180) % 360 - 180

        def is_within_horizontal_range(angle: float, left: float, right: float) -> bool:
            """
            判断水平角度是否在[left, right]范围内，考虑角度跨越-180/180边界的情况。

            参数：
                angle (float): 要检查的角度，已规范化到[-180, 180]。
                left (float): 左侧边界角度。
                right (float): 右侧边界角度。

            返回：
                bool: 如果angle在范围内则返回True，否则返回False。
            """
            # 如果左边界小于右边界，直接比较
            if left <= right:
                return left <= angle <= right
            else:
                # 范围跨越了-180/180边界
                return angle >= left or angle <= right

        def is_within_vertical_range(angle: float, down: float, up: float) -> bool:
            """
            判断垂直角度是否在[down, up]范围内。

            参数：
                angle (float): 要检查的角度。
                down (float): 向下俯角的最小角度。
                up (float): 向上仰角的最大角度。

            返回：
                bool: 如果angle在范围内则返回True，否则返回False。
            """
            # 确保down <= up
            return down <= angle <= up

        if not visible:
            return False

        try:
            # 计算防御塔与目标的连线差值
            dx = enemy_pos[0] - self.position[0]
            dy = enemy_pos[1] - self.position[1]  # 假设 y 轴为高度
            dz = enemy_pos[2] - self.position[2]

            # 计算水平夹角并调整偏转角度
            horizontal_angle = calculate_angle(dx, dz) - deflection_angle
            # 规范化到[-180, 180]
            horizontal_angle = normalize_angle(horizontal_angle)

            # 计算与水平面的距离
            distance = math.sqrt(dx**2 + dz**2)
            # 计算垂直夹角
            vertical_angle = calculate_elevation(dy, distance)

            # 解构范围限制
            left_range, right_range, up_range, down_range = range_limits

            # 检查是否在可瞄准范围内
            can_aim_horizontal = is_within_horizontal_range(
                horizontal_angle, left_range, right_range
            )
            can_aim_vertical = is_within_vertical_range(
                vertical_angle, down_range, up_range
            )
            can_aim = can_aim_horizontal and can_aim_vertical

            # logger.info(
            #     f"塔坐标: {self.position}, 类型: {self.type}, 目标坐标: {enemy_pos}, "
            #     f"正方向旋转角度: {deflection_angle}, 水平夹角: {horizontal_angle:.2f}, "
            #     f"垂直夹角: {vertical_angle:.2f}, 能否瞄准: {can_aim}"
            # )
            # logger.info(
            #     f"塔坐标: {self.position}, 类型: {""}, 目标坐标: {enemy_pos}, "
            #     f"正方向旋转角度: {deflection_angle}, 水平夹角: {horizontal_angle:.2f}, "
            #     f"垂直夹角: {vertical_angle:.2f}, 能否瞄准: {can_aim}"
            # )

            return can_aim

        except Exception as e:
            logger.error(f"计算瞄准失败: {e}")
            return False

    def calculate_distance(
        self, position1: List[float], position2: List[float]
    ) -> float:
        """
        计算两点之间的三维欧几里得距离。

        :param position1: 第一个点的坐标 [x, y, z]。
        :param position2: 第二个点的坐标 [x, y, z]。
        :return: 两点之间的距离。
        """
        dx = position2[0] - position1[0]
        dy = position2[1] - position1[1]
        dz = position2[2] - position1[2]
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def scan_and_attack(self, blue_units: List[Dict], delta_time: float):
        """
        扫描蓝方单位并进行攻击。

        :param blue_units: 蓝方单位列表，每个单位是一个字典，包含id和position。
        :param delta_time: 时间步长（秒）。
        """
        # 如果当前没有目标，开始扫描
        if not self.target or not self.target.is_alive:
            self.target = self.scan_for_targets(blue_units)

        # 如果找到目标并且目标存活，进行攻击
        if self.target and self.target.is_alive:
            self.attack_target(delta_time)
        else:
            # 如果目标死亡或未找到，清空目标
            self.target = None

    def scan_for_targets(self, blue_units: List[Dict]) -> Dict:
        """
        扫描蓝方单位，寻找最近的目标。

        :param blue_units: 蓝方单位列表，每个单位是一个字典，包含id和position。
        :return: 最近的目标单位字典，或者 None 如果没有找到。
        """
        closest_target = None
        closest_distance = float("inf")

        for unit in blue_units:
            if not unit.is_alive:
                continue

            unit_position = unit.current_position

            # 检查通视表中是否有通视
            if not self.is_visible(unit_position):
                logger.info(f"蓝方单位 {unit.id} 与炮塔无通视关系，跳过。")
                continue
            # print(f"有通视关系。")
            # 判断目标是否在射界范围内
            in_sight = self.cal_sight(
                enemy_pos=unit_position,
                deflection_angle=self.yaw,  # 当前炮塔偏航角
                range_limits=[
                    self.max_angles[MAX_YAW_LEFT],
                    self.max_angles[MAX_YAW_RIGHT],  # 左右射界
                    self.max_angles[MAX_PITCH_UP],
                    self.max_angles[MAX_PITCH_DOWN],  # 上下射界
                ],
                visible=True,  # 假设通视成功
            )

            if not in_sight:
                logger.info(f"蓝方单位 {unit.id} 不在射界范围内，跳过。")
                continue

            # 计算单位与炮塔的距离
            distance = self.calculate_distance(self.position, unit_position)

            # 检查是否是更近的目标
            if distance < closest_distance:
                closest_target = unit
                closest_distance = distance

        if closest_target:
            logger.info(
                f"炮塔锁定目标: {closest_target.id}，距离: {closest_distance:.2f} 米"
            )
        else:
            logger.info("未找到符合条件的目标。")
        return closest_target

    def attack_target(self, delta_time: float):
        """
        对当前目标进行攻击。

        :param delta_time: 时间步长（秒）。
        """
        # 计算攻击间隔时间
        self.time_since_last_attack += delta_time
        attack_interval = 1 / self.fire_rate

        if self.time_since_last_attack >= attack_interval:
            # 确定目标类型并设置命中概率和伤害值
            if isinstance(self.target, InfantryEnemy):
                hit_chance = 0.5  # 20% 命中概率
                damage = 50
                target_type = "步兵"
            elif isinstance(self.target, ArmoredVehicle):
                hit_chance = 0.6  # 90% 命中概率
                damage = 100
                target_type = "装甲车"
            else:
                # 如果目标类型未知，使用默认值
                hit_chance = 0.5
                damage = self.damage
                target_type = "未知类型"

            # 生成一个 0 到 1 之间的随机数
            hit_roll = random.random()

            if hit_roll <= hit_chance:
                # 命中
                self.target.take_damage(damage)
                logger.info(
                    f"炮塔攻击目标: {self.target.id} ({target_type})，造成伤害: {damage}，目标剩余HP: {self.target.health:.2f}"
                )
            else:
                # 未命中
                logger.info(f"炮塔攻击目标: {self.target.id} ({target_type})，未命中。")

            # 重置攻击计时
            self.time_since_last_attack = 0

        # 如果目标被击毁，解除锁定
        if self.target.health <= 0:
            logger.info(f"目标 {self.target.id} ({target_type}) 被击毁！")
            self.target.is_alive = False
            self.target = None

    def is_in_range(self, target_position, weapon):
        """
        判断目标是否在指定武器的攻击范围内
        """
        distance = math.sqrt(
            (target_position[0] - self.position[0]) ** 2
            + (target_position[1] - self.position[1]) ** 2
            + (target_position[2] - self.position[2]) ** 2
        )
        return distance <= weapon.attack_range

    def fire(self, target, delta_time):
        """
        攻击目标，所有准备好的武器都会尝试攻击
        """
        if not self.weapons:
            logger.warning(f"炮塔 {self.id} 没有装备任何武器！")
            return

        for weapon in self.weapons:
            weapon.update_cooldown(delta_time)
            if weapon.is_ready_to_fire() and self.is_in_range(target.position, weapon):
                weapon.fire(target)

    def add_weapon(self, weapon):
        """
        添加一个武器到炮塔
        """
        self.weapons.append(weapon)
        logger.info(f"炮塔 {self.id} 添加了武器 {weapon.weapon_type}。")

    def remove_weapon(self, weapon_type):
        """
        移除指定类型的武器
        """
        initial_count = len(self.weapons)
        self.weapons = [w for w in self.weapons if w.weapon_type != weapon_type]
        if len(self.weapons) < initial_count:
            logger.info(f"炮塔 {self.id} 移除了武器类型 {weapon_type}。")
        else:
            logger.warning(f"炮塔 {self.id} 未装备武器类型 {weapon_type}。")

    def to_dict(self):
        return {
            "id": str(self.id),
            "type": self.__class__.__name__,  # 添加 "type" 字段
            "position": self.position,
            "health": self.health,
            "orientation": [self.roll, self.pitch, self.yaw],
            # "weapons": [weapon.to_dict() for weapon in self.weapons],
            "is_alive": not self.health == 0,
        }

    def __repr__(self):
        """
        返回炮塔的当前状态
        """
        weapons_str = ", ".join([repr(weapon) for weapon in self.weapons])
        return (
            f"炮塔类型 {self.__class__.__name__} - ID: {self.id}, 位置 {self.position}, 生命值 {self.health}, "
            f"武器列表=[{weapons_str}], "
            f"欧拉角 [r={self.roll}, p={self.pitch}, y={self.yaw}], "
            f"朝向方向 {self.angle}°, "
            f"最大旋转角度 {self.max_angles}"
        )

    @classmethod
    def load_turret_from_json(
        cls, json_file, position_and_angle, turret_type, los_data
    ):
        """
        从 JSON 文件和位置角度加载单个炮塔属性并创建炮塔对象
        输入：
            - json_file: JSON 文件的路径
            - position_and_angle: 包含 [x, y, z, angle] 的单个位置和角度
            - turret_type: 要加载的炮塔类型 (如 "bazhua" 或 "bigua")
            - los_data: 通视表数据字典
        输出：
            - 一个炮塔对象（如果成功加载）；否则返回 None
        """
        try:
            # 读取 JSON 文件
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)
                turrets_data = data.get("turrets", [])

                # 查找对应的炮塔类型数据
                turret_data = next(
                    (
                        t
                        for t in turrets_data
                        if t.get("type").lower() == turret_type.lower()
                    ),
                    None,
                )
                if not turret_data:
                    logger.warning(
                        f"JSON 文件中未找到炮塔类型 '{turret_type}' 的配置，跳过加载。"
                    )
                    return None

                # 验证位置和角度数据
                if (
                    not isinstance(position_and_angle, (list, tuple))
                    or len(position_and_angle) != 4
                ):
                    logger.warning(
                        f"位置角度数据无效：{position_and_angle}，跳过加载。"
                    )
                    return None

                x, y, z, angle = position_and_angle
                position = (x, y, z)
                default_orientation = (0, 0, angle)  # [roll, pitch, yaw]

                # 动态获取对应的炮塔子类
                TurretClass = TURRET_CLASSES.get(turret_type.lower())
                if not TurretClass:
                    logger.warning(f"未知的炮塔类型 '{turret_type}'，跳过加载。")
                    return None

                # 解析武器列表
                weapons_data = turret_data.get("weapons", [])
                weapons = []
                for weapon_data in weapons_data:
                    if "weapon_type" not in weapon_data:
                        logger.warning(
                            f"炮塔类型 '{turret_type}' 的武器数据不完整：{weapon_data}"
                        )
                        continue
                    # 验证武器数据的完整性
                    required_fields = [
                        "attack_frequency",
                        "attack_range",
                        "damage",
                        "ammo",
                    ]
                    if not all(field in weapon_data for field in required_fields):
                        logger.warning(f"武器数据缺少必要字段：{weapon_data}")
                        continue
                    weapon = Weapon(
                        weapon_type=weapon_data.get("weapon_type"),
                        attack_frequency=weapon_data.get("attack_frequency"),
                        attack_range=weapon_data.get("attack_range"),
                        damage=weapon_data.get("damage"),
                        ammo=weapon_data.get("ammo"),
                    )
                    weapons.append(weapon)

                # 创建炮塔对象，并传递 LOS 数据
                turret = TurretClass(
                    position=position,
                    health=turret_data.get("health", 100),
                    default_orientation=default_orientation,
                    max_angles=turret_data.get(
                        "max_angles", TurretClass.default_max_angles
                    ),
                    weapons=weapons,
                    los_data=los_data,  # 传递 LOS 数据
                )
                logger.info(f"成功加载炮塔类型 '{turret_type}'，ID: {turret.id}")
                return turret

        except FileNotFoundError:
            logger.error(f"错误：找不到文件 {json_file}。")
        except json.JSONDecodeError as e:
            logger.error(f"错误：文件 {json_file} 不是有效的 JSON 格式。错误详情: {e}")
        except KeyError as e:
            logger.error(f"错误：缺少必要的键 {e}。")
        except Exception as e:
            logger.error(f"加载炮塔时发生未预料的错误：{e}")

        return None


class BazhuaTurret(Turret):
    default_max_angles = [30, 30, 90, 90]

    def __init__(
        self,
        position,
        health,
        default_orientation=(0, 0, 0),
        max_angles=None,
        weapons=None,
        los_data=None,
    ):
        """
        初始化巴抓炮塔
        """
        if max_angles is None:
            max_angles = self.default_max_angles  # 巴抓炮塔的默认最大旋转角度
        super().__init__(
            position=position,
            health=health,
            default_orientation=default_orientation,
            max_angles=max_angles,
            weapons=weapons,
            los_data=los_data,  # 传递 LOS 数据
        )
        logger.info(f"已创建巴抓炮塔，ID: {self.id}")


# 将子类添加到类映射中
TURRET_CLASSES["bazhua"] = BazhuaTurret


class BiguaTurret(Turret):
    default_max_angles = [60, 60, 120, 120]

    def __init__(
        self,
        position,
        health,
        default_orientation=(0, 0, 0),
        max_angles=None,
        weapons=None,
        los_data=None,
    ):
        """
        初始化碧瓜炮塔
        """
        if max_angles is None:
            max_angles = self.default_max_angles  # 碧瓜炮塔的默认最大旋转角度
        super().__init__(
            position=position,
            health=health,
            default_orientation=default_orientation,
            max_angles=max_angles,
            weapons=weapons,
            los_data=los_data,  # 传递 LOS 数据
        )
        logger.info(f"已创建碧瓜炮塔，ID: {self.id}")


# 将子类添加到类映射中
TURRET_CLASSES["bigua"] = BiguaTurret


# 打击方法的测试程序


# import time

# class BlueUnit:
#     def __init__(self, unit_id, position, health):
#         """
#         初始化蓝方单位
#         """
#         self.id = unit_id
#         self.current_position = position
#         self.health = health
#         self.is_alive = True

#     def take_damage(self, damage):
#         """
#         受到伤害并更新生命状态
#         """
#         self.health -= damage
#         if self.health <= 0:
#             self.health = 0
#             self.is_alive = False


# def test_scan_and_attack():
#     # 初始化蓝方单位
#     blue_units = [
#         BlueUnit(unit_id=1, position=[100, 0, 100], health=50),  # 单位1，位置(100, 0, 100)，HP 50
#         BlueUnit(unit_id=2, position=[200, 0, 200], health=100), # 单位2，位置(200, 0, 200)，HP 100
#         BlueUnit(unit_id=3, position=[50, 0, 50], health=30)     # 单位3，位置(50, 0, 50)，HP 30
#     ]

#     # 初始化炮塔
#     turret = BazhuaTurret(
#         position=[0, 0, 0],               # 炮塔位置
#         health=100,                       # 炮塔生命值
#         default_orientation=(0, 0, 0),    # 默认朝向
#         max_angles=[45, 45, 90, 90],      # 最大角度
#         weapons=None                      # 无武器，使用默认攻击参数
#     )

#     # 模拟 LOS 通视表
#     turret.los_file_path = "./test_los_table.json"
#     with open(turret.los_file_path, "w") as los_file:
#         los_entries = [
#             {"turret_pos": [0, 0, 0], "enemy_pos": [100, 0, 100], "visible": True},
#             {"turret_pos": [0, 0, 0], "enemy_pos": [200, 0, 200], "visible": False},
#             {"turret_pos": [0, 0, 0], "enemy_pos": [50, 0, 50], "visible": True},
#         ]
#         for entry in los_entries:
#             los_file.write(json.dumps(entry) + "\n")

#     # 模拟时间步长
#     delta_time = 1.0  # 每帧1秒

#     print("=== 测试开始 ===")
#     for _ in range(5):  # 模拟5帧
#         turret.scan_and_attack(blue_units, delta_time)
#         time.sleep(1)  # 模拟每帧的时间间隔

#         # 输出蓝方单位的状态
#         for unit in blue_units:
#             status = "存活" if unit.is_alive else "死亡"
#             print(f"蓝方单位 {unit.id}：HP={unit.health}, 状态={status}")

#         # 如果所有目标都被摧毁，结束测试
#         if all(not unit.is_alive for unit in blue_units):
#             print("=== 所有目标已被摧毁，测试结束 ===")
#             break

#     print("=== 测试结束 ===")

# # 执行测试
# test_scan_and_attack()
