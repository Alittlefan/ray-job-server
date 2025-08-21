import json
import logging
import math
import random
import uuid
from typing import List, Optional, Tuple

# 创建模块级日志器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 设置日志级别

# 避免日志传播到根日志器
logger.propagate = False

# 定义类映射（如果有多种敌人类型，可以在此添加）
ENEMY_CLASSES = {}


class Enemy:
    """
    基础敌人类，所有敌人类型的基类
    """

    def __init__(
        self,
        position: Tuple[float, float, float],
        default_orientation: Tuple[float, float, float],
        trajectory: List[Tuple[float, float, float]],
        speed: List[float],
        health: float,
        attack_range: float,
        attack_damage: float,
        attack_cooldown: float,
        orientation_angle: float,
    ):
        """
        初始化基础敌人对象
        :param position: 初始位置，[x, y, z]
        :param default_orientation: 默认朝向欧拉角 [roll, pitch, yaw]
        :param trajectory: 轨迹数据，列表 [(x1, y1, z1), (x2, y2, z2), ...]
        :param speed: 速度列表（单位/秒）
        :param health: 血量
        :param attack_range: 攻击范围（单位距离）
        :param attack_damage: 攻击力
        :param attack_cooldown: 攻击冷却时间（秒）
        :param orientation_angle: 当前偏航角（yaw）
        """
        self.id = uuid.uuid4()
        self.position = position  # 初始位置
        self.default_orientation = default_orientation  # 默认朝向 [roll, pitch, yaw]
        self.roll, self.pitch, self.yaw = default_orientation

        self.orientation_angle = orientation_angle  # 当前偏航角

        self.path = trajectory  # 轨迹点列表，已是元组形式
        self.speed_options = speed  # 速度列表
        self.current_speed = (
            random.choice(self.speed_options) if self.speed_options else 0.0
        )  # 当前速度
        self.health = health
        self.attack_range = attack_range
        self.attack_damage = attack_damage
        self.attack_cooldown = attack_cooldown

        self.current_position = self.path[0] if self.path else tuple(position)
        self.total_distance = 0.0
        self.is_alive = True
        self.cooldown_timer = 0.0

    def move(self, delta_time: float):
        """
        根据时间更新敌人的位置
        :param delta_time: 刷新时间间隔（秒）
        """
        if not self.is_alive:
            return

        # 计算新的累计距离
        self.total_distance += self.current_speed * delta_time
        index = int(self.total_distance)

        if index >= len(self.path):
            self.current_position = self.path[-1]
            # logging.info(f"敌人 {self.id} 已到达终点。")
        else:
            self.current_position = self.path[index]
            logger.info(f"敌人 {self.id} 移动到位置 {self.current_position}。")

    def reset_distance(self):
        """
        重置累计移动距离
        """
        self.total_distance = 0.0
        # logging.info(f"敌人 {self.id} 的累计移动距离已重置。")

    def take_damage(self, damage: float):
        """
        承受伤害，减少血量
        :param damage: 伤害值
        """
        if not self.is_alive:
            return

        self.health -= damage
        # logging.info(f"敌人 {self.id} 承受了 {damage} 点伤害，剩余血量 {self.health}。")
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
            # logging.info(f"敌人 {self.id} 被击杀！")

    def fire(self, targets, delta_time: float):
        """
        敌人开火攻击范围内的目标
        :param targets: 目标列表，每个目标应具有 `current_position` 和 `take_damage(damage)` 方法
        :param delta_time: 刷新时间间隔（秒）
        """
        if not self.is_alive:
            return

        self.cooldown_timer -= delta_time
        if self.cooldown_timer > 0:
            return

        for target in targets:
            distance = math.sqrt(
                (target.current_position[0] - self.current_position[0]) ** 2
                + (target.current_position[1] - self.current_position[1]) ** 2
                + (target.current_position[2] - self.current_position[2]) ** 2
            )
            if distance <= self.attack_range:
                target.take_damage(self.attack_damage)
                logging.info(
                    f"敌人 {self.id} 攻击了目标 {target.id}，造成 {self.attack_damage} 点伤害。"
                )
                self.cooldown_timer = self.attack_cooldown
                break

    def reset_orientation(self):
        """
        重置敌人的朝向到默认欧拉角
        """
        self.roll, self.pitch, self.yaw = self.default_orientation
        logging.info(
            f"敌人 {self.id} 的朝向已重置到默认角度：滚转 {self.roll}°, 俯仰 {self.pitch}°, 偏航 {self.yaw}°。"
        )

    def to_dict(self):
        return {
            "id": str(self.id),
            "type": self.__class__.__name__,
            "position": self.current_position,
            "health": self.health,
            "is_alive": self.is_alive,
        }

    def __repr__(self):
        """
        返回敌人的当前状态
        """
        return (
            f"敌人 {self.id}: 位置 {self.current_position}, "
            f"血量 {self.health}, {'存活' if self.is_alive else '死亡'}, "
            f"累计移动距离 {self.total_distance:.2f}, 朝向 [r={self.roll}°, p={self.pitch}°, y={self.yaw}°], "
            f"当前速度 {self.current_speed}°"
        )

    @classmethod
    def load_enemies_from_json(
        cls,
        coord: Tuple[float, float, float],
        main_json_file: Optional[str] = None,
        trajectory_json_file: Optional[str] = None,
        unit_type: str = "infantry",
    ) -> List["Enemy"]:
        """
        从主 JSON 文件和轨迹 JSON 文件加载指定类型的敌人对象。
        选择轨迹文件中所有轨迹的第一个坐标点与 coord 计算距离，选择距离最近的轨迹作为单位的轨迹。
        :param main_json_file: 主 JSON 文件的路径，包含敌人基本信息
        :param trajectory_json_file: 轨迹 JSON 文件的路径，包含多个轨迹数据
        :param coord: 参考坐标点 [x, y, z]，用于选择最接近的轨迹
        :param unit_type: 要实例化的敌人类型，必须为 'infantry' 或 'armored_vehicle'
        :return: 创建的敌人对象列表
        """
        enemies = []
        unit_type_lower = unit_type.lower()
        valid_unit_types = ENEMY_CLASSES.keys()

        if unit_type_lower not in valid_unit_types:
            logging.error(
                f"无效的 unit_type '{unit_type}'. 仅支持 {list(valid_unit_types)}。"
            )
            return enemies

        try:
            # 读取轨迹数据
            if trajectory_json_file is not None:
                loaded_trajectories = []
                with open(trajectory_json_file, "r", encoding="utf-8") as traj_file:
                    for line_number, line in enumerate(traj_file, 1):
                        line = line.strip()
                        if not line:
                            continue  # 跳过空行
                        try:
                            traj_data = json.loads(line)
                            traj_id = traj_data.get("id")
                            traj_points = traj_data.get("data", [])
                            if not traj_id or not isinstance(traj_points, list):
                                logging.warning(
                                    f"轨迹文件第 {line_number} 行数据格式不正确，跳过。"
                                )
                                continue
                            # 将每个点转换为元组 (x, y, z)
                            trajectory = []
                            for point in traj_points:
                                x = point.get("x")
                                y = point.get("y")
                                z = point.get("z")
                                if x is None or y is None or z is None:
                                    logging.warning(
                                        f"轨迹 '{traj_id}' 中的一个点缺少坐标，跳过该点。"
                                    )
                                    continue
                                trajectory.append((x, y, z))
                            if trajectory:
                                loaded_trajectories.append(trajectory)
                            else:
                                logging.warning(
                                    f"轨迹 '{traj_id}' 没有有效的轨迹点，跳过。"
                                )
                        except json.JSONDecodeError as e:
                            logging.warning(
                                f"轨迹文件第 {line_number} 行不是有效的 JSON 格式，跳过。错误详情: {e}"
                            )

                if not loaded_trajectories:
                    logging.error("没有可用的轨迹数据来创建敌人。")
                    return enemies
            else:
                logging.error("未提供轨迹文件路径。")
                return enemies

            # 选择与 coord 最近的轨迹
            closest_trajectory = None
            min_distance = float("inf")
            for trajectory in loaded_trajectories:
                first_point = trajectory[0]
                distance = math.sqrt(
                    (first_point[0] - coord[0]) ** 2
                    + (first_point[1] - coord[1]) ** 2
                    + (first_point[2] - coord[2]) ** 2
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_trajectory = trajectory

            if closest_trajectory is None:
                logging.error("未能找到与 coord 最近的轨迹。")
                return enemies

            # 获取第一个坐标点作为初始位置
            initial_position = closest_trajectory[0]

            # 读取主 JSON 文件
            if main_json_file is not None:
                with open(main_json_file, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    enemies_data = data.get("enemies", [])

                    # 计数已加载的敌人
                    loaded_count = 0

                    for index, enemy_data in enumerate(enemies_data):
                        current_enemy_type = enemy_data.get("type", "").lower()
                        if current_enemy_type != unit_type_lower:
                            continue  # 跳过不匹配的类型

                        EnemyClass = ENEMY_CLASSES.get(current_enemy_type)
                        if EnemyClass is None:
                            logging.warning(
                                f"未知的敌人类型 '{current_enemy_type}'，跳过该敌人。"
                            )
                            continue

                        # 使用选择的轨迹
                        trajectory = closest_trajectory
                        if not trajectory:
                            logging.warning(f"选择的轨迹为空，跳过敌人 {index + 1}。")
                            continue

                        # 解析敌人属性
                        speed = enemy_data.get(
                            "speed", [1.5, 4.5]
                        )  # 默认为步兵速度数组
                        if not isinstance(speed, list) or not all(
                            isinstance(s, (int, float)) for s in speed
                        ):
                            logging.warning(
                                f"敌人速度数据无效：{speed}，使用默认速度。"
                            )
                            speed = (
                                [1.5, 4.5]
                                if current_enemy_type == "infantry"
                                else [1.5]
                            )
                        health = enemy_data.get("health", 100)
                        attack_range = enemy_data.get("attack_range", 5.0)
                        attack_damage = enemy_data.get("attack_damage", 10)
                        attack_cooldown = enemy_data.get("attack_cooldown", 2.0)

                        # 特殊属性
                        splash_radius = enemy_data.get("splash_radius", 0.0)

                        # 从 JSON 中获取默认朝向
                        default_orientation = enemy_data.get(
                            "default_orientation", [0.0, 0.0, 0.0]
                        )
                        if (
                            not isinstance(default_orientation, list)
                            or len(default_orientation) != 3
                        ):
                            logging.warning(
                                f"敌人默认朝向数据无效：{default_orientation}，使用 [0.0, 0.0, 0.0]。"
                            )
                            default_orientation = [0.0, 0.0, 0.0]

                        # 创建敌人对象
                        if EnemyClass == ArmoredVehicle:
                            enemy = EnemyClass(
                                position=tuple(initial_position),
                                default_orientation=tuple(default_orientation),
                                trajectory=trajectory,
                                speed=speed,
                                health=health,
                                attack_range=attack_range,
                                attack_damage=attack_damage,
                                attack_cooldown=attack_cooldown,
                                splash_radius=splash_radius,
                                orientation_angle=0.0,  # 根据 positions_and_angles 中的角度
                            )
                        else:
                            enemy = EnemyClass(
                                position=tuple(initial_position),
                                default_orientation=tuple(default_orientation),
                                trajectory=trajectory,
                                speed=speed,
                                health=health,
                                attack_range=attack_range,
                                attack_damage=attack_damage,
                                attack_cooldown=attack_cooldown,
                                orientation_angle=0.0,  # 根据 positions_and_angles 中的角度
                            )
                        enemies.append(enemy)
                        loaded_count += 1
                        # logging.info(f"成功创建敌人类型 '{unit_type}', ID: {enemy.id}")

                    # if loaded_count == 0:
                    #     logging.warning(f"主 JSON 文件中没有类型为 '{unit_type}' 的敌人。")
                    # else:
                    # logging.info(f"成功加载了 {loaded_count} 个类型为 '{unit_type}' 的敌人。")
            else:
                logging.error("未提供 main_json_file 路径，无法加载敌人。")

        except FileNotFoundError as e:
            logging.error(f"错误：找不到文件。详情: {e}")
        except json.JSONDecodeError as e:
            logging.error(f"错误：文件不是有效的 JSON 格式。错误详情: {e}")
        except KeyError as e:
            logging.error(f"错误：缺少必要的键 {e}。")
        except Exception as e:
            logging.error(f"加载敌人时发生未预料的错误：{e}")
        return enemies


class InfantryEnemy(Enemy):
    def __init__(
        self,
        position: Tuple[float, float, float],
        default_orientation: Tuple[float, float, float],
        trajectory: List[Tuple[float, float, float]],
        speed: List[float],
        health: float,
        attack_range: float,
        attack_damage: float,
        attack_cooldown: float,
        orientation_angle: float,
    ):
        """
        初始化步兵敌人对象
        :param position: 初始位置，[x, y, z]
        :param default_orientation: 默认朝向欧拉角 [roll, pitch, yaw]
        :param trajectory: 轨迹数据，列表 [(x1, y1, z1), (x2, y2, z2), ...]
        :param speed: 速度列表（1.5 或 4.5）
        :param health: 血量
        :param attack_range: 攻击范围（单位距离）
        :param attack_damage: 攻击力
        :param attack_cooldown: 攻击冷却时间（秒）
        :param orientation_angle: 当前偏航角（yaw）
        """
        # 验证速度列表是否包含 1.5 或 4.5
        if not any(s in [1.5, 4.5] for s in speed):
            raise ValueError("步兵的速度列表必须包含 1.5 或 4.5")
        super().__init__(
            position,
            default_orientation,
            trajectory,
            speed,
            health,
            attack_range,
            attack_damage,
            attack_cooldown,
            orientation_angle,
        )
        logging.info(f"创建步兵敌人，ID: {self.id}")

    def __repr__(self):
        """
        返回步兵敌人的当前状态
        """
        return (
            f"步兵 {self.id}: 位置 {self.current_position}, "
            f"血量 {self.health}, {'存活' if self.is_alive else '死亡'}, "
            f"累计移动距离 {self.total_distance:.2f}, 朝向 [r={self.roll}°, p={self.pitch}°, y={self.yaw}°], "
            f"当前速度 {self.current_speed}°"
        )


class ArmoredVehicle(Enemy):
    def __init__(
        self,
        position: Tuple[float, float, float],
        default_orientation: Tuple[float, float, float],
        trajectory: List[Tuple[float, float, float]],
        speed: List[float],
        health: float,
        attack_range: float,
        attack_damage: float,
        attack_cooldown: float,
        splash_radius: float,
        orientation_angle: float,
    ):
        """
        初始化装甲车对象
        :param position: 初始位置，[x, y, z]
        :param default_orientation: 默认朝向欧拉角 [roll, pitch, yaw]
        :param trajectory: 轨迹数据，列表 [(x1, y1, z1), (x2, y2, z2), ...]
        :param speed: 速度列表（4.5）
        :param health: 血量
        :param attack_range: 攻击范围（单位距离）
        :param attack_damage: 攻击力
        :param attack_cooldown: 攻击冷却时间（秒）
        :param splash_radius: 范围攻击的半径
        :param orientation_angle: 当前偏航角（yaw）
        """
        # 验证速度列表是否仅包含 4.5
        if speed != [1.5]:
            raise ValueError("装甲车的速度列表必须仅包含 4.5")
        super().__init__(
            position,
            default_orientation,
            trajectory,
            speed,
            health,
            attack_range,
            attack_damage,
            attack_cooldown,
            orientation_angle,
        )
        self.splash_radius = splash_radius

    def fire(self, targets, delta_time: float):
        """
        装甲车的范围攻击逻辑
        :param targets: 目标列表，每个目标应具有 `current_position` 和 `take_damage(damage)` 方法
        :param delta_time: 刷新时间间隔（秒）
        """
        if not self.is_alive:
            return

        self.cooldown_timer -= delta_time
        if self.cooldown_timer > 0:
            return

        for target in targets:
            distance = math.sqrt(
                (target.current_position[0] - self.current_position[0]) ** 2
                + (target.current_position[1] - self.current_position[1]) ** 2
                + (target.current_position[2] - self.current_position[2]) ** 2
            )
            if distance <= self.attack_range:
                logging.info(f"装甲车 {self.id} 对目标 {target.id} 进行范围攻击！")
                for other_target in targets:
                    splash_distance = math.sqrt(
                        (other_target.current_position[0] - target.current_position[0])
                        ** 2
                        + (
                            other_target.current_position[1]
                            - target.current_position[1]
                        )
                        ** 2
                        + (
                            other_target.current_position[2]
                            - target.current_position[2]
                        )
                        ** 2
                    )
                    if splash_distance <= self.splash_radius:
                        other_target.take_damage(self.attack_damage)
                        logging.info(
                            f"目标 {other_target.id} 受到了溅射伤害 {self.attack_damage} 点！"
                        )
                self.cooldown_timer = self.attack_cooldown
                break

    def __repr__(self):
        """
        返回装甲车的当前状态
        """
        return (
            f"装甲车 {self.id}: 位置 {self.current_position}, 血量 {self.health}, "
            f"{'存活' if self.is_alive else '死亡'}, "
            f"累计移动距离 {self.total_distance:.2f}, 朝向 [r={self.roll}°, p={self.pitch}°, y={self.yaw}°], "
            f"当前速度 {self.current_speed}°, 攻击半径 {self.splash_radius}"
        )


# 将子类添加到类映射中
ENEMY_CLASSES["infantry"] = InfantryEnemy
ENEMY_CLASSES["armored_vehicle"] = ArmoredVehicle


# main_json_file = './unit/unit_blue.json'
# trajectory_json_file = './1x_soilder_path_processed.json'

# # 加载步兵敌人
# infantry_enemies = Enemy.load_enemies_from_json(
#     main_json_file=main_json_file,
#     trajectory_json_file=trajectory_json_file,
#     unit_type='infantry'
# )

# # 打印步兵敌人的信息
# print("加载的步兵敌人信息：")
# for enemy in infantry_enemies:
#     print(enemy)
