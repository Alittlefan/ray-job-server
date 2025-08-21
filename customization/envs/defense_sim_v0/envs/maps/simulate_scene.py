import copy
import json
import logging
import os
import re
import time
import uuid
from typing import Dict, List

from customization.envs.defense_sim_v0.envs.maps import coord_transformer
from customization.envs.defense_sim_v0.envs.maps.unit.unit_blue import (
    ArmoredVehicle,
    Enemy,
    InfantryEnemy,
)
from customization.envs.defense_sim_v0.envs.maps.unit.unit_red import Turret

# 创建模块级日志器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 设置日志级别

# 避免日志传播到根日志器
logger.propagate = False


class Scene:
    def __init__(self):
        """
        初始化场景对象，加载红方炮塔和蓝方单位。

        :param red_json_file: 红方炮塔的JSON配置文件路径。
        :param red_positions_and_angles: 红方炮塔的位置和角度数组。
        :param blue_file_path: 包含蓝方单位规格的字典，键为unit_type，值为规格字典。
        """
        dir_path = os.path.dirname(__file__)
        red_json_file = os.path.join(dir_path, "unit/unit_red.json")
        self.red_json_file = red_json_file
        self.blue_file_path = {
            "infantry": {
                "unit_type": "infantry",
                "main_json_file": os.path.join(dir_path, "unit/unit_blue.json"),
                "trajectory_json_file": os.path.join(
                    dir_path, "unit/trajectories/1x_soilder_path_processed.json"
                ),
            },
            "armored_vehicle": {
                "unit_type": "armored_vehicle",
                "main_json_file": os.path.join(dir_path, "unit/unit_blue.json"),
                "trajectory_json_file": os.path.join(
                    dir_path, "unit/trajectories/1x_vehicle_path_processed.json"
                ),
            },
        }
        self.red_file_path = {}
        self.red_side_turrets = []
        self.blue_side_units = []
        self.los_file_path = os.path.join(dir_path, "los_table.json")
        self.los_data = {}  # 初始化 LOS 数据字典

    def load_los_data(self):
        """
        加载通视表数据并存储在 self.los_data 中。
        通视表数据结构假设为字典，键为 (turret_pos, enemy_pos) 元组，值为可视性布尔值。
        """
        try:
            with open(self.los_file_path, "r") as file:
                for line in file:
                    entry = json.loads(line)
                    turret_pos = tuple(entry["turret_pos"])  # 转换为元组以便用作键
                    enemy_pos = tuple(entry["enemy_pos"])
                    visible = entry["visible"]
                    self.los_data[(turret_pos, enemy_pos)] = visible
        except FileNotFoundError:
            logging.error(f"找不到通视表文件: {self.los_file_path}")
        except json.JSONDecodeError as e:
            logging.error(f"通视表文件格式错误: {e}")
        except Exception as e:
            logging.error(f"加载通视表时发生未预料的错误: {e}")

    def update_units(self, delta_time: float):
        """
        更新所有单位的状态。

        :param delta_time: 时间步长（秒）。
        """
        # 更新蓝方单位的行为
        for unit in self.blue_side_units:
            if not unit.is_alive:
                continue
            unit.move(delta_time=delta_time)

        # 更新红方炮塔的行为（扫描和攻击）
        for turret in self.red_side_turrets:
            turret.scan_and_attack(self.blue_side_units, delta_time=delta_time)

    def check_end_conditions(self) -> bool:
        """
        检查游戏是否需要结束：所有敌人死亡或存活的敌人到达路径终点。

        Returns:
            bool: 如果需要结束游戏，返回 True；否则返回 False。
        """
        all_dead = all(
            not unit.is_alive for unit in self.blue_side_units
        )  # 检查敌人是否全部死亡
        all_at_end = all(
            not unit.is_alive or unit.current_position == unit.path[-1]
            for unit in self.blue_side_units
            if unit.path
        )  # 检查敌人是否到达路径终点（跳过没有路径的单位）

        if all_dead:
            # logging.info("所有敌人已被摧毁。")
            return True

        if all_at_end:
            # logging.info("所有存活的敌人到达路径终点。")
            return True

        return False

    def start_episode(self, scenario):
        """
        从 scenario 中提取蓝方和红方单位的类型及位置，并加载蓝方单位。

        Args:
            scenario (ScenarioManager): 场景管理器实例，包含蓝方和红方的部署信息。

        Returns:
            Tuple[int, int, bool]: 步兵死亡数，装甲车死亡数，红方是否胜利。
        """

        # 加载通视表数据
        self.load_los_data()

        # 提取蓝方和红方单位的信息
        blue_units_info = self.extract_blue_units(scenario)
        red_units_info = self.extract_red_units(scenario)

        # 加载红方炮塔
        self.red_side_turrets = []
        for red_unit in red_units_info:
            try:
                # 提取位置和角度
                position = red_unit["position"]
                lat, lon, alt, yaw = (
                    position["lat"],
                    position["lon"],
                    position["alt"],
                    position["yaw"],
                )
                x, y, z = self.convert_position({"lat": lat, "alt": alt, "lon": lon})
                position_and_angle = [x, z, y, yaw]

                # 提取炮塔类型
                turret_type = red_unit.get("type", "").lower()
                if not turret_type:
                    logger.warning(f"红方单位缺少炮塔类型信息，跳过该单位: {red_unit}")
                    continue

                # 调用 load_turret_from_json 方法加载单个炮塔，并传递 LOS 数据
                turret = Turret.load_turret_from_json(
                    self.red_json_file,
                    position_and_angle,
                    turret_type,
                    self.los_data,  # 确保这里传递了 los_data
                )
                if turret:
                    self.red_side_turrets.append(turret)
                    logger.info(
                        f"成功加载红方炮塔类型 '{turret_type}'，位置: {position_and_angle}"
                    )
                else:
                    logger.warning(
                        f"加载红方炮塔失败，类型: '{turret_type}'，位置: {position_and_angle}"
                    )

            except KeyError as e:
                logger.error(f"红方单位缺少必要的键: {e}，单位信息: {red_unit}")
            except Exception as e:
                logger.error(f"加载红方炮塔时发生未预料的错误: {e}")

        logger.info(f"成功加载了 {len(self.red_side_turrets)} 个红方炮塔。")

        # 根据蓝方单位的信息逐个创建蓝方单位
        blue_units = []
        for unit_info in blue_units_info:
            unit_type = unit_info["type"]
            position_geo = unit_info["position"]
            converted = self.convert_position(position_geo)
            # x, y, z, angle = converted
            angle = 0
            x, y, z = converted
            coord = (x, z, y)

            # 查找对应的 enemy_spec
            spec = self.blue_file_path.get(unit_type.lower())
            if not spec:
                logger.warning(f"没有找到单位类型 '{unit_type}' 的规格，跳过该单位。")
                continue

            # 加载敌人，传入具体的 coord
            try:
                enemies = Enemy.load_enemies_from_json(
                    main_json_file=spec["main_json_file"],
                    trajectory_json_file=spec["trajectory_json_file"],
                    unit_type=unit_type,
                    coord=coord,
                )
                if enemies:
                    enemy = enemies[0]  # 每次传入一个 coord，只取第一个
                    # 覆盖敌人的初始位置和朝向
                    enemy.current_position = coord
                    enemy.position = coord
                    enemy.orientation_angle = angle
                    blue_units.append(enemy)
                    logger.info(f"创建蓝方单位 '{unit_type}'，ID: {enemy.id}")
                else:
                    logger.warning(f"未能创建蓝方单位 '{unit_type}'。")
            except Exception as e:
                logger.error(f"创建蓝方单位 '{unit_type}' 时出错: {e}")

        self.blue_side_units = blue_units
        logger.info(f"成功创建了 {len(self.blue_side_units)} 个蓝方单位。")

        # 初始化回放数据
        replay_data = {
            "units": {
                "red": [turret.to_dict() for turret in self.red_side_turrets],
                "blue": [unit.to_dict() for unit in self.blue_side_units],
            },
            "timesteps": [],  # 将在模拟过程中填充
        }

        # 构建 episode 数据
        episode_data = {
            "blue_units": blue_units_info,
            "red_units": red_units_info,
            "timestamp": time.time(),
        }
        # # 示例：将数据序列化为 JSON 并发送
        # message_str = json.dumps(episode_data)
        # self.send_message(message_str)

        total_time = 6000
        time_step = 0.5
        elapsed_time = 0.0

        # 初始化死亡计数器
        infantry_deaths = 0
        vehicle_deaths = 0

        # 为避免重复计数，使用集合记录已死亡单位的ID
        dead_infantry_ids = set()
        dead_vehicle_ids = set()

        while elapsed_time < total_time:
            logger.info(f"--- 时间步 {elapsed_time:.1f} 秒 ---")

            # 更新场景内所有单位状态
            self.update_units(delta_time=time_step)

            # 获取场景信息
            # current_info = self.get_info()
            # print(current_info)
            # 获取当前时间步的信息
            current_info = self.get_info()

            # 将当前时间步的信息添加到回放数据中
            replay_data["timesteps"].append(current_info)

            # frame = self.get_frame(current_info)
            # print(frame)
            logger.info(
                f"当前场景信息: {json.dumps(current_info, indent=2, ensure_ascii=False)}"
            )

            # 更新死亡计数
            for unit in self.blue_side_units:
                if not unit.is_alive:
                    if (
                        isinstance(unit, InfantryEnemy)
                        and unit.id not in dead_infantry_ids
                    ):
                        infantry_deaths += 1
                        dead_infantry_ids.add(unit.id)
                    elif (
                        isinstance(unit, ArmoredVehicle)
                        and unit.id not in dead_vehicle_ids
                    ):
                        vehicle_deaths += 1
                        dead_vehicle_ids.add(unit.id)

            elapsed_time += time_step

            # 检查是否需要结束游戏
            if self.check_end_conditions():
                logger.info("游戏结束：所有敌人已被摧毁或到达路径终点。")
                break

            # time.sleep(time_step)

        logger.info("模拟结束。")

        # 确定红方是否胜利（所有蓝方单位已死亡）
        red_victory = all(not unit.is_alive for unit in self.blue_side_units)

        logger.info(f"步兵死亡数: {infantry_deaths}")
        logger.info(f"装甲车死亡数: {vehicle_deaths}")
        logger.info(f"红方是否胜利: {red_victory}")

        # 保存回放数据
        self.save_replay(replay_data)

        return infantry_deaths, vehicle_deaths, red_victory

    def transform_info(self, data: dict, dx: float = 0, dy: float = 0):
        """
        提取数据中的所有位置，对其进行坐标平移变化后再放回数据中。

        参数：
            data (dict): 包含红色单位和蓝色单位数据的字典。
            dx (float): x 方向的平移量。
            dy (float): y 方向的平移量。

        返回：
            dict: 修改后的数据。
        """
        # 深拷贝原数据，避免修改原始数据
        modified_data = copy.deepcopy(data)

        # 定义一个正则表达式来匹配位置字段
        position_pattern = re.compile(r"位置 \(([-\d\.]+), ([-\d\.]+), ([-\d\.]+)\)")

        # 遍历红色单位的数据
        for i, unit in enumerate(modified_data["red_units"]):
            # 查找位置字段
            match = position_pattern.search(unit)
            if match:
                # 提取原始坐标
                x, y, z = map(float, match.groups())
                # 红色单位：x 加 dx，y 加 dy
                new_x = x + dx
                new_y = y + dy
                new_z = z

                lonlat = coord_transformer.CoordinateTransformer().to_lonlat(
                    new_x, new_y, new_z
                )

                # 构造新的位置字符串
                new_position = f"位置 ({lonlat[1]}, {lonlat[0]}, {lonlat[2]})"
                # 替换原始位置字符串
                modified_data["red_units"][i] = position_pattern.sub(new_position, unit)

        # 遍历蓝色单位的数据
        for i, unit in enumerate(modified_data["blue_units"]):
            # 查找位置字段
            match = position_pattern.search(unit)
            if match:
                # 提取原始坐标
                x, y, z = map(float, match.groups())
                # 蓝色单位：x 加 dx，z 加 dy，y 和 z 对调
                new_x = x + dx
                new_y = z + dy
                new_z = y

                lonlat = coord_transformer.CoordinateTransformer().to_lonlat(
                    new_x, new_y, new_z
                )

                # 构造新的位置字符串
                new_position = f"位置 ({lonlat[1]}, {lonlat[0]}, {lonlat[2]})"
                # 替换原始位置字符串
                modified_data["blue_units"][i] = position_pattern.sub(
                    new_position, unit
                )

        return modified_data

    def get_info(self) -> Dict[str, List[str]]:
        """
        获取场景内当前所有单位的信息。

        Returns:
            dict: 包含红方炮塔和蓝方单位信息的字典，格式如下：
                {
                    "red_units": ["红方单位状态字符串", ...],
                    "blue_units": ["蓝方单位状态字符串", ...]
                }
        """
        red_units_info = []
        blue_units_info = []

        # 获取红方炮塔信息
        for turret in self.red_side_turrets:
            try:
                red_units_info.append(repr(turret))
            except Exception as e:
                logging.error(f"获取红方单位信息时出错: {e}")

        # 获取蓝方单位信息
        for unit in self.blue_side_units:
            try:
                blue_units_info.append(repr(unit))
            except Exception as e:
                logging.error(f"获取蓝方单位信息时出错: {e}")

        transformed_info = self.transform_info(
            {"red_units": red_units_info, "blue_units": blue_units_info},
            998833.0,
            999250.0,
        )

        return transformed_info

    def get_frame(self, data):
        frame = {}

        # 处理红方单位
        red_units = data.get("red_units", [])
        for idx, unit in enumerate(red_units):
            if "BazhuaTurret" in unit:
                unit_type = "Red_BaZhua_"
            elif "BiguaTurret" in unit:
                unit_type = "Red_BiGua_"
            else:
                continue

            # 提取ID
            id_start = unit.find("ID: ") + len("ID: ")
            id_end = unit.find(",", id_start)
            unit_id = unit[id_start:id_end].strip()

            # 提取位置
            pos_start = unit.find("位置 (") + len("位置 (")
            pos_end = unit.find(")", pos_start)
            position = unit[pos_start:pos_end].split(", ")
            lat, lon, alt = map(float, position)

            # 提取生命值
            life_start = unit.find("生命值 ") + len("生命值 ")
            life_end = unit.find(",", life_start)
            lifevalue = float(unit[life_start:life_end].strip())

            # 构造键名
            key = f"{unit_type}{idx}"  # 使用索引作为键名的一部分

            # 添加到frame
            frame[key] = {
                "lifevalue": lifevalue,
                "camp": 0,
                "position": {"lat": lat, "lon": lon, "alt": alt},
            }
        blue_units = data.get("blue_units", [])
        for idx, unit in enumerate(blue_units):
            if "步兵" in unit:
                unit_type = "Blue_Man_"
            elif "装甲车" in unit:
                unit_type = "Blue_Vehicle_"
            else:
                continue

            # 提取位置
            pos_start = unit.find("位置 (") + len("位置 (")
            pos_end = unit.find(")", pos_start)
            position = unit[pos_start:pos_end].split(", ")
            lat, lon, alt = map(float, position)

            # 提取生命值
            life_start = unit.find("血量 ") + len("血量 ")
            life_end = unit.find(",", life_start)
            lifevalue = float(unit[life_start:life_end].strip())

            # 构造键名
            key = f"{unit_type}{idx}"

            # 添加到frame
            frame[key] = {
                "lifevalue": lifevalue,
                "camp": 1,
                "position": {"lat": lat, "lon": lon, "alt": alt},
            }

        return frame

    def extract_blue_units(self, scenario) -> List[Dict]:
        """
        从 scenario 中提取所有蓝方单位的类型和位置。

        Args:
            scenario (ScenarioManager): 场景管理器实例。

        Returns:
            List[Dict]: 蓝方单位列表，每个单位包含名称、类型和位置。
        """
        blue_units = []
        for formation in scenario.blue_formations:
            # 修改为使用字典的键访问
            lat, lon, alt, yaw = formation.start_point  # 使用字典访问 start_point
            # 提取人员
            for personnel in formation.personnel_list:  # 使用字典访问 personnel_list
                unit = {
                    "name": personnel,
                    "type": "infantry",  # 根据 blue_file_path 中的 unit_type 调整
                    "position": {"lat": lat, "lon": lon, "alt": alt, "yaw": yaw},
                }
                blue_units.append(unit)
            # 提取车辆
            for vehicle in formation.vehicle_list:  # 使用字典访问 vehicle_list
                unit = {
                    "name": vehicle,
                    "type": "armored_vehicle",  # 根据 blue_file_path 中的 unit_type 调整
                    "position": {"lat": lat, "lon": lon, "alt": alt, "yaw": yaw},
                }
                blue_units.append(unit)
        return blue_units

    def extract_red_units(self, scenario) -> List[Dict]:
        """
        从 scenario 中提取所有红方单位的类型和位置。

        Args:
            scenario (ScenarioManager): 场景管理器实例。

        Returns:
            List[Dict]: 红方单位列表，每个单位包含名称、类型和位置。
        """
        red_units = []
        for turret in scenario.red_turrets:
            # turret_type 是字符串 'bigua' 或 'bazhua'
            turret_type_str = "bigua" if turret.turret_type == 1 else "bazhua"
            if turret_type_str not in ["bigua", "bazhua"]:
                # logging.warning(f"未知的 turret_type '{turret.turret_type}'，跳过该炮塔。")
                continue

            # 确保 position 包含 4 个元素
            if (
                not isinstance(turret.position, (list, tuple))
                or len(turret.position) != 4
            ):
                # logging.warning(f"炮塔位置数据无效：{turret.position}，跳过该炮塔。")
                continue

            lat, lon, alt, yaw = turret.position
            unit = {
                "name": turret.turret_name,
                "type": turret_type_str,  # 直接使用 'bigua' 或 'bazhua'
                "position": {"lat": lat, "lon": lon, "alt": alt, "yaw": yaw},
            }
            red_units.append(unit)
        return red_units

    # def send_message(self, message: str):
    #     """
    #     发送消息的方法。

    #     Args:
    #         message (str): 要发送的消息字符串。
    #     """
    #     # 这里可以实现具体的消息发送逻辑，例如通过网络发送、写入日志文件等
    #     logging.info(f"发送消息: {message}")

    def save_replay(self, replay_data: Dict):
        """
        将回放数据保存为一个唯一的 JSON 文件，存放在 replay 文件夹下。
        """
        try:
            # 确保 replay 文件夹存在
            os.makedirs("replay", exist_ok=True)

            # 生成唯一文件名
            replay_filename = f"replay_{uuid.uuid4()}.json"
            replay_filepath = os.path.join("replay", replay_filename)

            # 保存回放数据到 JSON 文件
            with open(replay_filepath, "w", encoding="utf-8") as f:
                json.dump(replay_data, f, ensure_ascii=False, indent=4)

            logger.info(f"回放文件已保存: {replay_filepath}")

        except Exception as e:
            logger.error(f"保存回放文件时发生错误: {e}")

    def convert_position(self, position: Dict[str, float]) -> List[float]:
        """
        将地理位置转换为笛卡尔坐标并进行平移。
        """
        try:
            # 提取输入参数
            lat = position["lat"]
            lon = position["lon"]
            alt = position["alt"]

            # 调用转换工具进行地理坐标到笛卡尔坐标的转换
            new_x, new_y, new_z = coord_transformer.CoordinateTransformer().to_xyz(
                lon, lat, alt
            )

            # 对坐标进行平移调整
            new_x -= 998833.0
            new_y -= 999250.0

            # 返回转换后的结果，不包括 yaw
            return [new_x, new_y, new_z]

        except KeyError as e:
            # logging.error(f"缺少必要的键: {e}")
            raise ValueError(f"输入字典缺少必要的键: {e}")

        except Exception as e:
            # logging.error(f"转换位置时出错: {e}")
            raise RuntimeError(f"位置转换失败: {e}")
