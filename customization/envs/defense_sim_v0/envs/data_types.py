from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Formation:
    start_point: Tuple[float, float, float, float]
    destination: Tuple[float, float, float]
    personnel_list: List[str]
    vehicle_list: List[str]

    # 类变量用于跟踪ID
    _personnel_id = 0
    _vehicle_id = 0

    @classmethod
    def create_formation(
        cls,
        start: Tuple[float, float, float, float],
        dest: Tuple[float, float, float],
        personnel_count: int,
        vehicle_count: int,
    ) -> "Formation":
        """工厂方法：根据人员和车辆数量创建编组，并自增ID"""
        personnel = []
        for _ in range(personnel_count):
            personnel.append(f"Blue_Man_{cls._personnel_id}")
            cls._personnel_id += 1

        vehicles = []
        for _ in range(vehicle_count):
            vehicles.append(f"Blue_Vehicle_{cls._vehicle_id}")
            cls._vehicle_id += 1

        return cls(start, dest, personnel, vehicles)

    @classmethod
    def reset_ids(cls):
        """重置ID计数器"""
        cls._personnel_id = 0
        cls._vehicle_id = 0

    def __str__(self) -> str:
        return (
            f"Formation(\n"
            f"  Start: {self.start_point}\n"
            f"  Destination: {self.destination}\n"
            f"  Personnel: {self.personnel_list}\n"
            f"  Vehicles: {self.vehicle_list}\n"
            f")"
        )


@dataclass
class TurretDeployment:
    position: Tuple[float, float, float, float]  # 经度、维度、高度、朝向
    turret_name: str
    turret_type: int  # 1或2

    _bigua_id = 0  # 壁挂类型的ID计数器
    _bazhua_id = 0  # 八爪类型的ID计数器

    @classmethod
    def create_turret(
        cls, pos: Tuple[float, float, float, float], turret_type: int
    ) -> "TurretDeployment":
        """工厂方法：根据位置和类型创建炮塔"""
        if turret_type not in [1, 2]:
            raise ValueError("Turret type must be 1 (BiGua) or 2 (BaZhua)")

        if turret_type == 1:
            turret_name = f"Red_BiGua_{cls._bigua_id}"
            cls._bigua_id += 1
        else:
            turret_name = f"Red_BaZhua_{cls._bazhua_id}"
            cls._bazhua_id += 1

        return cls(pos, turret_name, turret_type)

    @classmethod
    def reset_ids(cls):
        """重置ID计数器"""
        cls._bigua_id = 0
        cls._bazhua_id = 0

    def __str__(self) -> str:
        turret_type_str = "BiGua" if self.turret_type == 1 else "BaZhua"
        return (
            f"TurretDeployment(\n"
            f"  Position: {self.position}\n"
            f"  Name: {self.turret_name}\n"
            f"  Type: {turret_type_str}\n"
            f")"
        )

    @property
    def defend_area(self) -> Tuple[float, float, float, float]:
        """获取防御区域参数"""
        if self.turret_type == 1:  # 壁挂
            return (40, -80, -90, 90)
        else:  # 八爪
            return (45, -45, -90, 90)


@dataclass
class ScenarioConfig:
    """场景配置数据类"""

    time_limit: float
    blue_deploy_pos: List[List[Tuple[float, float, float, float, int, int]]]
    blue_target_pos: List[Tuple[float, float, float]]
    red_bigua_pos: List[Tuple[float, float, float, float]]
    red_bazhua_pos: List[Tuple[float, float, float, float]]
    num_blue_group: int
    num_blue_man: int
    num_blue_vehicle: int
    num_red_bigua: int
    num_red_bazhua: int
