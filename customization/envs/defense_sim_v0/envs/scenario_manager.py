import random
import time
from typing import Dict, List

from tcp_comm import *

from customization.envs.defense_sim_v0.envs.data_types import (
    Formation,
    ScenarioConfig,
    TurretDeployment,
)


class ScenarioManager:
    """场景管理类"""

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.messages: List = []

        # 蓝方部署
        self.blue_formations: List[Formation] = []
        self.blue_unit_target: List = []

        # 红方部署
        self.red_turrets: List[TurretDeployment] = []

        # 状态追踪
        self.units_state: Dict = {}

        self.conv_pos = {}
        self.action_mapping(self.config.red_bigua_pos, self.config.red_bazhua_pos)

    def action_mapping(self, red_bazhua_pos, red_bigua_pos):
        self.conv_pos = {}
        index = 0
        # 映射bazhua部署点
        for pos in red_bazhua_pos:
            self.conv_pos[index] = pos
            index += 1
        # 映射bigua部署点
        for pos in red_bigua_pos:
            self.conv_pos[index] = pos
            index += 1

    def reset(self):
        """重置场景状态"""
        self.messages.clear()
        self.blue_formations.clear()
        self.red_turrets.clear()
        self.units_state.clear()
        self.blue_unit_target.clear()

        Formation.reset_ids()
        TurretDeployment.reset_ids()

    def create_blue_formations(self):
        """创建蓝方编组"""
        # 随机选择蓝方配置
        blue_deploy = random.choice(self.config.blue_deploy_pos)
        for deploy_pos in blue_deploy:
            start_pos = deploy_pos[:4]
            num_blue_man = deploy_pos[4]
            num_blue_vehicle = deploy_pos[5]
            formation = Formation.create_formation(
                start=start_pos,
                dest=self.config.blue_target_pos[0],
                personnel_count=num_blue_man,
                vehicle_count=num_blue_vehicle,
            )
            self.blue_formations.append(formation)

    def create_red_turrets(self, actions: List[List[float]]):
        """创建红方炮塔"""
        for action in actions:
            turret = TurretDeployment.create_turret(
                pos=tuple(action[:4]), turret_type=int(action[4])
            )
            self.red_turrets.append(turret)

    def generate_unit_messages(self):
        """生成所有单位的初始化消息"""
        self._generate_blue_messages()
        self._generate_red_messages()

    def _generate_blue_messages(self):
        """生成蓝方相关消息"""
        blue_init_msg = []
        blue_group_info = []

        # 创建单位初始化消息
        for i, formation in enumerate(self.blue_formations):
            group_members = []
            lat, lon, alt, yaw = formation.start_point
            if len(formation.personnel_list) == 0 and len(formation.vehicle_list) == 0:
                continue

            # 添加人员
            for name in formation.personnel_list:
                blue_init_msg.append(
                    ObjectInitActionMsg(
                        timestamps=time.time(),
                        objectName=name,
                        objectType=ObjectType.ObjectType_Man_Blue_Marine_ShooterM14,
                        objectCamp=1,
                        objectPos=Position3D(lon, lat, alt),
                        objectOrientation=RotationAngles(yaw),
                    )
                )
                group_members.append(name)

            # 添加车辆
            for name in formation.vehicle_list:
                blue_init_msg.append(
                    ObjectInitActionMsg(
                        timestamps=time.time(),
                        objectName=name,
                        objectType=ObjectType.ObjectType_Vehicle_Blue_Marine_LAVAT,
                        objectCamp=1,
                        objectPos=Position3D(lon, lat, alt),
                        objectOrientation=RotationAngles(yaw),
                    )
                )
                group_members.append(name)

            blue_group_info.append(group_members)

        # 创建编组相关消息
        self.messages.extend(blue_init_msg)
        self._create_group_messages(blue_group_info)

    def _generate_red_messages(self):
        """生成红方相关消息"""
        red_init_msg = []
        defend_msgs = []

        for turret in self.red_turrets:
            # 炮塔初始化消息
            red_init_msg.append(
                ObjectInitActionMsg(
                    timestamps=time.time(),
                    objectName=turret.turret_name,
                    objectType=(
                        ObjectType.ObjectType_UnmannedVehicle_Red_BiGua
                        if turret.turret_type == 1
                        else ObjectType.ObjectType_UnmannedVehicle_Red_BaZhaoYu
                    ),
                    objectCamp=0,
                    objectPos=Position3D(
                        turret.position[1], turret.position[0], turret.position[2]
                    ),
                    objectOrientation=RotationAngles(turret.position[3]),
                )
            )

            # 防御区域消息
            defend_msgs.append(
                DefendAreaInfoMsg(
                    timestamps=time.time(),
                    objectName=turret.turret_name,
                    defendAreaInfo=DefendAreaInfo(*turret.defend_area),
                )
            )

        self.messages.extend(red_init_msg)
        self.messages.extend(defend_msgs)

    def _create_group_messages(self, blue_group_info):
        """创建蓝方编组相关消息"""
        group_msgs = []
        target_msgs = []
        style_msgs = []
        group_target_info = []

        # 编组消息
        for i, group in enumerate(blue_group_info):
            group_msg = CreateGroupInfoMsg(
                timestamps=time.time(),
                objectNameList="#".join(o for o in group),
                groupID=i,
            )
            group_msgs.append(group_msg)

            # 记录编组成员
            nameList = group_msg.objectNameList.split("#")
            group_target_info.append({"members": nameList, "targets": (0, 0)})

        # 目标点消息
        for i, formation in enumerate(self.blue_formations):
            lat, lon, alt = formation.destination
            if len(formation.personnel_list) == 0 and len(formation.vehicle_list) == 0:
                continue
            target_msg = SetGroupTargetInfoMsg(
                timestamps=time.time(),
                groupID=i,
                targetPos=Position3D(lon, lat, alt),
            )
            target_msgs.append(target_msg)

            # 记录目标点
            group_target_info[i]["targets"] = (lon, lat)
            self.blue_unit_target.append(
                [
                    {"name": s, "targets": group_target_info[i]["targets"]}
                    for s in group_target_info[i]["members"]
                ]
            )

        # 战斗风格消息
        for i in range(len(blue_group_info)):
            style_msg = SetCombatModeMsg(
                timestamps=time.time(),
                groupID=i,
                mode=random.choice([0, 1, 2]),
            )
            style_msgs.append(style_msg)

        self.messages.extend(group_msgs)
        self.messages.extend(target_msgs)
        self.messages.extend(style_msgs)
