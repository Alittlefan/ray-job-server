import copy
import os
import random
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from collections import defaultdict

import orjson
from tcp_comm import *

from customization.envs.defense_sim.rule_env.utils import *


class simple_env:  #
    def __init__(self, config=None):

        self.config = config
        self.client = CommClient(self.config["client"])
        if not self.client.connected():
            self.client.connect()
        # releated to scenario
        self.red_bigua_pos = config["red_bigua_pos"]
        self.red_bazhua_pos = config["red_bazhua_pos"]
        self.wqpt_pos = config["red_wqpt_pos"]
        self.zcdy_pos = config["red_zcdy_pos"]
        self.blue_deploy_pos = config["blue_deploy_pos"]
        self.blue_target_pos = config["blue_target_pos"]

        self.max_time = config["max_time"]
        self.msg_dict = defaultdict(int)  # data collect dict
        self.dead_dict = {}
        self.blue_pos = {}
        self.red_pos = {}
        self.total_enemy_num = 0
        self.path_points = {}
        self.mode = "normal"
        self.total_bazhua = 0
        self.total_bigua = 0
        self.blue_deaths = 0
        self.red_deaths = 0
        self.msg_dict["blue_dead_pos"] = []
        self.msg_dict["hurt_num"] = 0
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        # 胜负信息字典
        self.result_dict = {
            "blue_deploy_pos": [],
            "soldier_num": 0,
            "vehicle_num": 0,
            "bazhua_pos": [],
            "bigua_pos": [],
            "result": 0,
        }

    def reset(self):
        self.messages = []
        self.units_state = {}
        self._episode_actions = []
        self._step_index = 0
        self.blue_unit_target = []
        self.msg_dict = defaultdict(int)
        self.dead_dict = {}
        self.red_attack_dict = {}
        self.blue_pos = {}
        self.red_pos = {}
        self.path_points = {}
        self.msg_dict["hurt_num"] = 0
        self.is_fired = False
        self.msg_dict["blue_dead_pos"] = []
        self.total_enemy_num = 0
        # 胜负信息字典
        self.result_dict = {
            "blue_deploy_pos": [],
            "soldier_num": 0,
            "vehicle_num": 0,
            "bazhua_pos": [],
            "bigua_pos": [],
            "blue_death": 0,
            "result": 0,
        }
        self.total_bazhua = 0
        self.total_bigua = 0
        self.total_wqpt = 0
        self.total_zcdy = 0
        self.total_soldier = random.randint(
            self.config["total_soldier"][0], self.config["total_soldier"][1]
        )
        self.total_vehicle = random.randint(
            self.config["total_vehicle"][0], self.config["total_vehicle"][1]
        )
        self.total_uav = random.randint(
            self.config["total_uav"][0], self.config["total_uav"][1]
        )
        self.blue_deaths = 0
        self.red_deaths = 0
        self.msg_dict["blue_dead_pos"] = []
        self.msg_dict["hurt_num"] = 0
        # self.scenario_id = random.randint(0,14)
        self.scenario_id = 0
        self.blue_deploy_pos = self.generate_blue_deployment(
            self.total_soldier, self.total_vehicle, self.total_uav, self.scenario_id
        )

    def create_red_units(
        self, bazhua_pos=None, bigua_pos=None, wqpt_pos=None, zcdy_pos=None
    ):
        self.result_dict["bazhua_pos"] = bazhua_pos
        self.result_dict["bigua_pos"] = bigua_pos
        self.result_dict["wqpt_pos"] = wqpt_pos
        self.result_dict["zcdy_pos"] = zcdy_pos
        # if type(bazhua_pos) == tuple:
        #     red_bazhua_pos = [bazhua_pos]
        # 八爪实体创建
        self.red_attack_dict = defaultdict(list)
        bazhua_init_msg: list[ObjectInitActionMsg] = []
        if bazhua_pos is not None:
            self.msg_dict["red_bazhua_pos"] = bazhua_pos
            for i in range(len(bazhua_pos)):
                lat, lon, alt, yaw = bazhua_pos[i]
                bazhua_init_msg.append(
                    ObjectInitActionMsg(
                        timestamps=0,
                        objectName=f"Red_Bazhua_{i}",
                        objectType=ObjectType.ObjectType_UnmannedVehicle_Red_BaZhaoYu,
                        objectCamp=0,
                        objectPos=Position3D(lon, lat, alt),
                        objectOrientation=RotationAngles(yaw),
                    )
                )
                self.total_bazhua += 1
                # 弹药量设置
            bazhua_init_msg.append(
                SetAmmoCountInfoMsg(
                    timestamps=0,
                    objectName="Red_Bazhua_0",
                    ammoCount=orjson.dumps(
                        [
                            AmmoCountInfo(
                                "rc_chn_mag_2rnd_Missile_ldzsyzswqdy_AKD10", 999
                            )
                        ]
                    ).decode("utf-8"),
                )
            )
            self.messages.extend(bazhua_init_msg)

        # 壁挂实体创建
        bigua_init_msg: list[ObjectInitActionMsg] = []
        if bigua_pos is not None:
            self.msg_dict["red_bigua_pos"] = bigua_pos
            for i in range(len(bigua_pos)):
                lat, lon, alt, yaw = bigua_pos[i]
                bigua_init_msg.append(
                    ObjectInitActionMsg(
                        timestamps=0,
                        objectName=f"Red_Bigua_{i}",
                        objectType=ObjectType.ObjectType_UnmannedVehicle_Red_BiGua,
                        objectCamp=0,
                        objectPos=Position3D(lon, lat, alt),
                        objectOrientation=RotationAngles(yaw),
                    )
                )
                self.red_attack_dict[f"Red_Bigua_{i}"] = [(lon, lat, alt)]
                self.total_bigua += 1
            self.messages.extend(bigua_init_msg)

        # 武器平台实体创建
        wqpt_init_msg: list[ObjectInitActionMsg] = []
        if wqpt_pos is not None:
            for i in range(len(wqpt_pos)):
                lat, lon, alt, yaw = wqpt_pos[i]
                wqpt_init_msg.append(
                    ObjectInitActionMsg(
                        timestamps=0,
                        objectName=f"Red_wqpt_{i}",
                        objectType=ObjectType.ObjectType_UnmannedVehicle_Red_AirDefence,
                        objectCamp=0,
                        objectPos=Position3D(lon, lat, alt),
                        objectOrientation=RotationAngles(yaw),
                    )
                )
                self.total_wqpt += 1
            self.messages.extend(wqpt_init_msg)

        # 侦察单元实体创建
        zcdy_init_msg: list[ObjectInitActionMsg] = []
        if zcdy_pos is not None:
            for i in range(len(zcdy_pos)):
                lat, lon, alt, yaw = zcdy_pos[i]
                zcdy_init_msg.append(
                    ObjectInitActionMsg(
                        timestamps=0,
                        objectName=f"Red_zcdy_{i}",
                        objectType=ObjectType.ObjectType_UnmannedVehicle_Red_ZCDY,
                        objectCamp=0,
                        objectPos=Position3D(lon, lat, alt),
                        objectOrientation=RotationAngles(yaw),
                    )
                )
                self.total_zcdy += 1
            self.messages.extend(zcdy_init_msg)
        # 防守区域-bazhua
        defend_msgs = []
        for object in bazhua_init_msg:
            defend_msg = DefendAreaInfoMsg(
                timestamps=time.time(),
                objectName=object.objectName,
                defendAreaInfo=DefendAreaInfo(45, -45, -180, 180),
            )
            defend_msgs.append(defend_msg)

        # 防守区域-bigua
        defend_msgs = []
        for object in bigua_init_msg:
            defend_msg = DefendAreaInfoMsg(
                timestamps=time.time(),
                objectName=object.objectName,
                defendAreaInfo=DefendAreaInfo(40, -80, -135, 135),
            )
            defend_msgs.append(defend_msg)
        self.messages.extend(defend_msgs)

        # 防守区域-wqpt
        defend_msgs = []
        for object in wqpt_init_msg:
            defend_msg = DefendAreaInfoMsg(
                timestamps=time.time(),
                objectName=object.objectName,
                defendAreaInfo=DefendAreaInfo(40, -80, -135, 135),
            )
            defend_msgs.append(defend_msg)
        self.messages.extend(defend_msgs)

        # 防守区域-zcdy
        defend_msgs = []
        for object in zcdy_init_msg:
            defend_msg = DefendAreaInfoMsg(
                timestamps=time.time(),
                objectName=object.objectName,
                defendAreaInfo=DefendAreaInfo(40, -80, -135, 135),
            )
            defend_msgs.append(defend_msg)
        self.messages.extend(defend_msgs)

    def create_blue_units(self, blue_deploy_pos, blue_target_pos):

        # 分散无人机点位到一个小圆圈上
        def spread_lonlat(lon, lat, i, total, radius=0.00008):
            import math

            if total == 1:
                return lon, lat
            angle = 2 * math.pi * i / total
            dlon = radius * math.cos(angle)
            dlat = radius * math.sin(angle)
            return lon + dlon, lat + dlat

        self.result_dict["blue_deploy_pos"] = blue_deploy_pos
        blue_init_msg: list[ObjectInitActionMsg] = []
        soldier_vehicle_group_info = []
        uav_group_info = []
        idx = 0

        self.msg_dict["blue_deploy_pos"] = blue_deploy_pos[:][:4]
        # if type(blue_deploy_pos) == tuple:
        #     blue_deploy_pos = [blue_deploy_pos]

        for i in range(len(blue_deploy_pos)):
            lat, lon, alt, yaw = blue_deploy_pos[i][:4]
            soldier_vehicle_group_info.append([])
            uav_group_info.append([])
            for num_soldier in range(blue_deploy_pos[i][4]):
                blue_init_msg.append(
                    ObjectInitActionMsg(
                        timestamps=time.time(),
                        objectName=f"Blue_Man_{idx}",
                        objectType=ObjectType.ObjectType_Man_Blue_Marine_ShooterM14,
                        objectCamp=1,
                        objectPos=Position3D(lon, lat, alt),
                        objectOrientation=RotationAngles(yaw),
                    )
                )

                soldier_vehicle_group_info[i].append(f"Blue_Man_{idx}")
                self.blue_pos[f"Blue_Man_{idx}"] = [lon, lat, alt]
                self.path_points[f"Blue_Man_{idx}"] = [(lon, lat, alt)]
                idx += 1

            for num_vehicles in range(blue_deploy_pos[i][5]):
                blue_init_msg.append(
                    ObjectInitActionMsg(
                        timestamps=time.time(),
                        objectName=f"Blue_Vehicle_{idx}",
                        objectType=ObjectType.ObjectType_Vehicle_Blue_Marine_LAVAT,
                        objectCamp=1,
                        objectPos=Position3D(lon, lat, alt),
                        objectOrientation=RotationAngles(),
                    )
                )
                soldier_vehicle_group_info[i].append(f"Blue_Vehicle_{idx}")
                self.blue_pos[f"Blue_Vehicle_{idx}"] = [lon, lat, alt]
                self.path_points[f"Blue_Vehicle_{idx}"] = [(lon, lat, alt)]
                idx += 1

            # 防止uav之间碰撞爆炸，出生位置做随机小范围偏差处理

            for num_uav in range(blue_deploy_pos[i][6]):
                # 对点位做随机化处理
                lon, lat = spread_lonlat(
                    lon, lat, i=num_uav, total=blue_deploy_pos[i][6]
                )
                blue_init_msg.append(
                    ObjectInitActionMsg(
                        timestamps=time.time(),
                        objectName=f"Blue_Uav_{idx}",
                        objectType=ObjectType.ObjectType_USA_UAV,
                        objectCamp=1,
                        objectPos=Position3D(lon, lat, 200),
                        objectOrientation=RotationAngles(),
                    )
                )
                uav_group_info[i].append(f"Blue_Uav_{idx}")
                self.blue_pos[f"Blue_Uav_{idx}"] = [lon, lat, alt]
                self.path_points[f"Blue_Uav_{idx}"] = [(lon, lat, alt)]
                idx += 1

        self.total_enemy_num = self.total_soldier + self.total_vehicle + self.total_uav
        self.result_dict["soldier_num"] = self.total_soldier
        self.result_dict["vehicle_num"] = self.total_vehicle
        # 人&车 编组
        group_target_info = []
        group_msgs = []
        uav_group_msgs = []
        uav_group_target_info = []
        for i, group in enumerate(soldier_vehicle_group_info):
            group_msg = CreateGroupInfoMsg(
                timestamps=time.time(),
                objectNameList="#".join(o for o in group),
                groupID=i,
            )
            nameList = group_msg.objectNameList
            nameList = nameList.split("#")
            group_target_info.append({"members": nameList, "targets": (0, 0)})
            group_msgs.append(group_msg)
        # 无人机编组
        for i, group in enumerate(uav_group_info):
            group_msg = CreateGroupInfoMsg(
                timestamps=time.time(),
                objectNameList="#".join(o for o in group),
                groupID=i + len(soldier_vehicle_group_info),
            )
            nameList = group_msg.objectNameList
            nameList = nameList.split("#")
            uav_group_target_info.append({"members": nameList, "targets": (0, 0)})
            uav_group_msgs.append(group_msg)

        # 编组目标
        group_target_msgs = []
        uav_group_target_msgs = []
        for i, group in enumerate(soldier_vehicle_group_info):
            target_msg = SetGroupTargetInfoMsg(
                timestamps=time.time(),
                groupID=i,
                targetPos=(
                    Position3D(
                        blue_target_pos[0][1],
                        blue_target_pos[0][0],
                        blue_target_pos[0][2],
                    )
                    if self.mode == "normal"
                    else Position3D(
                        blue_deploy_pos[i][1],
                        blue_deploy_pos[i][0],
                        blue_deploy_pos[i][2],
                    )
                ),
            )
            group_target_info[i]["targets"] = (
                blue_target_pos[0][0],
                blue_target_pos[0][1],
            )
            self.blue_unit_target.append(
                [
                    {
                        "name": s,
                        "pos": (0, 0),
                        "targets": group_target_info[i]["targets"],
                    }
                    for s in group_target_info[i]["members"]
                ]
            )
            group_target_msgs.append(target_msg)

        for i, group in enumerate(uav_group_info):
            target_msg = SetGroupTargetInfoMsg(
                timestamps=time.time(),
                groupID=i + len(uav_group_info),
                targetPos=(
                    Position3D(
                        blue_target_pos[0][1],
                        blue_target_pos[0][0],
                        blue_target_pos[0][2],
                    )
                    if self.mode == "normal"
                    else Position3D(
                        blue_deploy_pos[i][1],
                        blue_deploy_pos[i][0],
                        blue_deploy_pos[i][2],
                    )
                ),
            )
            uav_group_target_info[i]["targets"] = (
                blue_target_pos[0][0],
                blue_target_pos[0][1],
            )
            self.blue_unit_target.append(
                [
                    {
                        "name": s,
                        "pos": (0, 0),
                        "targets": uav_group_target_info[i]["targets"],
                    }
                    for s in uav_group_target_info[i]["members"]
                ]
            )
            uav_group_target_msgs.append(target_msg)

        # # 编组风格
        # group_style_msgs = []
        # for i, _ in enumerate(blue_group_info):
        #     style_msg = SetCombatModeMsg(
        #         timestamps=time.time(),
        #         groupID=i,
        #         mode=random.choice([0, 1, 2]),
        #     )
        #     group_style_msgs.append(style_msg)

        self.messages.extend(blue_init_msg)
        self.messages.extend(group_msgs)
        self.messages.extend(uav_group_msgs)
        self.messages.extend(group_target_msgs)
        self.messages.extend(uav_group_target_msgs)
        # self.messages.extend(group_style_msgs)

    def count_casualties(self):
        """
        统计红蓝双方死亡数量
        Returns:
            tuple: (红方死亡数, 蓝方死亡数)
        """
        red_deaths = 0
        blue_deaths = 0
        hurt_num = 0
        soldier_death = 0
        vehicle_death = 0
        for unit in self.units_state.values():
            if unit.objectCamp == 1:
                if unit.lifeValue == 0:
                    hurt_num += 1
            # 判断生命值为0且阵营为红方
            if unit.lifeValue == 0:  # 默认值1确保数据缺失时不计入死亡
                if unit.objectCamp == 0:
                    red_deaths += 1
                else:
                    if "Man" in unit.objectName:
                        soldier_death += 1
                    else:
                        vehicle_death += 1
                    blue_deaths += 1
        return red_deaths, blue_deaths, hurt_num, soldier_death, vehicle_death

    def path_record(self):
        for unit in self.units_state.values():
            if unit.objectCamp == 1:
                lon = unit.objectPosition.lon
                lat = unit.objectPosition.lat
                alt = unit.objectPosition.altitude
                self.path_points[unit.objectName].append((lon, lat, alt))

    def step(self):
        self.client.send_message(
            self.client._world_client, SetMsgFrequencyMsg(time.time(), frequency=50000)
        )
        self.client.start_episode(self.messages)
        self.client.send_message(
            self.client._world_client, SimulationSpeedInfoMsg(time.time(), speed=16)
        )  # 速度配置
        start_time = time.time()  # 红蓝方部署完毕后开始计时
        frame = None
        while True:
            time.sleep(0.1)
            frame = self.client.get_frame()
            if frame is None:
                # print("frame is not")
                continue
            # if time.time() - start_time > 60:
            #     print("time stop")
            self.update_units_state(frame)
            # self.frame_manager.save_frame_msg(self.units_state)
            new_red_deaths, new_blue_deaths, hurt_num, soldier_death, vehicle_dath = (
                self.count_casualties()
            )

            # print(time.time() - start_time)
            # if new_blue_deaths>0:
            #     print("车死亡")
            #     break
            # print("红方死亡数:",new_red_deaths,"蓝方死亡数:", new_blue_deaths)
            # frame_check(self.units_state)
            # if hurt_num > 0 :

            #     print(f"bazhua fired: {self.bazhua_pos}")
            #     break
            # print(f"当前帧红方死亡：{new_red_deaths}")
            # print(f"当前帧蓝方死亡：{new_blue_deaths}")
            # print(f"检查frame内容长度:{len(self.units_state)}")
            # self.path_record()
            # print(new_blue_deaths, new_red_deaths)
            if new_blue_deaths != self.blue_deaths:
                for unit in self.units_state.values():
                    if (
                        unit.lifeValue == 0 and unit.objectCamp == 1
                    ):  # 默认值1确保数据缺失时不计入死亡
                        blue_name = unit.objectName
                        if not any(
                            blue_name == dead_pos[0]
                            for dead_pos in self.msg_dict["blue_dead_pos"]
                        ):  # 防止已经死亡的单位被二次记录
                            lon = unit.objectPosition.lon
                            lat = unit.objectPosition.lat
                            alt = unit.objectPosition.altitude
                            self.msg_dict["blue_dead_pos"].append(
                                (blue_name, lon, lat, alt)
                            )
            self.blue_deaths = new_blue_deaths
            self.red_deaths = new_red_deaths
            escape_num = self.calculate_escape_num()
            # print(f"死亡数: {self.blue_deaths}, 撤离数：{escape_num}")
            if self.is_done(self.blue_deaths, self.red_deaths, start_time, escape_num):
                if self.blue_deaths >= (self.total_enemy_num - 2):
                    print("blue units all dead!")
                    break
                    # append_dict_to_json(self.get_alive_msg(),'data_collect\\nor_bigau_pos.json')
                elif self.red_deaths == self.total_bigua + self.total_bazhua:
                    print("red units all dead!")
                    break
                elif time.time() - start_time > self.max_time:
                    print("time out!")
                    break

        result = (
            1
            if (self.blue_deaths >= (self.total_enemy_num - 2) or escape_num <= 2)
            else 0
        )
        print(f"蓝方死亡数量：{new_blue_deaths}")
        self.result_dict["blue_death"] = new_blue_deaths
        self.result_dict["soldier_death"] = soldier_death
        self.result_dict["vehicle_death"] = vehicle_dath
        self.result_dict["escape_num"] = escape_num
        self.result_dict["result"] = result
        # self.client.clear_frames()
        # print("frame帧数据写入成功")
        # for unit in self.units_state.values():
        # if unit.objectCamp == 0:
        #     # print(f"人员：{unit.objectName}:{unit.ammos[0].ammoNum}")
        #     if unit.ammos[0].ammoNum <1200 :
        #         unit_name = unit.objectName
        #         initial_pos = self.red_attack_dict.get(unit_name)
        #         self.msg_dict["attack_bigua"].append((unit_name,initial_pos))
        self.msg_dict["hurt_num"] = hurt_num

        self.msg_dict["is_blue_dead"] = True if self.blue_deaths > 0 else False
        self.msg_dict["is_red_dead"] = True if self.red_deaths > 0 else False
        self.msg_dict["result"] = (
            True if self.blue_deaths == self.total_enemy_num else False
        )
        self.frame_manager.epoch_idx += 1
        self.client.end_episode()
        print("this epoch ends")

    def get_alive_msg(self):
        alive_unit = defaultdict()
        for unit in self.units_state.values():
            if unit.lifeValue == 0 and unit.objectCamp == 1:
                name = unit.objectName
                lon = unit.objectPosition.lon
                lat = unit.objectPosition.lat
                alt = unit.objectPosition.altitude
                alive_unit[name] = (lon, lat, alt)
        return alive_unit

    def update_units_state(self, frame):
        """
        更新状态字典
        Args:
            frame_data: 新的一帧数据
        """
        new_objects = frame.objects

        for object_name, object_data in new_objects.items():
            self.units_state[object_name] = copy.deepcopy(object_data)

    def is_done(self, blue_deaths, red_deaths, start_time, escape_num):
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
            blue_deaths >= (self.total_enemy_num - 2)
            or red_deaths == self.total_bigua + self.total_bazhua
            or time.time() - start_time > self.max_time
            # TODO: blue_group all arrive the end
            or (escape_num >= 2)
        )

    def is_blue_group_all_arrived(self, blue_unit_target):
        if self.units_state == {}:
            return False
        for group in blue_unit_target:
            for blue_unit in group:
                if blue_unit["name"] in self.units_state.keys():
                    lat = self.units_state[blue_unit["name"]].objectPosition.lat
                    lon = self.units_state[blue_unit["name"]].objectPosition.lon
                    if (
                        abs(lat - blue_unit["targets"][0]) > 0.00105
                        or abs(lon - blue_unit["targets"][1]) > 0.00105
                    ):
                        if self.units_state[blue_unit["name"]].lifeValue == 0:
                            continue
                        return False
        return True

    def generate_blue_deployment(
        self, total_men, total_vehicles, total_uav, scenario_id
    ):
        """
        生成蓝方部署信息

        Args:
            total_men (int): 蓝方总人数
            total_vehicles (int): 蓝方总车辆数
            total_uav (int) : 蓝方总无人机数
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

        uav_per_position = total_uav // num_active
        uav_remainder = total_uav % num_active

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

                # 分配无人机(考虑余数)
                position_uav = uav_per_position
                if active_positions.index(i) < uav_remainder:
                    position_uav += 1

                # 添加完整的部署信息
                pos_info = base_positions[i] + (
                    position_men,
                    position_vehicles,
                    position_uav,
                )

                deployment.append(pos_info)

        return deployment

    def calculate_escape_num(self):
        escape_num = 0
        if self.units_state == {}:
            return False
        for group in self.blue_unit_target:
            for blue_unit in group:
                if self.units_state[blue_unit["name"]].lifeValue == 0:
                    continue
                lat = self.units_state[blue_unit["name"]].objectPosition.lat
                lon = self.units_state[blue_unit["name"]].objectPosition.lon
                if (
                    abs(lat - blue_unit["targets"][0]) <= 0.00105
                    and abs(lon - blue_unit["targets"][1]) <= 0.00043
                ):
                    escape_num += 1
        return escape_num
