import copy
import random
import time

import gymnasium as gym
import numpy as np
import pyproj
from gymnasium import spaces
from tcp_comm import *


class DefenseGymV0(gym.Env):
    def __init__(self, config=None):
        super(DefenseGymV0, self).__init__()
        self.config = config

        self.client = CommClient(self.config["client"])

        # releated to scenario
        self.red_bigua_pos = config["red_bigua_pos"]
        self.red_bazhua_pos = config["red_bazhua_pos"]
        self.blue_deploy_pos = config["blue_deploy_pos"]
        self.blue_target_pos = config["blue_target_pos"]
        self.num_red_bigua = config["num_red_bigua"]
        self.num_red_bazhua = config["num_red_bazhua"]
        self.num_blue_man = config["num_blue_man"]
        self.num_blue_vehicle = config["num_blue_vehicle"]
        self.max_time = config["max_time"]

        self.action_space = spaces.Discrete(
            len(self.red_bigua_pos) + len(self.red_bazhua_pos)
        )
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        4 * 1
                        + 3 * (len(self.red_bazhua_pos) + len(self.red_bigua_pos)),
                    ),
                    dtype=np.float32,
                )
            }
        )
        self._action_mask = np.ones(
            len(self.red_bigua_pos) + len(self.red_bazhua_pos), dtype=np.int8
        )

        self._episode_step_num = self.num_red_bazhua + self.num_red_bigua
        self._episode_actions = []
        self._step_index = 0

        self.messages = []

        self.units_state = {}
        self.conv_pos = {}
        self.actual_blue_deploy_pos = []
        self.blue_unit_target = []

        wgs84 = pyproj.CRS("EPSG:4326")
        geocentric = pyproj.CRS("EPSG:32651")

        self.transformer_wgs84 = pyproj.Transformer.from_crs(
            geocentric, wgs84, always_xy=True
        )
        self.transformer_geocentric = pyproj.Transformer.from_crs(
            wgs84, geocentric, always_xy=True
        )

    def reset(self, seed=None, options=None):
        # selcet sceanrio
        if not self.client.connected():
            self.client.connect()

        self.actual_blue_deploy_pos = []
        self.actual_blue_deploy_pos.append(
            self.blue_deploy_pos[random.randint(0, len(self.blue_deploy_pos) - 1)]
        )
        self.messages = []
        self.units_state = {}
        self._episode_actions = []
        self._step_index = 0
        self.blue_unit_target = []

        self.create_blue_units(
            self.num_blue_man,
            self.num_blue_vehicle,
            self.actual_blue_deploy_pos,
            self.blue_target_pos,
        )
        self.action_mapping(self.red_bazhua_pos, self.red_bigua_pos)

        self.state = self.get_observation(self.actual_blue_deploy_pos)
        self._action_mask = np.ones(
            len(self.red_bigua_pos) + len(self.red_bazhua_pos), dtype=np.int8
        )
        return {"obs": self.state}, {}

    def step(self, action):
        # self.update_action_mask(action)
        c_action = self.convert_action(action)

        self._episode_actions.append(c_action)
        self.update_state(c_action, self.actual_blue_deploy_pos)

        if self._step_index < self._episode_step_num - 1:
            self._step_index += 1
            return {"obs": self.state}, 0, False, False, {}

        self.create_red_units(self._episode_actions)
        self.client.start_episode(self.messages)
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
            if self.is_done(blue_deaths, red_deaths, start_time):
                break

        result = self.client.end_episode()
        win = self.is_win(blue_deaths)
        reward = self.get_reward(blue_deaths, win)
        return {"obs": self.state}, reward, True, False, {}

    def close(self):
        self.client.disconnect()

    def to_xyz(
        self, lon, lat, alt, offsets=(644911.9272301428, 1769688.8120349043, 0.0)
    ):
        x, y, z = self.transformer_geocentric.transform(lon, lat, alt)
        x -= offsets[0]
        y -= offsets[1]
        z -= offsets[2]
        return x, y, z

    def update_units_state(self, frame: Frame):
        """Update the state of units

        Args:
            frame_data: New frame data
        """
        new_objects = frame.objects

        for object_name, object_data in new_objects.items():
            self.units_state[object_name] = copy.deepcopy(object_data)

    def count_casualties(self):
        """Count the number of deaths on both sides

        Returns:
            tuple: (blue_deaths, red_deaths)
        """
        red_deaths = 0
        blue_deaths = 0

        for unit in self.units_state.values():
            if unit.lifeValue == 0:
                if unit.objectCamp == 0:
                    red_deaths += 1
                elif unit.objectCamp == 1:
                    blue_deaths += 1
        return blue_deaths, red_deaths

    def is_done(self, blue_deaths, red_deaths, start_time):
        """Check if the episode is done

        Args:
            blue_deaths: Number of blue deaths
            red_deaths: Number of red deaths
            start_time: Start time of the episode

        Returns:
            bool: Whether the episode is done
        """
        return (
            blue_deaths
            == len(self.actual_blue_deploy_pos)
            * (self.num_blue_man + self.num_blue_vehicle)
            or red_deaths == self.num_red_bigua + self.num_red_bazhua
            or time.time() - start_time > self.max_time
            or self.is_blue_group_all_arrived(self.blue_unit_target)
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
                        abs(lat - blue_unit["targets"][0]) > 1e-3
                        or abs(lon - blue_unit["targets"][1]) > 1e-3
                    ):
                        if self.units_state[blue_unit["name"]].lifeValue == 0:
                            continue
                        return False
        return True

    def is_win(self, blue_deaths):
        return blue_deaths == len(self.actual_blue_deploy_pos) * (
            self.num_blue_man + self.num_blue_vehicle
        )

    def get_reward(self, blue_deaths, win):
        return blue_deaths * 5 + 100 if win else -100

    def create_blue_units(
        self, num_blue_man, num_blue_vehicle, blue_deploy_pos, blue_target_pos
    ):
        blue_init_msg: list[ObjectInitActionMsg] = []
        blue_group_info = []
        idx = 0
        for i in range(len(blue_deploy_pos)):
            lat, lon, alt, yaw = blue_deploy_pos[i]
            blue_group_info.append([])
            for j in range(num_blue_man):
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
                blue_group_info[i].append(f"Blue_Man_{idx}")
                idx += 1
            for k in range(num_blue_vehicle):
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
                blue_group_info[i].append(f"Blue_Vehicle_{idx}")
                idx += 1

        # Group info
        group_target_info = []
        group_msgs = []
        for i, group in enumerate(blue_group_info):
            group_msg = CreateGroupInfoMsg(
                timestamps=time.time(),
                objectNameList="#".join(o for o in group),
                groupID=i,
            )
            nameList = group_msg.objectNameList
            nameList = nameList.split("#")
            group_target_info.append({"members": nameList, "targets": (0, 0)})
            group_msgs.append(group_msg)

        # Group target
        group_target_msgs = []
        for i, group in enumerate(blue_group_info):
            target_msg = SetGroupTargetInfoMsg(
                timestamps=time.time(),
                groupID=i,
                targetPos=Position3D(
                    blue_target_pos[i][1], blue_target_pos[i][0], blue_target_pos[i][2]
                ),
            )
            group_target_info[i]["targets"] = (
                blue_target_pos[i][0],
                blue_target_pos[i][1],
            )
            self.blue_unit_target.append(
                [
                    {"name": s, "targets": group_target_info[i]["targets"]}
                    for s in group_target_info[i]["members"]
                ]
            )
            group_target_msgs.append(target_msg)

        # Group style
        group_style_msgs = []
        for i, _ in enumerate(blue_group_info):
            style_msg = SetCombatModeMsg(
                timestamps=time.time(),
                groupID=i,
                mode=random.choice([0, 1, 2]),
            )
            group_style_msgs.append(style_msg)

        self.messages.extend(blue_init_msg)
        self.messages.extend(group_msgs)
        self.messages.extend(group_target_msgs)
        self.messages.extend(group_style_msgs)

    def get_observation(self, actual_blue_deploy_pos):
        observation = []
        observation.extend(
            [
                actual_blue_deploy_pos[0][0],
                actual_blue_deploy_pos[0][1],
                self.num_blue_man,
                self.num_blue_vehicle,
            ]
        )
        for pos in self.red_bigua_pos + self.red_bazhua_pos:
            observation.extend([pos[0], pos[1], 0])

        return np.array(observation)

    def update_action_mask(self, action):
        self._action_mask[action] = 0
        if self._step_index < self.num_red_bigua:
            self._action_mask[self.num_red_bigua - 1 :] = 0
        else:
            self._action_mask[: self.num_red_bigua] = 0
            self._action_mask[self.num_red_bigua :] = 1

    def convert_action(self, action):
        action_type = 1 if self._step_index < self.num_red_bigua else 2
        c_action = list(self.conv_pos[action])
        c_action.append(action_type)
        return c_action

    def update_state(self, action, actual_blue_deploy_pos):
        # Calculate the number of blue and red
        n_blue = len(actual_blue_deploy_pos)
        n_red = len(self.red_bazhua_pos) + len(self.red_bigua_pos)

        # Extract the deployment status of the red side (skip the blue side)
        red_start = 4 * n_blue
        red_state = self.state[red_start:].reshape(-1, 3)  # [n_red, 3]

        # Iterate over the red deployment points, find the matching position and update
        for i in range(len(red_state)):
            if (
                abs(red_state[i][0] - action[0]) < 1e-8
                and abs(red_state[i][1] - action[1]) < 1e-8
            ):
                red_state[i][2] = action[4]
                break

        # Update the state
        self.state[red_start:] = red_state.flatten()

    def action_mapping(self, red_bazhua_pos, red_bigua_pos):
        self.conv_pos = {}
        index = 0

        for pos in red_bazhua_pos:
            self.conv_pos[index] = pos
            index += 1

        for pos in red_bigua_pos:
            self.conv_pos[index] = pos
            index += 1

    def create_red_units(self, episode_actions):
        red_bigua_pos = []
        red_bazhua_pos = []
        for action in episode_actions:
            if action[4] == 1:
                red_bigua_pos.append(tuple(action[:4]))
            elif action[4] == 2:
                red_bazhua_pos.append(tuple(action[:4]))

        bigua_init_msg: list[ObjectInitActionMsg] = []
        for i, (lat, lon, alt, yaw) in enumerate(red_bigua_pos):
            bigua_init_msg.append(
                ObjectInitActionMsg(
                    timestamps=time.time(),
                    objectName=f"Red_BiGua_{i}",
                    objectType=ObjectType.ObjectType_UnmannedVehicle_Red_BiGua,
                    objectCamp=0,
                    objectPos=Position3D(lon, lat, alt),
                    objectOrientation=RotationAngles(yaw),
                )
            )

        bazhua_init_msg: list[ObjectInitActionMsg] = []
        for i, (lat, lon, alt, yaw) in enumerate(red_bazhua_pos):
            bazhua_init_msg.append(
                ObjectInitActionMsg(
                    timestamps=time.time(),
                    objectName=f"Red_BaZhua_{i}",
                    objectType=ObjectType.ObjectType_UnmannedVehicle_Red_BaZhaoYu,
                    objectCamp=0,
                    objectPos=Position3D(lon, lat, alt),
                    objectOrientation=RotationAngles(yaw),
                )
            )
        self.messages.extend(bigua_init_msg)
        self.messages.extend(bazhua_init_msg)

        # Define defend area
        defend_msgs = []
        for object in bigua_init_msg:
            defend_msg = DefendAreaInfoMsg(
                timestamps=time.time(),
                objectName=object.objectName,
                defendAreaInfo=DefendAreaInfo(40, -80, -90, 90),
            )
            defend_msgs.append(defend_msg)

        for object in bazhua_init_msg:
            defend_msg = DefendAreaInfoMsg(
                timestamps=time.time(),
                objectName=object.objectName,
                defendAreaInfo=DefendAreaInfo(45, -45, -90, 90),
            )
            defend_msgs.append(defend_msg)

        self.messages.extend(defend_msgs)
