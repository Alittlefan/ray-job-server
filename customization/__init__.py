from ray.tune.registry import register_env
from target_assign_rl import TaskAllocationEnv

import customization.model.defense_c_model
import customization.model.bky_sim_model
import customization.model.defense_sim_model
import customization.model.gpp_model
import customization.model.lpp2d_model
import customization.model.lpp3d_model
from customization.callbacks import CheckpointCallback, LoadCheckpointCallback
from customization.envs.defense_env.env_creator import defense_env_feb
from customization.envs.defense_sim_v0.env_creator import defense_sim_feb
from customization.envs.defense_sim.env_creator import bky_train_sim, bky_validate_sim
from customization.progress_report import report_progress


def global_3d(env_config: dict):
    from uav_3d.env import Uav3DEnv
    from uav_3d.utils.misc import deep_update

    conf = {
        "simulator": {
            "map_size": (500, 500, 100),
            "num_buildings": 15,
            "building_min_size": 50,
            "building_max_size": 80,
            "building_min_height": 30,
            "building_max_height": 90,
            "num_dynamic_obstacles": 0,
            "min_goal_distance": 30,
            "goal_threshold": 12,
            "scenario_type": "random",
        },
        "observation": {
            "type": "color_grid",
            "params": {"dilation": False, "resolution": 10, "occupancy_threshold": 0.3},
            "action_mask": True,
        },
        "action": {"type": "default", "params": {"speed": 10.0}},
        "reward": {
            "type": "experimental",
            "params": {
                "obstacle_coefficient": -5,
                "orientation_coefficient": 0,
            },
        },
        "env": {"max_steps": 300},
    }
    deep_update(conf, env_config, True)
    return Uav3DEnv(conf)


def global_3d_sim(env_config: dict):
    from copy import deepcopy

    from uav_3d.env.comm_env import DEFAULT_CONFIG, CommEnv
    from uav_3d.utils.misc import deep_update

    conf = deepcopy(DEFAULT_CONFIG)
    conf.update(
        {
            "simulator": {
                "host": "192.168.16.105",
                "mark_goal": True,
                "auto_goal_threshold": 15,
            }
        }
    )

    deep_update(conf, env_config, True)
    return CommEnv(conf)


def local_3d(env_config: dict):
    from uav_3d.env import LocalPathPlanning
    from uav_3d.utils import deep_update

    conf = {
        "observation": {
            "type": "lpp",
            "action_mask": False,
            "params": {"full_map": False},
        },
        "action": {"type": "lpp", "params": {"full_action": True}},
        "reward": {"type": "lpp"},
        "simulator": {
            "map_size": (500, 500, 100),
            "num_buildings": 10,
            "building_min_size": 20,
            "building_max_size": 50,
            "building_min_height": 30,
            "building_max_height": 90,
            "num_dynamic_obstacles": 3,
            "num_occur_obstacles": 2,
            "min_goal_distance": 280,
            "goal_threshold": 4.0,
            "num_waypoints": 5,
            "scenario_type": "lpp",
            "dynamic_obstacles_action_type": "astar",
            "planner_type": "astar",
            "planner_config": {
                "resolution": 10,
                "safe_distance": 3,
                "occupancy_threshold": 0.3,
                "enable_optimizer": True,
                "optimize_config": {"max_speed": 1, "max_acceleration": 0.5},
            },
        },
        "env": {"max_steps": 200},
    }
    deep_update(conf, env_config, True)
    return LocalPathPlanning(conf)


def local_3d_sim(env_config: dict):
    from copy import deepcopy

    from uav_3d.env.comm_env import DEFAULT_CONFIG, CommEnv
    from uav_3d.utils.misc import deep_update

    conf = deepcopy(DEFAULT_CONFIG)
    lpp_default = {
        "simulator": {
            "host": "192.168.16.105",
            "goal_threshold": 4.0,
            "mark_goal": True,
            "auto_goal_threshold": 10,
            "planner_type": "astar",
            "planner_config": {
                "resolution": 10,
                "safe_distance": 3,
                "occupancy_threshold": 0.3,
                "enable_optimizer": True,
                "optimize_config": {"max_speed": 1, "max_acceleration": 0.5},
            },
        },
        "observation": {"type": "lpp_comm"},
        "reward": {"type": "lpp_comm"},
        "action": {"type": "lpp"},
    }
    deep_update(conf, lpp_default, True)
    deep_update(conf, env_config, True)
    return CommEnv(conf)


def local_2d(env_config: dict):
    from uav_2d import Uav2DEnv
    from uav_2d.wrappers.raster_wrapper import RasterWrapper

    conf = {
        "fixed_obstacles": 20,
        "occur_obstacles": 1,
        "occur_number_max": 3,
        "dynamic_obstacles": 20,
    }
    conf.update(env_config)
    return RasterWrapper(Uav2DEnv(conf))


register_env("Defense_C_Feb", defense_env_feb)
register_env("Defense_Sim_Feb", defense_sim_feb)
register_env("Defense_Sim", bky_train_sim)
register_env("Defense_Sim_Infer", bky_validate_sim)
register_env("LocalPathPlanning_2D", local_2d)
register_env("LocalPathPlanning_3D", local_3d)
register_env("LocalPathPlanning_Sim", local_3d_sim)
register_env("GlobalPathPlanning_3D", global_3d)
register_env("GlobalPathPlanning_Sim", global_3d_sim)
register_env("TaskAllocation", TaskAllocationEnv)
