import os

from customization.envs.defense_env import BattleEnv

root = os.path.dirname(__file__)


def defense_env_feb(config):

    conf = {
        "num_soldier": 55,
        "num_vehicle": 5,
        "unit_num_per_step": 15,
        "minimum_unit_evacuated": 1,
        "num_far_turret": 3,
        "num_near_turret": 4,
        "near_turret_health": 200,
        "near_turret_attack": 50,
        "near_turret_angle": 270,
        "near_turret_attack_range": [0, 25],
        "near_turret_accuracy": 0.6,
        "near_turret_ammunition": 100,
        "far_turret_health": 200,
        "far_turret_attack": 50,
        "far_turret_angle": 360,
        "far_turret_attack_range": [10, 35],
        "far_turret_accuracy": 0.5,
        "far_turret_ammunition": 100,
        "soldier_health": 100,
        "soldier_mobility": 1,
        "soldier_attack": 20,
        "soldier_attack_range": [0, 25],
        "soldier_accuracy": 0.5,
        "soldier_ammunition": 200,
        "vehicle_health": 150,
        "vehicle_mobility": 2,
        "vehicle_attack": 30,
        "vehicle_attack_range": [0, 25],
        "vehicle_accuracy": 0.5,
        "vehicle_ammunition": 200,
        "state_space": "one_dim_grid",
        "action_space": "circle_decision",
        "map_path": os.path.join(root, "envs/resource/map.csv"),
        "blue_routes_path": os.path.join(root, "envs/resource/grounding_path.json"),
        "max_steps": 500,
    }
    conf.update(config)
    env = BattleEnv(conf)
    return env
