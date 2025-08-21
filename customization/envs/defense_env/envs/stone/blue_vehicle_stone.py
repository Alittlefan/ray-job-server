import math
import random
from collections import defaultdict

from customization.envs.defense_env.envs.constants import STONE_STATE
from customization.envs.defense_env.envs.stone.base_stone import BaseStone


class VehicleStone(BaseStone):
    def __init__(self, stats: dict):
        super().__init__(stats)
        self._attack_per_turn = 1
        self._attack_prob = stats.get("attack_prob", 1)
        # visual config
        self._image = "image/icons8-soldier-64.png"
        self._color = [
            (204, 153, 255, 128),
            (178, 102, 255, 128),
            (153, 51, 255, 128),
            (128, 0, 255, 128),
            (102, 0, 204, 128),
            (76, 0, 153, 128),
            (51, 0, 102, 128),
            (25, 0, 51, 128),
            (0, 0, 255, 128),
            (0, 0, 153, 128),
        ]

    def _is_enemy_in_range(self, enemy):
        dx = enemy.get_posx() - self._posx
        dy = enemy.get_posy() - self._posy
        return (
            self._attack_range[0]
            <= math.sqrt((dx * dx) + (dy * dy))
            <= self._attack_range[1]
        )

    def _select_target(self, enemies):
        return min(
            enemies,
            key=lambda enemy: math.sqrt(
                (enemy.get_posx() - self._posx) ** 2
                + (enemy.get_posy() - self._posy) ** 2
            ),
        )

    def get_remaining_distance(self):
        return len(self._route)

    def attack(self, enemies, health_record):
        if random.uniform(0, 1) > self._attack_prob:
            return {}, 0, 0
        attack_times = 0
        attack_successful = 0
        damage_record = defaultdict(int)
        for _ in range(self._attack_per_turn):
            if not self._can_attack():
                break
            enemies = self._get_attackable_enemies(enemies, health_record)
            if not enemies:
                break
            target = self._select_target(enemies)
            if target is None:
                break
            target_name = target.get_name()
            attack_times += 1
            if random.uniform(0, 1) < self._accuracy:
                damage = self._attack_power
                attack_successful += 1
            else:
                damage = 0
            damage_record[target_name] += damage
            health_record[target_name] -= damage
            self._ammunition -= 1
            self._is_attacked = True
        return dict(damage_record), attack_times, attack_successful

    def move(self):
        """
        蓝方车辆每部移动两个单位
        且不受攻击影响
        """
        if self._state == STONE_STATE.IMMOBILE:
            return
        # Step 1: Move one unit if possible
        if len(self._route) > 0:
            self._move(self._route.pop(0))
        # Step 2: Move a second unit if possible
        if len(self._route) > 0:
            self._move(self._route.pop(0))
        if len(self._route) > 0:
            self._move(self._route.pop(0))
        # Update state
        if len(self._route) == 0:
            self._state = STONE_STATE.EVACUATED
            return
