import math
import random
from collections import defaultdict

from customization.envs.defense_env.envs.stone.base_stone import BaseStone


class TurretStone(BaseStone):
    def __init__(self, stats: dict):
        super().__init__(stats)
        # self._orientation = stats.get("orientation", 0)
        self._angle_start = stats.get("angle_start", 20) % 360
        self._angle_end = stats.get("angle_end", 180) % 360
        if self._angle_end > self._angle_start:
            angle_diff = self._angle_end - self._angle_start
        else:
            angle_diff = 360 + self._angle_end - self._angle_start
        self._attack_per_turn = max(
            1, stats.get("max_attack_per_turn", 0) - (angle_diff // 60)
        )

    def _is_enemy_in_range(self, enemy):
        dx = enemy.get_posx() - self._posx
        dy = enemy.get_posy() - self._posy

        if not (
            self._attack_range[0]
            <= math.sqrt((dx * dx) + (dy * dy))
            <= self._attack_range[1]
        ):
            return False
        enemy_angle = (math.degrees(math.atan2(dx, -dy)) + 360) % 360

        if self._angle_start < self._angle_end:
            return self._angle_start <= enemy_angle <= self._angle_end
        else:
            return enemy_angle >= self._angle_start or enemy_angle <= self._angle_end

    def _select_target(self, enemies):
        return min(enemies, key=lambda enemy: enemy.get_remaining_distance())

    def get_distance_from_target(self, target):
        dx = target.get_posx() - self._posx
        dy = target.get_posy() - self._posy
        return dx * dx + dy * dy


class FarTurretStone(TurretStone):
    def __init__(self, stats):
        super().__init__(stats)

        # visual config
        self._color = [(204, 0, 0, 128)]
        self._image = "image/icons8-mortar-96.png"

    def attack(self, enemies, health_record):
        attack_times = 0
        attack_successful = 0
        damage_record = defaultdict(int)
        no_under_attack = 0
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
                if 625 < self.get_distance_from_target(target) <= 1225:
                    no_under_attack += 1
            else:
                damage = 0
            damage_record[target_name] += damage
            health_record[target_name] -= damage
            self._ammunition -= 1
        return dict(damage_record), attack_times, attack_successful, no_under_attack


class NearTurretStone(TurretStone):
    def __init__(self, stats):
        super().__init__(stats)

        # visual config
        self._color = [(255, 102, 102, 128)]
        self._image = "image/icons8-mortar-96.png"
