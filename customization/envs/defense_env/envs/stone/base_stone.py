import random
from abc import ABC, abstractmethod
from collections import defaultdict

from customization.envs.defense_env.envs.constants import DIRECTION_DELTAS, STONE_STATE


class BaseStone(ABC):
    def __init__(self, stats: dict):
        self._name = stats.get("name", None)
        self._posx = stats.get("posx", 0)
        self._posy = stats.get("posy", 0)
        self._health = stats.get("health", 0)
        self._attack_power = stats.get("attack_power", 0)
        self._attack_range = stats.get("attack_range", [0, 0])
        self._accuracy = stats.get("accuracy", 0)
        self._ammunition = stats.get("ammunition", 0)
        self._mobility = stats.get("mobility", 0)
        self._state = STONE_STATE.MOBILE if self._mobility > 0 else STONE_STATE.IMMOBILE
        self._route = stats.get("route", [])

        # need to be implemented in child class
        self._attack_per_turn = None

        self._is_attacked = False

    def get_name(self):
        return self._name

    def get_health(self):
        return self._health

    def get_posx(self):
        return self._posx

    def get_posy(self):
        return self._posy

    def get_states(self):
        return self._state

    def _get_attackable_enemies(self, enemies, health_record):
        enemies = [enemy for enemy in enemies if self._is_enemy_in_range(enemy)]
        enemies = [
            enemy
            for enemy in enemies
            if enemy.get_states() != STONE_STATE.DEAD
            and enemy.get_states() != STONE_STATE.EVACUATED
            and health_record[enemy.get_name()] > 0
        ]
        return enemies

    @abstractmethod
    def _is_enemy_in_range(self, enemy):
        pass

    @abstractmethod
    def _select_target(self, enemies):
        pass

    def _can_attack(self):
        if self._ammunition == 0:
            return False
        return True

    def attack(self, enemies, health_record):
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

    def set_damage(self, damage):
        self._health -= damage
        if self._health <= 0:
            self._state = STONE_STATE.DEAD

    def _move(self, direction):
        self._posx += DIRECTION_DELTAS[direction][0]
        self._posy += DIRECTION_DELTAS[direction][1]

    def move(self):
        if self._state == STONE_STATE.IMMOBILE:
            return
        self._move(self._route.pop(0))
        self._is_attacked = False
        if len(self._route) == 0:
            self._state = STONE_STATE.EVACUATED
            return
