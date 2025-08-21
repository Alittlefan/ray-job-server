class Weapon:
    def __init__(self, weapon_type, attack_frequency, attack_range, damage, ammo):
        """
        初始化武器对象
        :param weapon_type: 武器类型（如"machine_gun"、"rocket"）
        :param attack_frequency: 攻击频率（次/秒）
        :param attack_range: 攻击范围（单位距离）
        :param damage: 伤害值
        :param ammo: 弹药数量
        """
        self.weapon_type = weapon_type
        self.attack_frequency = attack_frequency
        self.attack_range = attack_range
        self.damage = damage
        self.ammo = ammo
        self.cooldown_timer = 0  # 冷却计时器

    def is_ready_to_fire(self):
        """
        判断武器是否准备好发射
        :return: 布尔值
        """
        return self.cooldown_timer <= 0 and self.ammo > 0

    def fire(self, target):
        """
        发射武器攻击目标
        :param target: 目标对象，具有 `take_damage(damage)` 方法
        """
        if self.is_ready_to_fire():
            print(f"使用 {self.weapon_type} 攻击目标 {target.id}！")
            target.take_damage(self.damage)
            self.ammo -= 1
            self.cooldown_timer = 1 / self.attack_frequency  # 重置冷却计时器
        else:
            if self.ammo <= 0:
                print(f"武器 {self.weapon_type} 弹药耗尽，无法攻击！")
            else:
                print(f"武器 {self.weapon_type} 正在冷却，无法攻击！")

    def update_cooldown(self, delta_time):
        """
        更新冷却计时器
        :param delta_time: 时间间隔（秒）
        """
        if self.cooldown_timer > 0:
            self.cooldown_timer -= delta_time

    def __repr__(self):
        return (f"武器类型: {self.weapon_type}, 攻击频率: {self.attack_frequency}次/秒, "
                f"攻击范围: {self.attack_range}, 伤害: {self.damage}, 弹药: {self.ammo}")
