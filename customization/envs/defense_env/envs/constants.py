from enum import Enum

DIRECTION_DELTAS = {
    1: (0, -1),  # 上
    2: (1, -1),  # 右上
    3: (1, 0),  # 右
    4: (1, 1),  # 右下
    5: (0, 1),  # 下
    6: (-1, 1),  # 左下
    7: (-1, 0),  # 左
    8: (-1, -1),  # 左上
}


class STONE_STATE(Enum):
    DEAD = 0  # 死亡
    MOBILE = 1  # 移动
    IMMOBILE = 2  # 不可移动
    EVACUATED = 3  # 撤退
