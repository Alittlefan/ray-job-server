import numpy as np
from typing import Tuple, List


def calculate_distance(
    pos1: Tuple[float, ...], pos2: Tuple[float, ...], dimensions: int = 2
) -> float:
    """计算两点之间的欧氏距离
    Args:
        pos1: 第一个点的坐标元组
        pos2: 第二个点的坐标元组
        dimensions: 计算距离使用的维度数，默认使用前2维(x,y)
    Returns:
        float: 两点之间的距离
    """
    return np.sqrt(sum((pos1[i] - pos2[i]) ** 2 for i in range(dimensions)))


def calculate_angle(pos1: Tuple[float, ...], pos2: Tuple[float, ...]) -> float:
    """计算两点之间的角度
    Args:
        pos1: 起始点坐标元组
        pos2: 终点坐标元组
    Returns:
        float: 角度(0-360度)
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return (angle + 360) % 360


def calculate_angle_diff(angle1: float, angle2: float) -> float:
    """计算两个角度之间的最小差值
    Args:
        angle1: 第一个角度(0-360度)
        angle2: 第二个角度(0-360度)
    Returns:
        float: 最小角度差(0-180度)
    """
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)


def calculate_centroid(
    positions: List[Tuple[float, ...]], weights: List[float] = None
) -> Tuple[float, float]:
    """计算一组点的质心
    Args:
        positions: 位置坐标列表
        weights: 权重列表，默认为None(等权重)
    Returns:
        Tuple[float, float]: 质心坐标
    """
    if not positions:
        return (0.0, 0.0)

    if weights is None:
        weights = [1.0] * len(positions)

    total_weight = sum(weights)
    weighted_x = sum(pos[0] * w for pos, w in zip(positions, weights))
    weighted_y = sum(pos[1] * w for pos, w in zip(positions, weights))

    return (weighted_x / total_weight, weighted_y / total_weight)


def calculate_area_coverage(positions: List[Tuple[float, ...]]) -> float:
    """计算一组点的覆盖面积
    Args:
        positions: 位置坐标列表
    Returns:
        float: 覆盖区域面积
    """
    if not positions:
        return 0.0

    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)

    return width * height


def normalize_scores(scores: List[float], target_sum: int = 10) -> List[int]:
    """将浮点数得分归一化为整数，并确保总和为指定值
    Args:
        scores: 原始得分列表
        target_sum: 目标总和，默认为10
    Returns:
        List[int]: 归一化后的整数得分列表
    """
    if not scores:
        return []

    total = sum(scores)
    if total == 0:
        return [target_sum // len(scores)] * len(scores)

    # 归一化并转换为整数
    normalized = [int(score / total * target_sum) for score in scores]

    # 处理舍入误差
    current_sum = sum(normalized)
    remainder = target_sum - current_sum

    if remainder != 0:
        # 根据原始分数的小数部分大小分配余数
        decimal_parts = [
            (score / total * target_sum) - int(score / total * target_sum)
            for score in scores
        ]
        sorted_indices = sorted(
            range(len(decimal_parts)), key=lambda k: decimal_parts[k], reverse=True
        )

        for i in range(abs(remainder)):
            idx = sorted_indices[i]
            normalized[idx] += 1 if remainder > 0 else -1

    return normalized


def is_point_visible(pos1: Tuple[float, ...], pos2: Tuple[float, ...]) -> bool:
    """判断两点之间是否可视
    Args:
        pos1: 第一个点的坐标
        pos2: 第二个点的坐标
    Returns:
        bool: 是否可视
    """
    return True  # TODO: 添加基于通视的判断


def calculate_main_direction(
    pos: Tuple[float, ...], route: List[Tuple[float, ...]]
) -> float:
    """计算位置到路线的主要方向
    Args:
        pos: 位置坐标
        route: 路线点列表
    Returns:
        float: 主要方向角度
    """
    mid_point = route[len(route) // 2]
    return calculate_angle(pos, mid_point)


def calculate_direction_score(
    yaw: float, pos: Tuple[float, ...], blue_routes: dict
) -> float:
    """计算炮塔朝向与进攻路线的匹配度
    Args:
        yaw: 炮塔朝向角度
        pos: 炮塔位置
        blue_routes: 蓝方进攻路线字典
    Returns:
        float: 匹配度得分(0-1)
    """
    score = 0
    for route in blue_routes.values():
        main_direction = calculate_main_direction(pos, route)
        diff = calculate_angle_diff(yaw, main_direction)
        score += max(0, 1 - diff / 180)
    return score / len(blue_routes)


# def calculate_position_score(pos: Tuple[float, ...], pos_list: List[Tuple[float, ...]], coverage_dict: dict, obs: np.ndarray = None) -> Tuple[float, List[int]]:
#     """计算位置得分
#     Args:
#         pos: 待评估的位置
#         pos_list: 所有可选位置列表
#         coverage_dict: 覆盖范围字典
#         obs: 观测空间数据
#     Returns:
#         Tuple[float, List[int]]: (得分, 覆盖的路线列表)
#     """
#     # 获取该位置能覆盖的路线
#     coverage = coverage_dict.get(pos, [])

#     if obs is None:
#         # 如果没有观测数据，使用原有逻辑
#         return len(coverage) / len(config["blue_deploy_pos"]), coverage

#     # 计算实际部署的蓝方位置数量
#     deployed_positions = 0
#     covered_positions = 0

#     # 遍历所有蓝方位置
#     for i in range(6):  # 6个蓝方位置
#         base_idx = 1 + i * 4  # 跳过地图ID
#         men = obs[base_idx + 2]  # 人数
#         vehicles = obs[base_idx + 3]  # 车辆数

#         # 如果该位置有部署
#         if men > 0 or vehicles > 0:
#             deployed_positions += 1
#             # 如果该位置被覆盖
#             if i + 1 in coverage:  # coverage中的编号从1开始
#                 covered_positions += 1

#     # 如果没有部署，返回默认分数
#     if deployed_positions == 0:
#         return 0.5, coverage

#     # 计算实际覆盖率
#     coverage_ratio = covered_positions / deployed_positions

#     return coverage_ratio, coverage


def calculate_target_coverage_score(
    pos: Tuple[float, ...],
    target_pos: Tuple[float, ...],
    optimal_distance: float = 0.01,
) -> float:
    """计算位置对目标区域的覆盖得分
    Args:
        pos: 炮塔位置
        target_pos: 目标位置
        optimal_distance: 最佳距离
    Returns:
        float: 覆盖得分(0-1)
    """
    distance = calculate_distance(pos, target_pos)

    # 距离得分：距离越接近最佳距离，得分越高
    distance_score = max(0, 1 - abs(distance - optimal_distance) / optimal_distance)

    # 高度优势得分：高度差越大，得分越高
    height_advantage = min(1.0, max(0, (pos[2] - target_pos[2]) / 320.0))

    return distance_score * 0.7 + height_advantage * 0.3
