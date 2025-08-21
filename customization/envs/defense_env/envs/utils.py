import random

import numpy as np

x_len = 8
y_len = 8
image_x = 15
image_y = 15


def unit_move(x, y, direct):
    if direct == 1:
        y = y - 1
    elif direct == 2:
        x = x + 1
        y = y - 1
    elif direct == 3:
        x = x + 1
    elif direct == 4:
        x = x + 1
        y = y + 1
    elif direct == 5:
        y = y + 1
    elif direct == 6:
        x = x - 1
        y = y + 1
    elif direct == 7:
        x = x - 1
    elif direct == 8:
        x = x - 1
        y = y - 1
    return (x, y)


def get_move_sequence(x, y, x2, y2):
    directions = []
    path = []

    while (x, y) != (x2, y2):
        if random.random() < 0.5:
            if x < x2 and y < y2:
                direction = 4  # Move diagonally down-right
            elif x < x2 and y > y2:
                direction = 2  # Move diagonally up-right
            elif x > x2 and y < y2:
                direction = 6  # Move diagonally down-left
            elif x > x2 and y > y2:
                direction = 8  # Move diagonally up-left
            elif x < x2:
                direction = 3  # Move right
            elif y > y2:
                direction = 1  # Move up
            elif x > x2:
                direction = 7  # Move left
            elif y < y2:
                direction = 5  # Move down
        else:
            if x > x2 and y > y2:
                direction = 8  # Move diagonally up-left
            elif x < x2 and y < y2:
                direction = 4  # Move diagonally down-right
            elif x < x2 and y > y2:
                direction = 2  # Move diagonally up-right
            elif x > x2 and y < y2:
                direction = 6  # Move diagonally down-left
            elif x > x2:
                direction = 7  # Move left
            elif y < y2:
                direction = 5  # Move down
            elif x < x2:
                direction = 3  # Move right
            elif y > y2:
                direction = 1  # Move up
        directions.append(direction)
        path.append((x, y))
        x, y = unit_move(x, y, direction)
    assert x == x2 and y == y2, "src:{}, dst:{}\npath:{}".format((x, y), (x2, y2), path)
    return directions, path


def get_hex_map_pos(x, y):
    if y % 2 == 0:
        base_x = x_len * 2 * x
        base_y = y_len * 3 * (y // 2)
    else:
        base_x = x_len * 2 * x + x_len
        base_y = y_len * 3 * (y // 2) + y_len // 2 + y_len
    return base_x, base_y


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])


def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def mix_colors(colors):
    """
    混合多个颜色，根据各自的数量进行加权平均。
    colors: [(color, count), ...]
    color: (R, G, B)
    count: 该颜色单位的数量
    """
    total_count = sum(count for _, count in colors)
    if total_count == 0:
        return (0, 0, 0)

    mixed_color = [0, 0, 0]

    for color, count in colors:
        for i in range(3):  # R, G, B
            mixed_color[i] += color[i] * count

    mixed_color = [int(c / total_count) for c in mixed_color]

    return tuple(mixed_color)


def cluster_coordinates(coordinates, eps, min_samples):
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
