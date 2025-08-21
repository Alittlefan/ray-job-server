import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
vmax_dict = {"deployed": 900, "firepower": 6300, "route": 40000}


def visualize_heatmap(csv_file, vmax, output_file=None):
    # 读取CSV文件
    data = pd.read_csv(csv_file, header=None)  # 读取没有header的CSV数据

    # 创建热力图
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        data,
        cmap="YlGnBu",
        annot=False,
        cbar=True,
        cbar_kws={"label": "Intensity"},
        vmin=0,
        vmax=vmax,
    )
    ax.set_facecolor("white")  # 设置背景色为白色

    # 保存或显示热力图
    if output_file:
        plt.savefig(output_file, facecolor="white")  # 保存热力图并设置背景色为白色
        print(f"热力图已保存到 {output_file}")
    else:
        plt.show()  # 显示热力图


def visualize_multiple_csv(save_dir):
    csv_files = glob.glob(
        f"{save_dir}/simulation_results/*.csv"
    )  # 获取文件夹下所有CSV文件
    heatmap_dir = os.path.join(save_dir, "heatmap")
    if not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir)
    for csv_file in csv_files:  # 遍历每个CSV文件
        if csv_file.endswith(".csv"):  # 确保是CSV文件
            for key in vmax_dict:  # 遍历每个需要的类型
                if key in csv_file:  # 如果文件名包含该类型
                    # 获取文件名（不含路径和扩展名）
                    base_filename = os.path.basename(csv_file).replace(
                        ".csv", "_heatmap.png"
                    )
                    # 生成保存到PNG文件夹中的文件路径
                    output_file = os.path.join(heatmap_dir, base_filename)
                    visualize_heatmap(
                        csv_file, vmax_dict[key], output_file
                    )  # 生成热力图


def save_data(
    env,
    round_number,
    style,
    firepower_coverage,
    blue_deployed_nums,
    red_deploy_count,
    route_record_in_map,
    reward_data,
    dir,
):
    """
    保存火力覆盖数据、蓝方出生点人数和红方部署点数据到 CSV
    文件命名格式为：`轮次_风格`
    """
    # 创建保存目录
    if not os.path.exists(f"{dir}/simulation_results"):
        os.makedirs(f"{dir}/simulation_results")

    # 定义文件名格式
    base_filename = f"{round_number}_风格_{style}"

    # 1. 保存火力覆盖数据到 CSV
    firepower_coverage_filename = os.path.join(
        f"{dir}/simulation_results", f"{base_filename}_firepower_coverage.csv"
    )
    np.savetxt(firepower_coverage_filename, firepower_coverage, delimiter=",", fmt="%d")
    print(f"火力覆盖数据已保存到 {firepower_coverage_filename}")

    # 2. 保存蓝方出生点数据到 CSV
    deployed_nums_array = np.zeros_like(
        env._simulator.firepower_coverage
    )  # 初始化一个与地图大小相同的二维数组
    for (x, y), value in blue_deployed_nums.items():
        deployed_nums_array[y, x] = value  # 将坐标对应位置赋值

    # 3. 保存红方部署点数据到 CSV
    for (x, y), value in red_deploy_count.items():
        deployed_nums_array[y, x] = value  # 将坐标对应位置赋值
    deployed_nums_filename = os.path.join(
        f"{dir}/simulation_results", f"{base_filename}deployed_nums.csv"
    )
    np.savetxt(deployed_nums_filename, deployed_nums_array, delimiter=",", fmt="%d")
    print(f"出生点和部署点数据已保存到 {deployed_nums_filename}")

    # 4. 保存路线记录数据到 CSV
    route_record_in_map_filename = os.path.join(
        f"{dir}/simulation_results", f"{base_filename}_route_record_in_map.csv"
    )
    np.savetxt(
        route_record_in_map_filename, route_record_in_map, delimiter=",", fmt="%d"
    )
    print(f"路线记录数据已保存到 {route_record_in_map_filename}")

    # 5. 保存奖励数据到 CSV
    reward_data_filename = os.path.join(
        f"{dir}/simulation_results", f"{base_filename}_reward_data.csv"
    )
    np.savetxt(reward_data_filename, reward_data, delimiter=",", fmt="%d")
    print(f"奖励数据已保存到 {reward_data_filename}")


def path_to_coords(start, path):
    """
    根据起始点和路径方向列表，返回路径经过的所有坐标列表。
    """
    start_x, start_y = start  # 起始点坐标
    coords = [(start_x, start_y)]  # 初始化坐标列表，以起始点作为第一个坐标
    current_x, current_y = start_x, start_y  # 当前坐标初始化为起始点

    # 遍历路径中的方向，将方向转换为坐标变化
    for direction in path:
        if direction == 1:  # 上
            current_y -= 1
        elif direction == 3:  # 右
            current_x += 1
        elif direction == 5:  # 下
            current_y += 1
        elif direction == 7:  # 左
            current_x -= 1
        coords.append((current_x, current_y))  # 将新的坐标添加到列表中
    return coords
