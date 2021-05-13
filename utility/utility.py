import numpy as np
import matplotlib.pyplot as plt


def get_plot_range(data):
    """通过数据，得到方形xy轴范围"""
    # 第一列为x，第二列为y
    x_max = np.max(data[:, 0])
    x_min = np.min(data[:, 0])
    x_range = x_max - x_min
    x_mid = (x_max + x_min) / 2

    y_max = np.max(data[:, 1])
    y_min = np.min(data[:, 1])
    y_range = y_max - y_min
    y_mid = (y_max + y_min) / 2

    if x_range > y_range:
        plot_range = x_range
        y_max = y_mid + plot_range / 2
        y_min = y_mid - plot_range / 2
    else:
        plot_range = y_range
        x_max = x_mid + plot_range / 2
        x_min = x_mid - plot_range / 2

    return np.array([[x_min, x_max], [y_min, y_max]])


def stroke_visualizer(data):
    """可视化笔触"""
    # 数据处理
    stroke_data_raw = np.array(data)
    stroke_data = np.transpose(np.vstack((stroke_data_raw[:, 0], -stroke_data_raw[:, 1])))  # 上下颠倒
    stroke_data = np.array(np.cumsum(stroke_data, axis=0, dtype=float))  # 按行累加
    stroke_lift = np.array(stroke_data_raw[:, 2])

    plt.figure(figsize=(3, 3), dpi=80)
    plt.ion()  # 打开交互模式
    # 设置绘图范围
    plot_range = get_plot_range(stroke_data)
    plt.xlim(plot_range[0, :])
    plt.ylim(plot_range[1, :])
    # 关闭坐标轴显示
    plt.xticks([])
    plt.yticks([])

    # 向量前两个值为移动方向，最后一个值为是否提笔
    point_former = stroke_data[0]  # 第一个点的位置
    for i in range(1, len(stroke_data_raw)):
        if stroke_lift[i-1] == 0:
            # 此时要画
            point_current = stroke_data[i]  # 当前点的位置
            points = np.vstack((point_former, point_current))
            plt.plot(points[:, 0], points[:, 1], linewidth='1', color='black', linestyle='-', marker='.')
            point_former = point_current  # 下一个点的位置

            plt.pause(0.01)  # 暂停
        else:
            # 此时不画
            point_current = stroke_data[i]
            point_former = point_current

    plt.ioff()  # 关闭交互模式
    plt.show()


class Demo:
    def __init__(self):
        print('Demo Initiated')
