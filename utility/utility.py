import numpy as np


def get_plot_range(data):
    """通过数据，得到方形xy轴范围"""
    data = np.array(data, dtype=np.ndarray)
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


class Demo:
    def __init__(self):
        print('Demo Initiated')
