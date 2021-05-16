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


def get_stroke_data_local(data):
    # stroke_data 指的是表示画笔移动的原始数据，用于机器学习
    stroke_data_raw = np.array(data, dtype=np.ndarray)
    # local 是去除画笔移动向量的其余向量，同时也忽略第一个向量
    stroke_data_local = np.empty([0, 0])
    flag = 0
    for item in stroke_data_raw[1:, :]:
        if flag == 0:
            stroke_data_local = np.append(stroke_data_local, item[:2])
        else:
            flag = 0
        if item[2] == 1:
            flag = 1
    return stroke_data_local.reshape(-1, 2)
