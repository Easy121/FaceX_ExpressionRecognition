import numpy as np

from utility.utility import *
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, FuncAnimation, PillowWriter


class Visualizer:
    def __init__(self, data):
        # 数据处理
        self.stroke_data_raw = np.array(data)
        stroke_data = np.transpose(np.vstack((self.stroke_data_raw[:, 0], -self.stroke_data_raw[:, 1])))  # 上下颠倒
        stroke_data = np.cumsum(stroke_data, axis=0, dtype=float).tolist()  # 按行累加
        stroke_lift = self.stroke_data_raw[:, 2].tolist()
        stroke_data_sequential = []

        # 向量前两个值为移动方向，最后一个值为是否提笔
        values = []
        for count, value in enumerate(stroke_lift):
            values.append(stroke_data[count])
            if value == 1:
                stroke_data_sequential.append(np.transpose(np.array(values)).tolist())
                values = []

        self.stroke_data = stroke_data
        self.stroke_lift = stroke_lift
        self.stroke_data_sequential = stroke_data_sequential

        self.fig, self.ax = plt.subplots(figsize=(3, 3), dpi=80)
        self.ln = self.ax.plot([], [], linewidth='1', color='black', linestyle='-', marker='.')
        self.ims = []  # 每一帧需要plot的artistic集合

    def visualizer_stroke(self):
        """可视化笔触"""
        # 设置绘图范围
        plot_range = get_plot_range(self.stroke_data)
        self.ax.set_xlim(plot_range[0, :])
        self.ax.set_ylim(plot_range[1, :])
        # 关闭坐标轴显示
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])

        for count, data in enumerate(self.stroke_data_sequential):
            for k in range(len(data[0])):
                im = []
                for i in range(count):
                    self.ax.plot(self.stroke_data_sequential[i][0],
                                 self.stroke_data_sequential[i][1],
                                 linewidth='1', color='black', linestyle='-', marker='.')
                self.ax.plot(data[0][0:(k + 1)], data[1][0:(k + 1)],
                             linewidth='1', color='black', linestyle='-', marker='.')
                plt.pause(0.01)  # 暂停

        plt.show()

    def visualizer_stroke_gif(self):
        """可视化笔触，并生成gif动图"""
        # 设置绘图范围
        plot_range = get_plot_range(self.stroke_data)
        self.ax.set_xlim(plot_range[0, :])
        self.ax.set_ylim(plot_range[1, :])
        # 关闭坐标轴显示
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])

        for count, data in enumerate(self.stroke_data_sequential):
            for k in range(len(data[0])):
                im = []
                for i in range(count):
                    im.append(self.ax.plot(self.stroke_data_sequential[i][0],
                                           self.stroke_data_sequential[i][1],
                                           linewidth='1', color='black', linestyle='-', marker='.')[0])
                im.append(self.ax.plot(data[0][0:(k + 1)], data[1][0:(k + 1)],
                                       linewidth='1', color='black', linestyle='-', marker='.')[0])
                self.ims.append(im)
                plt.pause(0.01)  # 暂停

        ani = ArtistAnimation(self.fig, self.ims, interval=40, repeat=True, blit=True)
        ani.save(filename="figure/test.gif", writer='pillow')
        plt.show()
