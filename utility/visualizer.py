import numpy as np

from utility.utility import *
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, FuncAnimation


class Visualizer:
    def __init__(self, data):
        # 数据处理
        self.stroke_data_raw = np.array(data)
        stroke_data = np.transpose(np.vstack((self.stroke_data_raw[:, 0], -self.stroke_data_raw[:, 1])))  # 上下颠倒
        stroke_data = np.cumsum(stroke_data, axis=0, dtype=float).tolist()  # 按行累加
        stroke_lift = self.stroke_data_raw[:, 2].tolist()
        stroke_data_sequential = [[]]

        # 向量前两个值为移动方向，最后一个值为是否提笔
        j = 0
        for i in range(len(stroke_lift)):
            if stroke_lift[i] == 0:
                stroke_data_sequential[j].append(stroke_data[i])
            else:
                stroke_data_sequential[j].append(stroke_data[i])
                if i != len(stroke_lift) - 1:
                    stroke_data_sequential.append([])
                stroke_data_sequential[j] = np.transpose(np.array(stroke_data_sequential[j])).tolist()
                j += 1

        self.stroke_data = stroke_data
        self.stroke_lift = stroke_lift
        self.stroke_data_sequential = stroke_data_sequential

        self.fig, self.ax = plt.subplots(figsize=(3, 3), dpi=80)
        self.ln = self.ax.plot([], [], linewidth='1', color='black', linestyle='-', marker='.')

    def visualizer_stroke(self):
        """可视化笔触"""
        # 设置绘图范围
        plot_range = get_plot_range(self.stroke_data)
        self.ax.set_xlim(plot_range[0, :])
        self.ax.set_ylim(plot_range[1, :])
        # 关闭坐标轴显示
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])

        j = 0
        for data in self.stroke_data_sequential:
            for k in range(len(data[0])):
                for i in range(j):
                    self.ax.plot(self.stroke_data_sequential[i][0], self.stroke_data_sequential[i][1],
                                 linewidth='1', color='black', linestyle='-', marker='.')
                self.ax.plot(data[0][0:(k+1)], data[1][0:(k+1)],
                             linewidth='1', color='black', linestyle='-', marker='.')
                plt.pause(0.01)  # 暂停
            j += 1

        plt.show()

    def init_stroke(self):
        # 设置绘图范围
        plot_range = get_plot_range(self.stroke_data)
        self.ax.set_xlim(plot_range[0, :])
        self.ax.set_ylim(plot_range[1, :])
        # 关闭坐标轴显示
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])
        return self.ln

    def update_stroke(self, frame):
        total_num = 0
        stroke_num = 1
        step_num = 0
        for data in self.stroke_data_sequential:
            total_num += len(data[0])
            if total_num >= frame:
                stroke_num -= 1
                step_num = frame - (total_num - len(data[0]))
                break
            stroke_num += 1

        for i in range(stroke_num):
            self.ln.append(self.ax.plot(self.stroke_data_sequential[i][0], self.stroke_data_sequential[i][1],
                           linewidth='1', color='black', linestyle='-', marker='.'))
        self.ln.append(self.ax.plot(self.stroke_data_sequential[stroke_num][0][0:(step_num+1)],
                                    self.stroke_data_sequential[stroke_num][1][0:(step_num+1)],
                                    linewidth='1', color='black', linestyle='-', marker='.'))

        return self.ln

    def visualizer_stroke_gif(self):
        """可视化笔触，并生成gif动图"""
        ani = FuncAnimation(self.fig, self.update_stroke, frames=len(self.stroke_data_raw),
                            init_func=self.init_stroke, interval=50, blit=False)
        # plt.show()
        ani.save("figure/test.gif", writer='pillow', )
