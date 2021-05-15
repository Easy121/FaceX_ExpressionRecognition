from utility.utility import *
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


class Visualizer:
    def __init__(self, data):
        # 数据处理

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
        stroke_data_local = stroke_data_local.reshape(-1, 2)

        # sketch_data 指的是在累加后，转换为在画板上的xy轴坐标，仅用于可视化
        sketch_data = np.transpose(np.vstack((stroke_data_raw[:, 0],
                                              -stroke_data_raw[:, 1])))  # 加负号上下颠倒，用transpose恢复到一列x，一列y
        sketch_data = np.cumsum(sketch_data, axis=0, dtype=float)  # 按行累加，得到
        # 向量前两个值为移动方向，最后一个值lift为是否提笔，1说明该向量的下一个向量只移动画笔，不作图
        sketch_lift = stroke_data_raw[:, 2]
        # 以每次提笔作为分割，得到代表绘画顺序的三维列表
        sketch_data_sequential = []
        values = np.empty([0, 0])
        for count, value in enumerate(sketch_lift):
            values = np.append(values, sketch_data[count])
            if value == 1:
                values = values.reshape(-1, 2)
                sketch_data_sequential.append(values)
                values = np.empty([0, 0])

        # 将值传入self
        self.sketch_data = sketch_data
        self.sketch_lift = sketch_lift
        self.sketch_data_sequential = sketch_data_sequential
        self.stroke_data_raw = stroke_data_raw
        self.stroke_data_local = stroke_data_local

        self.ims = []  # 每一帧需要plot的artistic集合

    def xy_plotter(self):
        """绘制特征在平面上的分布"""
        fig, ax = plt.subplots(figsize=(10, 10), dpi=80)
        # ax.scatter(self.stroke_data_raw[:, 0], self.stroke_data_raw[:, 1],
        #            label='With idle movement', s=10, c='blue', marker='o')
        ax.scatter(self.stroke_data_local[:, 0], self.stroke_data_local[:, 1],
                   label='Only local stroke', s=10, c='red', marker='o')
        plt.legend(loc='lower right', fontsize=10)  # 标签位置
        plt.show()

    def visualizer_stroke(self):
        """可视化笔触"""
        fig, ax = plt.subplots(figsize=(3, 3), dpi=80)
        # 设置绘图范围
        plot_range = get_plot_range(self.sketch_data)
        ax.set_xlim(plot_range[0, :])
        ax.set_ylim(plot_range[1, :])
        # 关闭坐标轴显示
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        for count, data in enumerate(self.sketch_data_sequential):
            for k in range(data.shape[0]):
                # for i in range(count):
                #     ax.plot(self.stroke_data_sequential[i][:, 0],
                #             self.stroke_data_sequential[i][:, 1],
                #             linewidth='1', color='black', linestyle='-', marker='.')
                ax.plot(data[0:(k + 1), 0], data[0:(k + 1), 1],
                        linewidth='1', color='black', linestyle='-', marker='.')
                plt.pause(0.01)  # 暂停

        plt.show()

    def visualizer_stroke_gif(self):
        """可视化笔触，并生成gif动图"""
        fig, ax = plt.subplots(figsize=(3, 3), dpi=80)
        # 设置绘图范围
        plot_range = get_plot_range(self.sketch_data)
        ax.set_xlim(plot_range[0, :])
        ax.set_ylim(plot_range[1, :])
        # 关闭坐标轴显示
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        for count, data in enumerate(self.sketch_data_sequential):
            for k in range(data.shape[0]):
                im = []
                for i in range(count):
                    im.append(ax.plot(self.sketch_data_sequential[i][:, 0],
                                      self.sketch_data_sequential[i][:, 1],
                                      linewidth='1', color='black', linestyle='-', marker='.')[0])
                im.append(ax.plot(data[0:(k + 1), 0], data[0:(k + 1), 1],
                                  linewidth='1', color='black', linestyle='-', marker='.')[0])
                self.ims.append(im)
                plt.pause(0.01)  # 暂停

        ani = ArtistAnimation(fig, self.ims, interval=50, repeat_delay=500, repeat=True, blit=True)
        ani.save(filename="figure/test.gif", writer='pillow')
        plt.show()
