from utility.utility import *


# 导入numpy压缩数据，列表类型
data = np.load('data/happy.npz', allow_pickle=True)
stroke_visualizer(data[500])


