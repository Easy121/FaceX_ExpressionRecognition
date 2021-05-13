from utility.utility import *
from utility.visualizer import *


# 导入numpy压缩数据，列表类型
data = np.load('data/Abstract_NPZ/Female_front/happy.npz', allow_pickle=True)
# 可视化
vis = Visualizer(data[500])
vis.visualizer_stroke()
# vis.visualizer_stroke_gif()
