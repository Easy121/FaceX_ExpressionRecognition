from utility.visualizer import *


# 情感类别定义
emotions_list = {1: 'angry', 2: 'disgusted', 3: 'fearful', 4: 'happy', 5: 'sad', 6: 'surprised'}
# 每种感情的数量
emotions_num = 10
# 样本的真实感情分类
emotions_category = np.empty([0, 0])
emotions_samples = np.empty([0, 0])

np.random.seed(2021)
# 导入numpy压缩数据，列表类型，每个样本的特征长度不一，将allow_pickle 设为True
for i in range(1, len(emotions_list) + 1):
    file_name = 'data/Abstract_NPZ/Female_front/' + emotions_list[i] + '.npz'
    data = np.array(np.load(file_name, allow_pickle=True), dtype=np.ndarray)

    emotions_samples = np.append(emotions_samples, np.random.choice(data, emotions_num))
    emotions_category = np.append(emotions_category, np.array([i] * emotions_num))

# 可视化，需要时取用
# vis = Visualizer(emotions_samples[31])
# vis.xy_plotter()
# vis.visualizer_stroke()
# vis.visualizer_stroke_gif()
