from utility.visualizer import *
from utility.kmean import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import datetime


start_time = datetime.datetime.now()

# 情感类别定义
emotions_list = {1: 'angry', 2: 'disgusted', 3: 'fearful',
                 4: 'happy', 5: 'sad', 6: 'surprised'}
# 每种感情的数量
emotions_num = 100
# 聚类数量
cluster_num = 10
# 样本的真实感情分类
emotions_category = np.empty([0, 0])
emotions_samples = np.empty([0, 0])

np.random.seed(2021)
# 导入numpy压缩数据，列表类型，每个样本的特征长度不一，将allow_pickle 设为True
for i in range(1, len(emotions_list) + 1):
    file_name = 'data/Abstract_NPZ/Female_front/' + emotions_list[i] + '.npz'
    data = np.array(np.load(file_name, allow_pickle=True), dtype=np.ndarray)
    data = np.random.choice(data, emotions_num)
    for datum in data:
        # 得到去除画笔移动向量的其余向量
        datum = get_stroke_data_local(datum)
        # 加入绘画顺序信息用于聚类
        datum = add_sequence_info(datum)
        # 归一化处理
        datum = homogenize(datum, weight_direction=1, weight_sequence=3)
        # 由于样本特征长度不同，先用Kmeans进行聚类
        datum = Kmeans(cluster_num).fit(datum, visualization=None).center
        # append，除去绘画顺序
        emotions_samples = np.append(emotions_samples, datum[:, 1:])

    emotions_category = np.append(emotions_category, np.array([i] * emotions_num))
emotions_samples = emotions_samples.reshape(emotions_num*len(emotions_list), -1)

X_train, X_test, y_train, y_test = train_test_split(
    emotions_samples, emotions_category,
    test_size=0.2, random_state=2021)

model = SVC(C=1, kernel='rbf').fit(X_train, y_train)
print('train_accuracy =', model.score(X_train, y_train))



print('predict_accuracy =', model.score(X_test, y_test))

end_time = datetime.datetime.now()
print('time_taken = ', (end_time - start_time).seconds, 's')

# 可视化，需要时取用
# vis = Visualizer(emotions_samples[31])
# vis.xy_plotter()
# vis.visualizer_stroke()
# vis.visualizer_stroke_gif()


