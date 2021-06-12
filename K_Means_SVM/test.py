import numpy as np
import datetime

# - 测量数据集数量
# file_name = '../data/Abstract_NPZ/Female_front/' + 'angry' + '.npz'
# data = np.array(np.load(file_name, allow_pickle=True), dtype=np.ndarray)
# print(len(data))
# # 得到19200个，每个数据集都是

# - 测试datatime
# start_time = datetime.datetime.now()
#
# vector1 = np.array([1, 1])
# vector2 = np.array([2, 2])
# print(np.linalg.norm(vector1-vector2))
#
# end_time = datetime.datetime.now()
# print('time_taken = ', (end_time - start_time).seconds)

# - 测试归一化
# a = np.matrix([[1, 2], [2, 3], [3, 4]])
# b = np.empty([0, 0])
# # for rol in a:
# #     print(rol)
# # for col in range(a.shape[1]):
# #     b = np.append(b, a[:, col]/np.max(a[:, col]))
# # print(b.reshape(-1, 2, order='F'))
# print(np.square(a))
# print(np.sqrt(np.max(np.square(a[:, 0]) + np.square(a[:, 1]))))
# print(a/np.sqrt(np.max(np.square(a[:, 0]) + np.square(a[:, 1]))))

# - 测试三维点聚类
# a = np.matrix([[1, 2], [2, 3], [3, 4]])
# print(np.arange(0, a.shape[0], 1).reshape(-1, 1))
# a = np.hstack((np.arange(0, a.shape[0], 1).reshape(-1, 1), a))
# print(a[:, 1:])
# sequence = a[:, 0]
# data = a[:, 1:]
# print(np.hstack((sequence / np.max(sequence),
#                  data / np.sqrt(np.max(np.square(data[:, 0]) + np.square(data[:, 1]))))))

emotions_samples_all = np.matrix([[1, 2, 3, 4], [2, 3, 3, 4], [3, 4, 3, 4]])
emotions_list = {1: 'angry', 2: 'disgusted', 3: 'fearful',
                 4: 'happy', 5: 'sad', 6: 'surprised'}
emotion_cluster_all = np.matrix([[10, 20, 3, 40], [2, 30, 3, 40], [30, 4, 30, 4], [30, 4, 30, 40], [30, 40, 30, 4], [3, 4, 3, 4]])

emotions_predict_all = np.empty([0, 0])
for sample in emotions_samples_all:
    distance_emotion = np.empty([0, 0])
    sample_re = sample.reshape(-1, 2)
    for i in range(1, len(emotions_list) + 1):
        distance_point = np.empty([0, 0])
        for point in sample_re:
            distance_point = \
                np.append(distance_point,
                          np.min([np.linalg.norm(point - c) for c in emotion_cluster_all[i-1].reshape(-1, 2)]))
        distance_emotion = np.append(distance_emotion, np.sum(distance_point))
    print(distance_emotion)
    emotion_predict = np.argmin(distance_emotion)
    emotions_predict_all = np.append(emotions_predict_all, emotion_predict+1)

print(emotions_predict_all)
