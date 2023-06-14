# from sklearn.neighbors import KNeighborsRegressor
# import torch 
# def knn_search(xyz,point, k=3):
#     """
#     xyz: B x N x 3 的点云张量
#     point: B x 1024 的采样点张量
#     k: 邻居个数，默认为3
#     """
#     B, N, _ = xyz.shape
#     _, K = point.shape
#     indices = []
#     for i in range(B):
#         # 将xyz的每个点作为训练数据，point作为目标值
#         regressor = KNeighborsRegressor(n_neighbors=k, algorithm='kd_tree')
#         regressor.fit(xyz[i],)
#         # 使用训练好的模型查找每个采样点的k个邻居
#         dists, nn_indices = regressor.kneighbors(point[i], n_neighbors=k)
#         print(dists, nn_indices)
#         indices.append(nn_indices)
#     return indices

# xyz = torch.randn(2,10,3)
# xyz1 = torch.ones(2,10,1)
# point = torch.Tensor([[1,3,5],[2,4,6]])
# knn_search(xyz,point)
# from sklearn.neighbors import KNeighborsRegressor
# import numpy as np
# import torch
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # 构造假数据
# point_cloud = torch.randn(100, 3).to(device)  # 100个点云，每个点有3个坐标
# sample_points = torch.randn(10, 3).to(device)  # 50个采样点，每个点有3个坐标

# point_cloud = point_cloud.cpu()
# sample_points = sample_points.cpu()
# # 创建KNeighborsRegressor对象，设置K值为5
# knn = KNeighborsRegressor(n_neighbors=5)

# # 将点云中的每个点作为训练数据，采样点作为目标值
# knn.fit(point_cloud, point_cloud)

# # 查找每个采样点的5个最近邻居
# distances, indices = knn.kneighbors(sample_points, n_neighbors=5)

# # 打印结果
# # for i in range(len(sample_points)):
# #     print('Sample point {}:'.format(i))
# #     for j in range(5):
# #         print('  Nearest neighbor {}: distance={}, index={}'.format(j, distances[i][j], indices[i][j]))
# print(distances, indices)
# print(type(distances), type(indices))

import torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor,NearestNeighbors
import numpy as np
import torch

# def _T(t, mode=False):
#     if mode:
#         return t.transpose(0, 1).contiguous()
#     else:
#         return t


class KNN(nn.Module):

    def __init__(self, k, transpose_mode=False):
        super(KNN, self).__init__()
        self.k = k
        self._t = transpose_mode

    def forward(self, ref, query):  #B N 3  B 1024 3
        assert ref.size(0) == query.size(0), "ref.shape={} != query.shape={}".format(ref.shape, query.shape)
        with torch.no_grad():
            batch_size = ref.size(0)
            D, I = [], []
            for bi in range(batch_size):
                point_cloud = ref[bi]
                sample_points = query[bi]
                point_cloud = point_cloud.detach().cpu()
                sample_points = sample_points.detach().cpu()
                knn = KNeighborsRegressor(n_neighbors=5)
                knn.fit(point_cloud.float(), point_cloud.float())
                distances, indices = knn.kneighbors(sample_points, n_neighbors=self.k)

                # r, q = _T(ref[bi], self._t), _T(query[bi], self._t)   #3 N  3 1024
                # d, i = knn(r.float(), q.float(), self.k)
                # d, i = _T(d, self._t), _T(i, self._t)   #N 128  1024 128
                D.append(distances)
                I.append(indices)
            D = torch.from_numpy(np.array(D))
            I = torch.from_numpy(np.array(I))
        return D, I

def fill_missing_values(x_data,x_label,y_data, k=1):
    # 创建最近邻居模型
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x_data)

    # 找到每个点的最近邻居
    distances, indices = nn.kneighbors(y_data)
    # print(distances.shape)
    # print(indices.shape)
    avg_values = np.mean(x_label[indices], axis=1)
    # print("avg_values.shape",avg_values.shape)
    return avg_values