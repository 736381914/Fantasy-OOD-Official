import numpy as np
import torch
import faiss
import umap
import time
#import matplotlib.pyplot as plt
import faiss.contrib.torch_utils

import torch.nn.functional as F

# torch.Size([500, 768]), faiss, 300, 50, 0
def KNN_dis_search_decrease(target, index, K=50, select=1,shift=0):
    '''
    data_point: Queue for searching k-th points
    target: the target of the search
    K
    '''
    # Normalize the features
    target_norm = torch.norm(target, p=2, dim=1,  keepdim=True)  # torch.Size([500, 1])
    normed_target = target / target_norm  # torch.Size([500, 768])
    # start_time = time.time()
    # D, I = index.search(q, k)
    # 在索引 index 中查找距离 q 最近的 k 个向量，并返回它们的索引和距离。
    # q: 查询向量
    # k: 要返回的最近邻数量
    # D: 距离数组, I: 索引数组
    distance, output_index = index.search(normed_target, K)  # torch.Size([500, 300]), torch.Size([500, 300])
    k_th_distance = distance[:, -1]  # distance 每一行从近到远排列, 选择排在300位最远的距离, torch.Size([500])
    #k_th_output_index = output_index[:, -1]
    if shift:
        k_th_distance, minD_idx = torch.topk(-k_th_distance, select)  # torch.Size([100]), torch.Size([100])
    else:
        # 获取 k_th_distance 中最大的 50 个值及其索引。
        # values, indices = torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
        # input: 输入张量。k: 要返回的最大元素的数量。
        k_th_distance, minD_idx = torch.topk(k_th_distance, select)  # torch.Size([50]), torch.Size([50])
    #k_th_index = k_th_output_index[minD_idx]
    return minD_idx, k_th_distance


# torch.Size([75000, 768]), faiss, 300, 2, 1500, 768, 0
def KNN_dis_search_distance(target, index, K=50, num_points=10, length=2000, depth=342, shift=0):
    '''
    data_point: Queue for searching k-th points
    target: the target of the search
    K
    '''
    #Normalize the features

    target_norm = torch.norm(target, p=2, dim=1,  keepdim=True)  # torch.Size([75000, 1])
    normed_target = target / target_norm  # torch.Size([75000, 768])
    #start_time = time.time()

    # 在索引 index 中查找距离 normed_target 最近的 300 个向量，并返回它们的索引和距离。
    distance, output_index = index.search(normed_target, K)  # torch.Size([75000, 300]), torch.Size([75000, 300])
    k_th_distance = distance[:, -1]  # distance 每一行从近到远排列, 选择排在300位最远的距离, torch.Size([75000])
    k_th = k_th_distance.view(length, -1)  # 将50个边界点采样特征的距离分成50列, torch.Size([1500, 50])
    target_new = target.view(length, -1, depth)  # torch.Size([1500, 50, 768])
    # k_th_output_index = output_index[:, -1]
    if shift:
        k_th_distance, minD_idx = torch.topk(-k_th, num_points, dim=0)  # torch.Size([1, 100]), torch.Size([1, 100])
    else:
        k_th_distance, minD_idx = torch.topk(k_th, num_points, dim=0)  # torch.Size([2, 50]), torch.Size([2, 50])
    # minD_idx = minD_idx.squeeze()  # torch.Size([2, 50]) 或 torch.Size([100])
    point_list = []
    # breakpoint()
    if len(minD_idx.size()) == 1:
        minD_idx = minD_idx.reshape(-1,1)  # torch.Size([100, 1])
    for i in range(minD_idx.shape[1]):  # range(0, 50)
        # (1, 3) (1501, 1503) (3001, 3003) .... (73501, 73503)
        point_list.append(i*length + minD_idx[:, i])  # len = 50 [torch.Size([2]), torch.Size([2]), torch.Size([2]), torch.Size([2]), torch.Size([2])]
    # return torch.cat(point_list, dim=0)
    # (1, 3) + (1501, 1503) + (3001, 3003) + ... + (73501, 73503) torch.Size([100])
    return target[torch.cat(point_list)]  # 50个边界各自采样特征中2个距离最大的特征, torch.Size([100, 768])


# ID: 每个类别的归一化 ID 队列, torch.Size([500, 768])
# input_index: faiss.GpuIndexFlatL2(res, 768), 指定了向量维度
# negative_samples: 从标准多元高斯分布中随机采样的样本, torch.Size([1500, 768])
# ID_points_num: 每个选定的边界ID样本生成的合成异常值的数量, 2
# K: top-K的值来计算KNN距离, 300
# select: 需要选择多少ID样本来定义为样本空间边界附近的点, 50
# cov_mat: 在确定采样范围之前，协方差矩阵的权重, 0.07
# sampling_ratio: 采样率, 1.0
# pic_nums: 用于生成离群值的边界ID样本数, 50
# depth=768
# shift=0
def generate_outliers(ID, input_index, negative_samples, ID_points_num=2,
                      K=20, select=1, cov_mat=0.1, sampling_ratio=1.0,
                      pic_nums=30, depth=342, shift=0):
    length = negative_samples.shape[0]  # 1500
    data_norm = torch.norm(ID, p=2, dim=1, keepdim=True)  # torch.Size([500, 1])
    normed_data = ID / data_norm  # torch.Size([500, 768]), torch.Size([500, 768]) / torch.Size([500, 1])
    # numpy.random.choice(a, size=None, replace=True, p=None)
    # a：一个数组或范围, 表示可以选择的元素集合。
    # size：整数或整数的序列, 表示要随机选择的元素数量。
    # replace：布尔值, 表示是否允许重复选择元素。如果为 True(默认值), 则允许重复; 如果为 False, 则不允许重复, 且 size 不应超过 a 的长度。
    rand_ind = np.random.choice(normed_data.shape[0], int(normed_data.shape[0] * sampling_ratio), replace=False)  # (500,)
    index = input_index
    index.add(normed_data[rand_ind])  # 将归一化的 ID 队列 torch.Size([500, 768]) 存入index中
    minD_idx, k_th = KNN_dis_search_decrease(ID, index, K, select, shift=shift)  # torch.Size([50]), torch.Size([50])
    boundary_data = ID[minD_idx]  # torch.Size([50, 768])
    # breakpoint()
    minD_idx = minD_idx[np.random.choice(select, int(pic_nums), replace=False)]  # torch.Size([50])
    # 第1个边界点重复 1500 次 cat ... cat 第50个边界点重复 1500 次, torch.Size([1500, 768]) cat ... cat torch.Size([1500, 768])
    data_point_list = torch.cat([ID[i:i+1].repeat(length, 1) for i in minD_idx])  # torch.Size([75000, 768])
    # 从标准多元高斯分布中随机采样的样本重复50次 (1-1500 ... 1-1500), 然后对应加到50个边界点中，来模拟从50个边界点周围采样
    negative_sample_cov = cov_mat * negative_samples.cuda().repeat(pic_nums, 1)  # torch.Size([75000, 768])
    # 11111111111111111  22222222222222222  505050505050505050  data_point_list
    #                  +                  +
    # 123456789....1500  123456789....1500  123456789....1500   negative_sample_cov
    # 边界点1周围采样的特征  边界点2周围采样的特征  边界点100周围采样的特征
    negative_sample_list = F.normalize(negative_sample_cov + data_point_list, p=2, dim=1)  # torch.Size([75000, 768])
    # breakpoint()
    point = KNN_dis_search_distance(negative_sample_list, index, K, ID_points_num, length, depth, shift=shift)  # torch.Size([100, 768])

    index.reset()

    #return ID[minD_idx]
    return point, boundary_data  # torch.Size([100, 768]), torch.Size([50, 768])

def generate_outliers_OOD(ID, input_index, negative_samples, K=100, select=100, sampling_ratio=1.0):
    data_norm = torch.norm(ID, p=2, dim=1, keepdim=True)
    normed_data = ID / data_norm
    rand_ind = np.random.choice(normed_data.shape[1], int(normed_data.shape[1] * sampling_ratio), replace=False)
    index = input_index
    index.add(normed_data[rand_ind])
    minD_idx, k_th = KNN_dis_search_decrease(negative_samples, index, K, select)

    return negative_samples[minD_idx]



def generate_outliers_rand(ID, input_index,
                           negative_samples, ID_points_num=2, K=20, select=1,
                           cov_mat=0.1, sampling_ratio=1.0, pic_nums=10,
                           repeat_times=30, depth=342):
    length = negative_samples.shape[0]
    data_norm = torch.norm(ID, p=2, dim=1, keepdim=True)
    normed_data = ID / data_norm
    rand_ind = np.random.choice(normed_data.shape[1], int(normed_data.shape[1] * sampling_ratio), replace=False)
    index = input_index
    index.add(normed_data[rand_ind])
    minD_idx, k_th = KNN_dis_search_decrease(ID, index, K, select)
    ID_boundary = ID[minD_idx]
    negative_sample_list = []
    for i in range(repeat_times):
        select_idx = np.random.choice(select, int(pic_nums), replace=False)
        sample_list = ID_boundary[select_idx]
        mean = sample_list.mean(0)
        var = torch.cov(sample_list.T)
        var = torch.mm(negative_samples, var)
        trans_samples = mean + var
        negative_sample_list.append(trans_samples)
    negative_sample_list = torch.cat(negative_sample_list, dim=0)
    point = KNN_dis_search_distance(negative_sample_list, index, K, ID_points_num, length,depth)

    index.reset()

    #return ID[minD_idx]
    return point