import faiss.contrib.torch_utils
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import grad
from scipy.special import iv  # iv 表示的是 第一类修正贝塞尔函数


def initialize_states(centroids, num_neighbor=5, n_cls=10):  # torch.Size([10, 128]), 4, 10
    # torch.Size([10, 128]), torch.Size([128, 10])
    cos_sim = torch.mm(centroids, centroids.T)  # torch.Size([10, 10])
    cos_sim.fill_diagonal_(-1)  # 填充其对角线上的元素为 -1
    # 每个质心寻找前 4 个相似度最大的元素及其对应的索引
    _, near_indices = torch.topk(cos_sim, num_neighbor, dim=1, largest=True)  # torch.Size([10, 4]), torch.Size([10, 4])
    # i 类  当前类索引  最近邻质心索引
    #    0      0           9
    # 0  1      0           1
    #    2      0           8
    #    3      0           7
    # -------------------------------
    #    4      1           9
    # 1  5      1           7
    #    6      1           8
    #    7      1           0
    # ...
    # 构造 当前类 - 近邻质心 表, 一共有 40 个 pair
    pairs = torch.zeros(n_cls * num_neighbor, 2, dtype=torch.long)  # torch.Size([40, 2])
    for i in range(n_cls):
        # span = 4
        pairs[i * num_neighbor: (i + 1) * num_neighbor, 0] = i
        pairs[i * num_neighbor: (i + 1) * num_neighbor, 1] = near_indices[i]
    # torch.Size([10, 128]), torch.Size([40])        torch.Size([10, 128]), torch.Size([40])
    # torch.Size([40, 128])                          torch.Size([40, 128])
    # 按照 pairs[:, 0] 的索引顺序依次取出质心, 按照 pairs[:, 1] 的索引顺序依次取出相应的最近邻类的质心
    # (0 质心 + 9 质心) / 2
    # (0 质心 + 1 质心) / 2
    # (0 质心 + 8 质心) / 2
    # (0 质心 + 7 质心) / 2
    # (1 质心 + 9 质心) / 2
    # (1 质心 + 7 质心) / 2
    # (1 质心 + 8 质心) / 2
    # (1 质心 + 0 质心) / 2
    midpoints = (centroids[pairs[:, 0]] + centroids[pairs[:, 1]]) * 0.5  # torch.Size([40, 128])
    midpoints = F.normalize(midpoints, dim=1)  # torch.Size([40, 128])
    # 构造一个 类别对之间的二元关系掩码（mask），用于表示哪些类别在一个 pair 中
    # 每列对应一个 (当前类, 近邻类) pair，总共有 n_cls * num_neighbor 个 pair, 每一行对应一个类别
    mask = torch.zeros((n_cls, n_cls * num_neighbor), requires_grad=True).int().cuda()  # torch.Size([10, 40])
    # 对每个 pair，把该 pair 中的 “当前类” 设置为 1
    mask[pairs[:, 0], torch.arange(n_cls * num_neighbor)] = 1
    # 对每个 pair 中的 “近邻类” 也对应设置为 1
    mask[pairs[:, 1], torch.arange(n_cls * num_neighbor)] = 1
    # mask:
    #     0  1  2  3  4  5  6  7  8 ....... pair
    # 0   1  1  1  1           1
    # 1      1        1  1  1  1
    # 2                           1  1  1  1
    # 3
    # 4
    # 5
    # 6
    # 7            1     1
    # 8         1           1
    # 9  1            1
    return midpoints, mask  # torch.Size([40, 128]), torch.Size([10, 40])


class KernelDensityEstimator:
    def __init__(self, kernel='vmf', feat_dim=512, bandwidth=3.0, K=200, res=None):  # 768, 2.0, 200, res[0-9]
        self.kernel = kernel  # 'vmf'
        self.bandwidth = bandwidth  # This will be the concentration parameter kappa for vMF, 2.0
        self.feat_dim = feat_dim  # feature dimension, 768
        self.K = K  # k value for nearest neighbors using faiss index, 200
        self.res = res  # faiss.StandardGpuResources()
        self.index = None # faiss index, None
        self.data = None # training data
        self.bessel = (2.0 * math.pi) ** (feat_dim * 0.5) * iv(feat_dim * 0.5 - 1, bandwidth)  # 6.2095436956480065e-37
        # self.epsilon = 6.2095436956480065e-37
        # self.bessel = (2.0 * math.pi) ** (feat_dim * 0.5) * iv(feat_dim * 0.5 - 1, bandwidth) + self.epsilon  # 6.2095436956480065e-37

    def fit(self, data):  # torch.Size([500, 768])
        self.data = data  # torch.Size([500, 768])
        if self.index is None:
            cfg = faiss.GpuIndexFlatConfig()  # 创建 GpuIndexFlat 的配置对象
            cfg.useFloat16 = True  # 启用 FP16 加速（节省内存、提高速度）
            cfg.device = list(range(torch.cuda.device_count()))[0]  # 指定使用哪块 GPU
            # self.res = faiss.StandardGpuResources()
            # faiss.StandardGpuResources(), 768, cfg
            self.index = faiss.GpuIndexFlatL2(self.res, self.feat_dim, cfg)
        else:
            self.index.reset()
        self.index.add(data)  # 将当前类别的队列 data 存入 index, torch.Size([500, 768])

    # 每个离群值和各自 200 个最近 ID 特征的相似度
    def vmf_kernel(self, dot_products):  # torch.Size([50, 200])
        # 下述代码本质上就是在计算: P(z) / max(P(z)) 只不过加了 数值稳定的处理
        # 其实就是 log vMF 密度函数，只不过省略了常数项
        # log_kernel = self.bandwidth * dot_products - math.log(self.bessel)  # torch.Size([50, 200])
        log_kernel = self.bandwidth * dot_products  # torch.Size([50, 200])
        # 数值稳定技巧，用于避免 exp 操作时的 数值溢出
        log_max, _ = torch.max(log_kernel, dim=1, keepdim=True)  # torch.Size([50, 1]), torch.Size([50, 1])
        kernel_values = torch.exp(log_kernel - log_max)  # torch.Size([50, 200]), 消除了 math.log(self.bessel)
        return kernel_values  # torch.Size([50, 200])

    # points: 由 cls 类生成的初始离群值演变而来的离群值 (当前位置的离群值)
    def score_samples(self, points):  # torch.Size([50, 768])
        with torch.no_grad():
            # self.index: torch.Size([500, 768])
            # points: torch.Size([50, 768])
            # self.K: 200
            # 在索引 index 中查找距离 points 最近的 200 个向量，并返回它们的索引和距离
            distances, indices = self.index.search(points, self.K)  # torch.Size([50, 200]), torch.Size([50, 200])

        # self.data: torch.Size([500, 768])
        # indices: torch.Size([50, 200])
        # 从 self.data（500 个样本中），按照 indices 中的索引，提取出 50 组样本，每组 200 个样本，每个样本是 768 维特征向量
        # 50 个离群值, 每个离群值都从 cls 类的 ID 数据中找到了 200 个最近的特征
        retrieved_data = self.data[indices]  # torch.Size([50, 200, 768])
        # retrieved_data[:, -1] 等价于 retrieved_data[:, -1, :]
        # : → 保留全部 50 个样本 (batch)
        # -1 → 200个最近的 ID 数据从近到远排列, 选择排在 200 位最远的 ID 特征
        # : → 保留所有 768 个特征
        far_points = retrieved_data[:, -1]  # torch.Size([50, 768])
        # 计算 当前位置离群值 和 第 cls 个 ID 类中排在 200 位最远的 ID 特征 之间的 欧氏距离
        # torch.Size([50, 768]), torch.Size([50, 768])
        l2_dist = torch.norm(points - far_points, p=2, dim=1)  # torch.Size([50]), 等价于 torch.sqrt(distances[:,-1]) 因为 FAISS 的 L2 距离返回的是平方后的值，而不是开方的值

        density = l2_dist #(torch.sum(kernel_values, dim=1) / self.K) * l2_dist
        return density  # torch.Size([50])

    def pdf(self, points):  # 当前的离群值特征, torch.Size([50, 768])
        with torch.no_grad():
            # self.index: torch.Size([500, 768])
            # points: torch.Size([50, 768])
            # self.K: 200
            # 在索引 index 中查找距离 points 最近的 200 个向量，并返回它们的索引和距离
            distances, indices = self.index.search(points, self.K)  # torch.Size([50, 200]), torch.Size([50, 200])

        # self.data: torch.Size([500, 768])
        # indices: torch.Size([50, 200])
        # 从 self.data（500 个样本中），按照 indices 中的索引，提取出 50 组样本，每组 200 个样本，每个样本是 768 维特征向量
        # 50 个离群值, 每个离群值都从 ID 数据中找到了 200 个最近的特征
        retrieved_data = self.data[indices]  # torch.Size([50, 200, 768])

        # torch.Size([50, 768]) -> torch.Size([50, 1, 768])
        # torch.Size([50, 200, 768]) -> torch.Size([50, 768, 200])
        # 每个离群值和当前 ID 类中各自 200 个最近 ID 特征的相似度
        dot_products = torch.bmm(points.unsqueeze(1), retrieved_data.transpose(1, 2)).squeeze(1)  # torch.Size([50, 200])

        if self.kernel == 'vmf':  # True
            # 计算每个离群值以当前 ID 类中 200 个最近特征为 μ 的 VMF 概率密度
            kernel_values = self.vmf_kernel(dot_products)  # torch.Size([50, 200])
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
        # 属于当前 ID 类的 VMF 概率密度 = 以当前 ID 类中每个特征为 μ 的 VMF 概率密度的平均值 (附录的含义)
        # 每个离群值和各自 200 个最近 ID 特征的概率密度求平均, 得到每个离群值属于当前 ID 类的概率密度
        prob = torch.sum(kernel_values, dim=1) / self.K  # torch.Size([50])

        return prob  # torch.Size([50])


class SphericalHMC(nn.Module):
    # dim: 特征维度, feat_dim, 768
    # L: 迭代次数, num_steps, 3
    # eps: 步长, step_size, 0.1
    # margin: 间隔, margin, 0.1
    # shift: inlier / outlier, 0
    def __init__(self, id_pdf, logpdf, dim, L, eps, margin=0.5, shift=0):  # id_pdf, logpdf, 768, 3, 0.1, 0.1, 0
        super(SphericalHMC, self).__init__()
        self.id_pdf = id_pdf  # id_pdf
        self.logpdf = logpdf  # logprob
        self.dim = dim  # 768
        self.L = L  # 3
        self.eps = eps  # 0.1
        self.margin = margin  # 0.1
        self.shift = shift  # 0
        # 单位矩阵, 大小为 [768, 768], 但不会参与梯度更新，也就是说它是常量
        self.I_a = nn.parameter.Parameter(torch.eye(self.dim), requires_grad=False).cuda()  # torch.Size([768, 768])
        self.threshold_min = None  # None

    @staticmethod
    def batch_square_norm(input):  # torch.Size([50, 768]), 计算 L2范数 (欧几里得范数) 的平方
        return torch.sum(torch.pow(input, 2), -1)  # torch.Size([50])

    @staticmethod
    def batch_outer(input1, input2):  # torch.Size([50, 768]), torch.Size([50, 768]), 计算 z z^T
        # i x 1 * 1 x j -> i x j
        return torch.einsum('bi,bj->bij', input1, input2)  # torch.Size([50, 768, 768])

    # batch_mvp:
    @staticmethod
    def batch_mvp(input1, input2):  # torch.Size([50, 768, 768]), torch.Size([50, 768]), 计算 (z z^T) q
        # i x j * j x 1 -> i x 1 -> i
        return torch.einsum('bij,bj->bi', input1, input2)  # torch.Size([50, 768])

    def H(self, theta_a, v):  # 当前位置的离群值 theta_a: torch.Size([50, 768]), 正交后的动量 v: torch.Size([50, 768])
        u = - self.logpdf(theta_a)  # torch.Size([50]), 50 个离群值的 U = - log (P 当前类)
        # H(z,q) = U(z) + K(q)
        # H(z,q) = -log (P 当前类) + (||q||2^2)/2
        return u + self.batch_square_norm(v) / 2.  # torch.Size([50])

    def sample(self, theta_0):  # 当前位置的离群值, torch.Size([50, 768])
        if self.threshold_min is None:
            # 得到 50 个初始离群值属于 ID 类的最大概率的 log 值
            log_pdf_val = self.id_pdf(theta_0)  # torch.Size([50])
            if self.shift:  # 1: inlier
                self.threshold_min = - log_pdf_val - (self.margin * 0.5)  # torch.Size([50])
            else:  # 0: outlier
                # self.margin: 0.1
                # 应该为 + (self.margin * 0.5), 这就是论文中的公式
                self.threshold_min = - log_pdf_val + (self.margin * 0.5)  # torch.Size([50])

        theta_a = theta_0  # 当前位置的离群值, torch.Size([50, 768])
        # 从标准正态分布中采样动量 v
        # 生成与 当前位置的离群值 形状一样的张量，其中的元素是从标准正态分布 (均值为 0，标准差为 1) 中随机采样的数值
        v = torch.randn_like(theta_a)  # torch.Size([50, 768])
        # 计算 q = q - z z^T q : 将 q (动量) 去掉与 z (离群值) 方向上的分量，留下与 z 垂直的部分
        # 如果 z 是单位向量，那么 (z z^T) q 和 z (z^T q) 结果是 相同的
        # batch_outer: torch.Size([50, 768]), torch.Size([50, 768]) -> torch.Size([50, 768, 768]) 计算 z z^T
        # batch_mvp: torch.Size([50, 768, 768]), torch.Size([50, 768]) -> torch.Size([50, 768]) 计算 (z z^T) q
        v -= SphericalHMC.batch_mvp(SphericalHMC.batch_outer(theta_a, theta_a), v)  # torch.Size([50, 768])
        # 计算 哈密尔顿函数 H(z^1,q^1), 即 系统的总能量
        # 当前位置的离群值 theta_a: torch.Size([50, 768]), 正交后的动量 v: torch.Size([50, 768])
        h_0 = self.H(theta_a, v)  # torch.Size([50])
        for _ in range(self.L):  # range(3), 0-2
            # 创建一个和 theta_a 数值相同、但不参与梯度传播、且是独立内存的张量 theta
            theta = theta_a.detach().clone()  # 当前位置的离群值, torch.Size([50, 768])
            theta.requires_grad = True
            u = -self.logpdf(theta)  # torch.Size([50])
            # 手动调用反向传播, 计算 g = ∂u / ∂θ
            # torch.autograd.grad(outputs, inputs, grad_outputs=None)
            # outputs: 需要进行求导的函数
            # inputs: 需要进行求导的变量
            # grad_outputs: outputs 为向量时，求解梯度，需要将 grad_outputs 设置为全 1 的、与 outputs 形状相同的张量
            # 返回值是元组，取第一个 [0] 就是你要的梯度
            g = grad(u, theta, torch.ones_like(u))[0]  # torch.Size([50, 768])
            theta.requires_grad = False
            # 计算 q = q - ϵ/2 (Id - z z^T) U'(z)
            # batch_outer: torch.Size([50, 768]), torch.Size([50, 768]) -> torch.Size([50, 768, 768]) 计算 z z^T
            # batch_mvp: torch.Size([50, 768, 768]), torch.Size([50, 768]) -> torch.Size([50, 768]) 计算 (I_a - z z^T) U'(z)
            v -= self.eps / 2. * SphericalHMC.batch_mvp(
                self.I_a - SphericalHMC.batch_outer(theta_a, theta), g)  # torch.Size([50, 768])
            v_norm = torch.unsqueeze(torch.norm(v, dim=-1), -1)  # torch.Size([50, 1])
            theta_a_new = theta_a * torch.cos(v_norm * self.eps) + \
                v / v_norm * torch.sin(v_norm * self.eps)  # 下一位置离群值 z^l+1, torch.Size([50, 768])
            v = -theta_a * v_norm * torch.sin(v_norm * self.eps) + \
                v * torch.cos(v_norm * self.eps)  # torch.Size([50, 768])
            theta_a = theta_a_new  # 下一位置离群值 z^l+1, torch.Size([50, 768])
            theta = theta_a.detach().clone()  # 下一位置离群值 z^l+1, torch.Size([50, 768])
            theta.requires_grad = True
            u = -self.logpdf(theta)  # torch.Size([50])
            g = grad(u, theta, torch.ones_like(u))[0]  # torch.Size([50, 768])
            theta.requires_grad = False
            # 计算 q = q - ϵ/2 (Id - z z^T) U'(z)
            # batch_outer: torch.Size([50, 768]), torch.Size([50, 768]) -> torch.Size([50, 768, 768]) 计算 z z^T
            # batch_mvp: torch.Size([50, 768, 768]), torch.Size([50, 768]) -> torch.Size([50, 768]) 计算 (I_a - z z^T) U'(z)
            v -= self.eps / 2. * SphericalHMC.batch_mvp(
                self.I_a - SphericalHMC.batch_outer(theta_a, theta), g)  # 下一位置动量 q^l+1, torch.Size([50, 768])
        # 计算 H(z^L+1, q^L+1)
        # z^L+1: torch.Size([50, 768]), q^L+1: torch.Size([50, 768])
        h = self.H(theta_a, v)  # torch.Size([50])
        # Metropolis-Hastings 规则: A = min(1, exp(H_old - H_new)), 能量降低 必定接受, 能量增加, 有概率接受
        # H(z^L+1, q^L+1) - H(z^1,q^1)
        delta_H = h - h_0  # torch.Size([50])
        # torch.Size([50, 768])
        # 得到 50 个最终离群值 z^L+1 属于 ID 类的最大概率的 log 值, 再取负值
        id_pdfs = -self.id_pdf(theta)

        if self.shift:  # 1: inlier
            mask = torch.unsqueeze(torch.logical_and(  # mask: torch.Size([50, 1])
                torch.rand_like(delta_H) < torch.exp(-delta_H),
                # torch.Size([50]), torch.Size([50]) -> torch.Size([50])
                id_pdfs < self.threshold_min  # torch.Size([50]), torch.Size([50]) -> torch.Size([50])
            ), -1)
        else:  # 0: outlier
            # torch.logical_and(A, B): 两个条件都满足才为 True
            # 1、生成值在区间 [0.0, 1.0) 之间 torch.Size([40]) 的 tensor < exp(-delta_H)
            # ---> h 降低, delta_H 为负数, -delta_H 为正数, exp(+) > 1, 必然满足 [0.0, 1.0) < exp(-delta_H) 1.3
            #      h 增加, delta_H 为正数, -delta_H 为负数, exp(-) < 1, 可能满足 [0.0, 1.0) < exp(-delta_H) 0.8
            # 2、40 个最终离群值 z^L+1 属于 ID 类的最大概率的 -log 值 > 40 个初始离群值属于 ID 类的最大概率的 -log 值
            # ---> 最终离群值属于 ID 类的最大概率 < 初始离群值属于 ID 类的最大概率
            mask = torch.unsqueeze(torch.logical_and(  # mask: torch.Size([50, 1])
                torch.rand_like(delta_H) < torch.exp(-delta_H),  # torch.Size([50]), torch.Size([50]) -> torch.Size([50])
                id_pdfs > self.threshold_min  # torch.Size([50]), torch.Size([50]) -> torch.Size([50])
            ), -1)
        # 根据 mask, 决定哪些新的离群值 theta 被接受, 其余仍保留初始离群值 theta_0
        # 最终的离群值 * mask + 初始离群值 * ~mask -> 组合成最终采样的离群值
        # torch.Size([50, 768]), torch.Size([50, 1]), torch.Size([50, 768]), torch.Size([50, 1])
        return theta * mask + theta_0 * ~mask  # torch.Size([50, 768])

# init_points: torch.Size([50, 768]), 边界附近的离群值
# num_samples: 2
# feat_dim: 768
# kdes: [10 个 KernelDensityEstimator]
# cls: init_points 是由哪个 ID 类生成的离群值, 0
# num_steps: 3
# step_size: 0.1
# margin: 0.1
# shift: 0 -- outlier / 1 -- inlier
def generate_Ham_outliers(init_points, num_samples, feat_dim, kdes, cls, num_steps=3, step_size=0.1, margin=0.5, shift=0):

    def logprob(x):  # 当前位置的离群值: torch.Size([50, 768])
        # 50 个离群值的 目标分布 即 (P 当前类) 的结果
        result = torch.zeros(len(x), requires_grad=True).float().cuda()  # torch.Size([50])

        if shift:  # 1: inlier
            probs = kdes[cls].score_samples(x)  # torch.Size([50])
            # (1) 倒数映射: P = 1 / (d + epsilon)
            # epsilon = 1e-8
            # probs = 1.0 / (probs + epsilon)  # torch.Size([50])
            # (2) 指数衰减: P = exp(-d)
            probs = torch.exp(-probs)  # torch.Size([50])

        else:  # 0: outlier
            # 当前位置离群值 x 是由 cls 类生成而来, 因此和 cls 类中的第 200 近邻计算距离
            # 当前位置离群值 和 cls 类中排在 200 位最远的 ID 特征 之间的 欧氏距离
            probs = kdes[cls].score_samples(x)  # torch.Size([50])

        result.add_(probs)  # torch.Size([50]), 50 个离群值的 (P 当前类) 的结果
        return result.log()  # torch.Size([50]), 50 个离群值的 log (P 当前类)

    def id_pdf(x):  # 当前的离群值特征, torch.Size([50, 768])
        # kdes 存储的是 10 个 ID 类的 KernelDensityEstimator 对象, x 则为当前的离群值特征
        # 每个 ID 类都和 x 计算一下概率密度, 每一行表示每个离群值属于 10 个 ID 类的 VMF 概率密度
        pdfs = torch.stack([kde.pdf(x) for kde in kdes], dim=1)  # torch.Size([50, 10])
        # 将每个离群值属于 10 个 ID 类的 VMF 概率密度转化为概率分布
        softmax_pdf = F.softmax(pdfs, dim=1)  # torch.Size([50, 10])
        # 从每个离群值属于 10 个 ID 类的概率分布中找一个最大的, 得到 50 个离群值属于 ID 类的最大概率
        max_pdf = torch.max(softmax_pdf, dim=1).values  # torch.Size([50])
        return max_pdf.log()  # torch.Size([50])

    # <function> id_pdf, <function> logprob, 768, 3, 0.1, 0.1, 0
    shmc = SphericalHMC(id_pdf, logprob, feat_dim, num_steps, step_size, margin, shift)
    samples = []  # 存放生成的离群值
    cur_positions = init_points  # torch.Size([50, 768])
    cnt = 0
    for _ in range(num_samples):  # range(2) 0 - 1
        # 最终的离群值 * mask + 初始离群值 * ~mask -> 组合成最终采样的离群值
        new_positions = shmc.sample(cur_positions)  # torch.Size([50, 768])
        if not torch.isnan(new_positions).any():  # True
            cnt += 1
            # 让 最终采样的离群值 充当下一轮的 初始离群值
            cur_positions = new_positions  # torch.Size([50, 768])
        samples.append(cur_positions)  # len = 2, [torch.Size([50, 768]), torch.Size([50, 768])]

    # samples: [torch.Size([50, 768]), torch.Size([50, 768])]
    result = torch.cat(samples, dim=0)  # torch.Size([100, 768])
    return result  # torch.Size([100, 768])