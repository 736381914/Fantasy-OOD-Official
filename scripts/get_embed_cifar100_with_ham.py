# -*- coding: utf-8 -*-
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import faiss
res = faiss.StandardGpuResources()
KNN_index = faiss.GpuIndexFlatL2(res, 768)
from torch.distributions import MultivariateNormal
from KNN import generate_outliers
from HamOS_cpu import KernelDensityEstimator, SphericalHMC, generate_Ham_outliers


parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# generation hyperparameters.
parser.add_argument('--shift', type=int, default=0)
parser.add_argument('--gaussian_mag_ood_det', type=float, default=0.07)
parser.add_argument('--gaussian_mag_ood_gene', type=float, default=0.01)
parser.add_argument('--K_in_knn', type=int, default=300)
parser.add_argument('--ood_det_select', type=int, default=50)
parser.add_argument('--ood_gene_select', type=int, default=100)
parser.add_argument('--gpu', type=int, default=0)
# HamOS hyperparameters.
parser.add_argument('--feat_dim', type=int, default=768)
parser.add_argument('--hamos_K', type=int, default=200)
parser.add_argument('--hamos_select', type=int, default=2)
parser.add_argument('--bandwidth', type=float, default=2.0)
parser.add_argument('--leapfrog', type=int, default=3)
parser.add_argument('--step_size', type=float, default=0.1)
parser.add_argument('--step_size_inlier', type=float, default=0.0001)
parser.add_argument('--margin', type=float, default=0.01)
parser.add_argument('--margin_inlier', type=float, default=0.0)
parser.add_argument('--num_neighbor', type=int, default=4)
args = parser.parse_args()


anchor = torch.from_numpy(np.load('./token_embed_c100.npy')).cuda()
num_classes = 100
sum_temp = 0

data_dict = torch.from_numpy(np.load('./id_feat_cifar100_199epoch.npy')).cuda()

for index in range(num_classes):
    sum_temp += 500  # number_dict[index]

# 768, 2.0, 200, res[0-99]
kde = [KernelDensityEstimator(feat_dim=args.feat_dim, bandwidth=args.bandwidth, K=args.hamos_K) for i in range(num_classes)]
# 初始化 100 个类别 kde 中的数据
for cls_idx in range(num_classes):  # range(0, 100)
    # KernelDensityEstimator, torch.Size([500, 768])
    kde[cls_idx].fit(F.normalize(data_dict[cls_idx].cpu(), p=2, dim=1))

# breakpoint()
if sum_temp == num_classes * 500:
    for index in range(num_classes):
        ID = F.normalize(data_dict[index], p=2, dim=1)
        if 1:
            if args.shift:  # inlier
                print(index)
                for index1 in range(100):
                    new_dis = MultivariateNormal(torch.zeros(768).cuda(), torch.eye(768).cuda())
                    negative_samples = new_dis.rsample((1500,))
                    sample_point1, boundary_point = generate_outliers(ID,
                                                                      input_index=KNN_index,
                                                                      negative_samples=negative_samples,
                                                                      ID_points_num=1,
                                                                      K=args.K_in_knn,
                                                                      select=args.ood_gene_select,
                                                                      cov_mat=args.gaussian_mag_ood_gene,
                                                                      sampling_ratio=1.0,
                                                                      pic_nums=100,
                                                                      depth=768, shift=1)

                    # HMC 初始值, torch.Size([50, 768])
                    init_points = sample_point1[
                        np.random.choice(sample_point1.shape[0], int(sample_point1.shape[0] // 2), replace=False)]

                    ham_samples = generate_Ham_outliers(
                        init_points=init_points,  # torch.Size([50, 768])
                        num_samples=args.hamos_select,  # 2
                        feat_dim=args.feat_dim,  # 768
                        kdes=kde,  # 100 个 KernelDensityEstimator
                        cls=index,  # 0
                        num_steps=args.leapfrog,  # 3
                        step_size=args.step_size_inlier,  # 0.0001
                        margin=args.margin_inlier,  # 0
                        shift=1  # 1
                    )  # torch.Size([100, 768])

                    if index1 == 0:
                        sample_npos = sample_point1  # torch.Size([100, 768])
                        sample_homos = ham_samples  # torch.Size([100, 768])
                    else:
                        sample_npos = torch.cat([sample_npos, sample_point1], 0)  # torch.Size([10000, 768])
                        sample_homos = torch.cat([sample_homos, ham_samples], 0)  # torch.Size([10000, 768])

            else:  # outlier
                print(index)
                for index1 in range(100):
                    new_dis = MultivariateNormal(torch.zeros(768).cuda(), torch.eye(768).cuda())
                    negative_samples = new_dis.rsample((1500,))
                    sample_point1, boundary_point = generate_outliers(ID,
                                                                      input_index=KNN_index,
                                                                      negative_samples=negative_samples,
                                                                      ID_points_num=2,
                                                                      K=args.K_in_knn,
                                                                      select=args.ood_det_select,
                                                                      cov_mat=args.gaussian_mag_ood_det, sampling_ratio=1.0, pic_nums=50,
                                                                      depth=768, shift=0)
                    # HMC 初始值, torch.Size([50, 768])
                    init_points = sample_point1[
                        np.random.choice(sample_point1.shape[0], int(sample_point1.shape[0] // 2), replace=False)]

                    ham_samples = generate_Ham_outliers(
                        init_points=init_points,  # torch.Size([50, 768])
                        num_samples=args.hamos_select,  # 2
                        feat_dim=args.feat_dim,  # 768
                        kdes=kde,  # 100 个 KernelDensityEstimator
                        cls=index,  # 0
                        num_steps=args.leapfrog,  # 3
                        step_size=args.step_size,  # 0.1
                        margin=args.margin,  # 0.1
                        shift=0  # 0
                    )  # torch.Size([100, 768])

                    if index1 == 0:
                        sample_npos = sample_point1  # torch.Size([100, 768])
                        sample_homos = ham_samples  # torch.Size([100, 768])
                    else:
                        # index类别中的 50 个边界, 每一个边界都在周围采样了200个离群值
                        sample_npos = torch.cat([sample_npos, sample_point1], 0)  # torch.Size([10000, 768])
                        sample_homos = torch.cat([sample_homos, ham_samples], 0)  # torch.Size([10000, 768])

            # 合并 NPOS 与 HamOS 离群值
            sample_point = torch.cat([sample_npos, sample_homos], 0)  # torch.Size([20000, 768])

            if index == 0:
                ood_samples = [sample_point * anchor[index].norm()]
            else:
                ood_samples.append(sample_point * anchor[index].norm())

print("ood_samples: ", torch.stack(ood_samples).shape)

if args.shift:
    np.save \
        ('./cifar100_inlier_npos_embed' + '_noise_' + str(args.gaussian_mag_ood_gene) + '_select_' + str(
        args.ood_gene_select) + '_KNN_' + str(args.K_in_knn) + '.npy', torch.stack(ood_samples).cpu().data.numpy())
else:
    np.save \
        ('./cifar100_outlier_npos_embed' + '_noise_' + str(args.gaussian_mag_ood_det) + '_select_' + str(
        args.ood_det_select) + '_KNN_' + str(args.K_in_knn) + '.npy', torch.stack(ood_samples).cpu().data.numpy())
